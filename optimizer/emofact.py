import torch
from torch.optim import Optimizer
import math
from collections import deque

"""
EmoFact v4.0 (251105) shadow-system v3.0 -effect NoN -moment v3.0
AMP対応完了(250725) p.data -> p 修正済み／低精度量子化への基本対応／低精度補償は別
emosens shadow-effect v1.0 反映した動的学習率と shadow-system 修正／３段階補正
optimizer 指定の際に True / False で shadow を切替できる(現在 False)
感情moment v3.0 とし、 動的感情スカラー と trust_coeff 追加／自己参照型反応学習
トラウマ的反応と慣れによる鈍化で安定性向上(暗黙的な v1.0 改良し安全性向上)
通常未使用の shadow の更新速度 (lerp) を倍化し信頼度で動的制御／trust_coeff の活用
optimizer 指定の際に True / False で trust を切替できる(現在 True)
ノルムベース学習率調整機構を追加 (max_norm=1.0 / min_lr=1e-6)／max_norm=0.0で無効化
動的学習率と感情スカラー値の履歴をTensorBoard連携 (writer=None)
"""

class EmoFact(Optimizer):
    # クラス定義＆初期化
    def __init__(
        self, 
        params, 
        lr=1e-3, 
        betas=(0.9, 0.999), 
        eps=1e-8, 
        weight_decay=0.01,
        use_shadow: bool = False, 
        use_trust: bool = True, 
        max_norm=1.0, 
        min_lr=1e-6, 
        writer=None,
    ):
        defaults = dict(
            lr=lr, 
            betas=betas, 
            eps=eps, 
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        
        self._init_lr = lr 
        self.should_stop = False # 停止フラグの初期化
        self.use_shadow = use_shadow # 🔸shadowの使用フラグを保存
        self.use_trust = use_trust # 2次moment信頼度調整
        self.max_norm = max_norm  # ここで max_norm を保存
        self.avg_norm = 0.0 # max_norm=0.0 で無効化
        self.min_lr = min_lr # 1e-5～5e-7 程度がおすすめ
        self.writer = writer #動的学習率と感情スカラーを渡す

    # 感情EMA更新(緊張と安静) short：3.3step、long：100step
    def _update_ema(self, state, loss_val):
        ema = state.setdefault('ema', {})
        ema['short'] = 0.3 * loss_val + 0.7 * ema.get('short', loss_val)
        ema['long'] = 0.01 * loss_val + 0.99 * ema.get('long', loss_val)
        return ema

    # 感情スカラー値生成(EMA差分、滑らかな非線形スカラー、tanh 5 * diff で鋭敏さ強調)
    # 感情スカラーの慣性(過敏性調節／安定安全性を確保し高学習率による最大最速進行を目指す)
    def _compute_scalar(self, ema, prev_scalar=0.0):
        diff = ema['short'] - ema['long']
        return math.tanh(5 * abs(prev_scalar) * diff)

    # (重要)現在は shadow-effect を参考に得た動的フィルタ効果の近似により use_shadow=False です
    # よって挙動や結果は shadow 考慮外 で成立します－コード確認時はこの shadow を無視してください

    # Shadow混合比 ３段階構成 タスクに応じ調整可、以下を参考に 開始値・範囲量･変化幅を調整
    # 修正1：scalar>±0.6 を "return 開始値 + (abs(scalar) - 0.6(範囲)) / 範囲量 * 変化幅"
    # 修正2：scalar>±0.1 を "return 開始値 + (abs(scalar) - 0.1(範囲)) / 範囲量 * 変化幅"
    def _decide_ratio(self, scalar):
        if not self.use_shadow:
            return 0.0 # 🔸use_shadow が False の場合は常に比率を 0 にする
        if abs(scalar) > 0.6:
            return 0.6 + (abs(scalar) - 0.6) / 0.4 * 0.4 # 元 return 0.7 + 0.2 * scalar
        elif abs(scalar) > 0.1:
            return 0.1 + (abs(scalar) - 0.1) / 0.5 * 0.5 # 元 return 0.3
        return 0.0

    # 損失取得(損失値 loss_val を数値化、感情判定に使用、存在しないパラメータ(更新不要)はスキップ)
    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        loss_val = loss.item() if loss is not None else 0.0
        total_norm = 0.0

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                total_norm += grad.norm(2).item() ** 2

                # 感情EMA更新・スカラー生成 (既存ロジックを維持)
                ema = self._update_ema(state, loss_val)
                prev_scalar = self.state[p].get('scalar', 0.0)
                scalar = self._compute_scalar(ema, prev_scalar)
                state['scalar'] = scalar # 更新して保存
                ratio = self._decide_ratio(scalar)
                trust_coeff = 1.0 - abs(scalar) * abs(scalar) if self.use_trust else 1.0

                # shadow_param：必要時のみ更新 (スパイク部分に現在値を10%ずつ追従させる動的履歴)
                if self.use_shadow and ratio > 0:
                    if 'shadow' not in state:
                        state['shadow'] = p.clone()
                    else:
                        p.mul_(1 - ratio).add_(state['shadow'], alpha=ratio)
                        state['shadow'].lerp_(p, 0.1 * trust_coeff)
                
                # --- 勾配補正ロジック ---
                # 行列の形状が2次元以上の場合、分散情報ベースのAB近似を使用
                if grad.dim() >= 2:
                    # 行と列の2乗平均を計算 (分散の軽量な近似)
                    r_sq = torch.mean(grad * grad, dim=tuple(range(1, grad.dim())), keepdim=True).add_(group['eps'])
                    c_sq = torch.mean(grad * grad, dim=0, keepdim=True).add_(group['eps'])

                    # 分散情報から勾配の近似行列を生成
                    # AB行列として見立てたものを直接生成し更新項を計算する
                    # A = sqrt(r_sq), B = sqrt(c_sq) とすることでAB行列の近似を再現
                    # これをEMAで平滑化する
                    beta1, beta2 = group['betas']
                    
                    state.setdefault('exp_avg_r', torch.zeros_like(r_sq)).mul_(beta1).add_(torch.sqrt(r_sq), alpha=(1 - beta1) * trust_coeff)
                    state.setdefault('exp_avg_c', torch.zeros_like(c_sq)).mul_(beta1).add_(torch.sqrt(c_sq), alpha=(1 - beta1) * trust_coeff)
                    
                    # 再構築した近似勾配の平方根の積で正規化
                    # これにより2次モーメントのような役割を果たす
                    denom = torch.sqrt(state['exp_avg_r'] * state['exp_avg_c']).add_(group['eps'])
                    
                    # 最終的な更新項を計算
                    update_term = grad / denom

                # 1次元(ベクトル)の勾配補正(decoupled weight decay 構造に近い)
                else:
                    exp_avg = state.setdefault('exp_avg', torch.zeros_like(p))
                    exp_avg_sq = state.setdefault('exp_avg_sq', torch.zeros_like(p))
                    beta1, beta2 = group['betas']
                    exp_avg.mul_(beta1).add_(grad, alpha=(1 - beta1) * trust_coeff)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    update_term = exp_avg / denom

                # 最終的なパラメータ更新 (decoupled weight decayも適用)
                # boost 値は初期学習率(目標学習率)への加減速倍率になる(簡易テスト確認済)
                # (重要)理論上はいくらでも上限拡張可能(初期爆発や収束安定性に要注意)
                p.add_(p, alpha=-group['weight_decay'] * group['lr'])
                p.add_(update_term, alpha=-group['lr'] * (1 - abs(scalar)))

                # --- Early Stop ロジック (既存ロジックを維持) ---
                # 感情機構の発火が収まり"十分に安定"していることを外部伝達できる(自動停止ロジックではない)
                # Early Stop用 scalar記録(バッファ共通で管理/最大32件保持/動静評価)
                # この部分は p.state ではなく self.state にアクセスする
                hist = self.state.setdefault('scalar_hist', deque(maxlen=32))
                hist.append(scalar)

        # Early Stop判断(静けさの合図)
        # 32ステップ分のスカラー値の静かな条件を満たした時"フラグ" should_stop = True になるだけ
        if len(self.state['scalar_hist']) >= 32:
            avg_abs = sum(abs(s) for s in hist) / len(hist)
            mean = sum(hist) / len(hist)
            std = sum((s - mean)**2 for s in hist) / len(hist)
            if avg_abs < 0.05 and std < 0.005:
                self.should_stop = True # 💡 外部からこれを見て判断可

        # 学習率調整（例：ノルムが大きすぎたら減衰）max_norm=0.0 で無効化
        # 例：avg_norm が 2倍なら 0.9025 に減衰 / in-placeで減衰
        if self.max_norm > 0.0:
            total_norm = total_norm ** 0.5
            self.avg_norm = 0.9 * self.avg_norm + 0.1 * total_norm

            for group in self.param_groups:
                if self.avg_norm > self.max_norm:
                    excess = self.avg_norm / self.max_norm
                    decay_factor = 0.95 ** excess 
                    group['lr'] = max(group['lr'] * decay_factor, self.min_lr)
                
        # TensorBoardへの記録（step関数の末尾に追加）
        if hasattr(self, 'writer') and self.writer is not None:
            self._step_count = getattr(self, "_step_count", 0) + 1
            for i, group in enumerate(self.param_groups):
                self.writer.add_scalar(f"LR/Grp_{i}", group['lr'], self._step_count)
            self.writer.add_scalar("eScalar", scalar, self._step_count)
            self.writer.add_scalar("avgnorm", self.avg_norm, self._step_count)
            self.writer.add_scalar("tcoeff", trust_coeff, self._step_count)

        return loss

"""
 https://github.com/muooon/EmoNavi
 Fact is inspired by Adafactor, and emoairy, 
 and its VRAM-friendly design is something everyone loves.
"""