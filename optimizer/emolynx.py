import torch
from torch.optim import Optimizer
import math
from collections import deque
from typing import Tuple, Callable, Union

"""
EmoLynx v4.0 (251105) shadow-system v3.0 -effect NoN -moment v3.0
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

# Helper function (Lynx)
def exists(val):
    return val is not None

class EmoLynx(Optimizer):
    # クラス定義＆初期化 lynx用ベータ･互換性の追加(lynx用beta1･beta2)
    def __init__(
        self, 
        params: Union[list, torch.nn.Module], 
        lr=1e-3, 
        betas=(0.9, 0.99), 
        eps=1e-8, 
        weight_decay=0.01, 
        decoupled_weight_decay: bool = False, 
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
        # lynxに応じてウェイト減衰のため保存
        
        self._init_lr = lr
        self.should_stop = False # 停止フラグの初期化
        self.decoupled_wd = decoupled_weight_decay
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
    def step(self, closure: Callable | None = None): # クロージャの型ヒントを追加
        loss = None
        if exists(closure): # 一貫性のためにexistsヘルパーを使う
            with torch.enable_grad():
                loss = closure()
        loss_val = loss.item() if loss is not None else 0.0
        total_norm = 0.0

        for group in self.param_groups:
            # リンクス共通パラメータ抽出
            lr, wd, beta1, beta2 = group['lr'], group['weight_decay'], *group['betas']
            
            # ウェイト減衰の処理を分離 (from lynx)
            _wd_actual = wd
            if self.decoupled_wd:
                _wd_actual /= self._init_lr # 非連結時ウェイト減衰調整

            for p in filter(lambda p: exists(p.grad), group['params']): # PGチェックにフィルタ

                grad = p.grad # PG直接使用(計算に".data"不要)
                state = self.state[p]
                total_norm += grad.norm(2).item() ** 2

                # EMA更新・スカラー生成(EMA差分からスカラーを生成しスパイク比率を決定)
                ema = self._update_ema(state, loss_val)
                prev_scalar = self.state[p].get('scalar', 0.0)
                scalar = self._compute_scalar(ema, prev_scalar)
                state['scalar'] = scalar # 更新して保存
                ratio = self._decide_ratio(scalar)
                trust_coeff = 1.0 - abs(scalar) * abs(scalar) if self.use_trust else 1.0

                # shadow_param：必要時のみ更新(スパイク部分に現在値を10%ずつ追従させる動的履歴)
                if self.use_shadow and ratio > 0:
                    if 'shadow' not in state:
                        state['shadow'] = p.clone()
                    else:
                        p.mul_(1 - ratio).add_(state['shadow'], alpha=ratio) 
                        state['shadow'].lerp_(p, 0.1 * trust_coeff)
                        # lynx更新前 p で shadow 更新(現在値を10%ずつ追従)
                        # p.mul_(1 - ratio).add_(state['shadow'], alpha=ratio) 
                        # EmoNavi: p = p * (1-ratio) + shadow * ratio

                # --- Start Lynx Gradient Update Logic ---
                
                # lynx初期化(exp_avg_sq)
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p)
                exp_avg = state['exp_avg']

                # Stepweight decay (from lynx): p = p * (1 - lr * wd)
                # decoupled_wd 考慮 _wd_actual 使用(EmoNaviのwdは最後に適用)
                p.mul_(1. - lr * _wd_actual)

                # 勾配ブレンド v3.0 信頼度修正
                # m_t = beta1 * exp_avg_prev + (1 - beta1) * grad
                # blended_grad = grad.mul(1. - beta1).add_(exp_avg, alpha=beta1) ※ 元コード
                blended_grad = grad.mul(1. - beta1).add_(exp_avg, alpha=beta1 * trust_coeff)

                # 最終的なパラメータ更新 (p: p = p - lr * sign(blended_grad))
                p.add_(blended_grad.sign_(), alpha = -lr * (1 - abs(scalar)))

                # exp_avg = beta2 * exp_avg + (1 - beta2) * grad
                exp_avg.mul_(beta2).add_(grad, alpha = 1. - beta2)

                # --- End Lynx Gradient Update Logic ---

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
 Lynx was developed with inspiration from Lion, Tiger, and emocats, 
 which we deeply respect for their lightweight and intelligent design.  
 Lynx also integrates EmoNAVI to enhance its capabilities.
"""