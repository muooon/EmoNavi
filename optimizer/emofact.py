import torch
from torch.optim import Optimizer
import math
from collections import deque

"""
EmoFact v3.1 (251125) shadow-system v3.0 -effect NoN -moment v3.0
(v1.0)AMP対応完了(250725) p.data -> p 修正済み／低精度量子化への基本対応／低精度補償は別
(v2.0)shadow-system 微調整／３段階補正を連続的に滑らかに／派生版では以下の切替も可能
optimizer 指定の際に True / False で shadow を切替できる(現在 False)
(v3.0)emosens shadow-effect v1.0 反映した動的学習率と shadow-system 切替をデフォルト化
(v3.1)通常未使用の shadow 更新速度 (lerp) を倍化し信頼度で動的制御／coeff 活用
動的学習率や感情スカラー値など TensorBoard 連携可 (現在 writer=None)／外部設定必要
全体の効率化や可読性を向上(emaやスカラーの多重処理を省く等、動的学習率のスケールや状態の見直し等、含む)
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
                writer=None):

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._init_lr = lr
        self.should_stop = False # 停止フラグの初期化
        self.use_shadow = use_shadow # 🔸shadow 使用フラグを保存
        self.writer = writer # 動的学習率や感情スカラー等を渡す

    # 感情EMA更新(緊張と安静)
    def _update_ema(self, state, loss_val):
        ema = state.setdefault('ema', {})
        ema['short'] = 0.3 * loss_val + 0.7 * ema.get('short', loss_val)
        ema['long'] = 0.01 * loss_val + 0.99 * ema.get('long', loss_val)
        return ema

    # 感情スカラー値生成(EMA差分、滑らかな非線形スカラー、tanh (diff) は ±1.0 で有界性)
    # 係数"1"：ema差分 のスケール調整処理に活用(感度調節係数)／通常は１(タスクに応じ調整可(非推奨))
    # scale_base：Loss値とema値の乖離を修正(分母 ema(long) 「改善率」共通化/loss種に非依存)
    # 1e-5(デフォルト)／1e-6(感度向上)／1e-4(安定性向上)：分母を０にせず安定させる
    def _compute_scalar(self, ema):
        scale_base = max(ema['long'], 1e-5)
        diff = (ema['long'] - ema['short']) / scale_base
        return math.tanh(1 * diff)

    # 急変時は論文通りの抑制則/悪化時は減速/改善時は加速/微動時は無介入で収束を安定させる
    def _decide_coeff(self, scalar):
        if abs(scalar) > 0.625:
            return 1.0 - abs(scalar)    # 急変｜強抑制
        elif scalar > 0.125:
            return 1.0 + scalar         # 改善｜加速
        elif scalar < -0.125:
            return 1.0 + scalar         # 悪化｜減速
        else:
            return 1.0                  # 微動｜無介入

    # (重要)現在は shadow-effect を参考に得た動的フィルタ効果の近似により use_shadow=False です
    # しかし全機能は shadow なしで全て成立します／コード確認時はこの shadow を考慮外として無視してください

    # Shadow混合比 ３段階構成 タスクに応じ調整可、以下を参考に 開始値・範囲量･変化幅を調整
    # 参考1：scalar>±0.6 を "return 開始値 + ((scalar) - 0.6(範囲)) / 範囲量 * 変化幅"
    # 参考2：scalar>±0.1 を "return 開始値 + ((scalar) - 0.1(範囲)) / 範囲量 * 変化幅"
    # return 開始値 + ((scalar) - 閾値) / 範囲量 * 変化幅 です(上記の値は感情スカラーを返すだけ)
    def _decide_ratio(self, scalar):
        if not self.use_shadow:
            return 0.0 # 🔸use_shadow が False の場合は常に比率を 0 にする
        if abs(scalar) > 0.9:
            return 0.9 + ((scalar) - 0.9) / 0.4 * 0.4 # これはスカラーそのまま返す参考例
        elif abs(scalar) > 0.3:
            return -0.1 # return<0 の場合は leap 専用(書き戻しはしないが履歴更新のみ)
        return 0.0

    # 損失取得(損失値 loss_val を数値化、感情判定に使用、存在しないパラメータ(更新不要)はスキップ)
    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        loss_val = loss.item() if loss is not None else 0.0

        # EMA更新・スカラー生成(EMA差分からスカラーを生成しスパイク比率等を決定)
        ema = self._update_ema(self.state, loss_val)
        scalar = self._compute_scalar(ema)
        coeff = self._decide_coeff(scalar)
        ratio = self._decide_ratio(scalar)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # 動的学習率の符号考慮補正導入により shadow 形成を最大15%程度とし信頼度で調整
                # shadow：必要時のみ(スパイクp部分に現在値を最大15%程度追従させる動的履歴更新)
                if self.use_shadow:
                    if ratio > 0: # 書き戻し(急変時の強い抑制と弱めの履歴更新)
                        p.mul_(1 - ratio).add_(state['shadow'], alpha=ratio)
                    if 'shadow' not in state: # shadow=False (デフォルト)
                        state['shadow'] = p.clone()
                    else: # ratio<0 の場合は leap 専用(書き戻しはしないが履歴更新のみ)
                        leap_ratio = 0.1 if ratio < 0 else 0.1 * coeff
                        state['shadow'].lerp_(p, leap_ratio)

                # 上記 shadow の説明：スカラー生成：短期と長期EMAの差分から信号を得る(高ぶりの強さ)
                # 混合比率：スカラーが閾値を超える場合にのみ計算される(信頼できる感情信号かどうかの選別)
                # スカラー値が小さい場合は ratio = 0 となり、shadow混合は行われない
                # 信頼できる強い差分のときのみ感情機構が発動する(暗黙的信頼度)
                # 信頼度 coeff による動的調整も加味する(明示的信頼度) 

                # --- 勾配補正ロジック ---
                # 行列の形状が2次元以上の場合、分散情報ベースのAB近似を使用
                if grad.dim() >= 2:
                    # 行と列の2乗平均を計算 (分散の軽量な近似)
                    r_sq = torch.mean(grad * grad, dim=tuple(range(1, grad.dim())), keepdim=True).add_(group['eps'])
                    c_sq = torch.mean(grad * grad, dim=0, keepdim=True).add_(group['eps'])

                    # 分散情報から勾配の近似行列を生成
                    # AB行列として見立てたものを直接生成し更新項を計算する
                    # A = sqrt(r_sq), B = sqrt(c_sq) AB行列の近似を再現しEMAで平滑化する
                    beta1, beta2 = group['betas']
                    state.setdefault('exp_avg_r', torch.zeros_like(r_sq)).mul_(beta1).add_(torch.sqrt(r_sq), alpha=1 - beta1)
                    state.setdefault('exp_avg_c', torch.zeros_like(c_sq)).mul_(beta1).add_(torch.sqrt(c_sq), alpha=1 - beta1)

                    # 再構築した近似勾配の平方根の積で正規化
                    denom = torch.sqrt(state['exp_avg_r'] * state['exp_avg_c']).add_(group['eps'])

                    # 最終的な更新項を計算
                    update_term = grad / denom

                # 1次元(ベクトル)の勾配補正
                else:
                    beta1, beta2 = group['betas']
                    exp_avg_sq = state.setdefault('exp_avg_sq', torch.zeros_like(p))
                    exp_avg_sq.mul_(beta1).addcmul_(grad, grad, value=(1 - beta2))
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    update_term = grad / denom

                # 最終的なパラメータ更新 (decoupled weight decayも適用)
                p.add_(p, alpha=-group['weight_decay'] * group['lr'])
                p.add_(update_term, alpha=-group['lr'] * coeff)

        # 感情機構の発火が収まり"十分に安定"していることを外部伝達できる(自動停止ロジックではない)
        # Early Stop用 scalar 記録(バッファ共通で管理/最大32件保持/動静評価)
        hist = self.state.setdefault('scalar_hist', deque(maxlen=32))
        hist.append(scalar)

        # Early Stop判断(静けさの合図)
        # 32ステップ分のスカラー値の静かな条件を満たした時"フラグ" should_stop = True になるだけ
        if len(hist) >= 32:
            avg_abs = sum(abs(s) for s in hist) / len(hist)
            mean = sum(hist) / len(hist)
            var = sum((s - mean)**2 for s in hist) / len(hist)
            if avg_abs < 0.05 and var < 0.005:
                self.should_stop = True # 💡 外部からこれを見て判断可

        # TensorBoardへの記録（step関数の末尾に追加）
        if hasattr(self, 'writer') and self.writer is not None:
            self._step_count = getattr(self, "_step_count", 0) + 1
            for i, group in enumerate(self.param_groups):
                self.writer.add_scalar(f"Lr{i}", group['lr'], self._step_count)
            self.writer.add_scalar("eScalar", scalar, self._step_count)

        return loss

"""
 https://github.com/muooon/EmoNavi
 Fact is inspired by Adafactor, and emoairy, 
 and its VRAM-friendly design is something everyone loves.
"""