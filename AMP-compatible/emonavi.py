import torch
from torch.optim import Optimizer
import math

"""
EmoNavi v2.0 (250815) shadow-system v2.0
AMP対応完了(202507) p.data -> p 修正済み
emosens shadow-effect v1.0 反映 shadow-system 修正
"""

class EmoNavi(Optimizer):
    # クラス定義＆初期化
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._init_lr = lr 
        self.should_stop = False # 停止フラグの初期化

    # 感情EMA更新(緊張と安静)
    def _update_ema(self, state, loss_val):
        ema = state.setdefault('ema', {})
        ema['short'] = 0.3 * loss_val + 0.7 * ema.get('short', loss_val)
        ema['long'] = 0.01 * loss_val + 0.99 * ema.get('long', loss_val)
        return ema

    # 感情スカラー値生成(EMA差分、滑らかな非線形スカラー、tanh 5 * diff で鋭敏さ強調)
    def _compute_scalar(self, ema):
        diff = ema['short'] - ema['long']
        return math.tanh(5 * diff)

    # Shadow混合比率(> abs 0.6：60〜100%、 > abs 0.1：10〜60%、 平時：0%) emosens反映
    # 旧：Shadow混合比率(> 0.6：80〜90%、 < -0.6：10%、 abs> 0.3：30%、 平時：0%)
    # 説明：scalar>+0.6 は "return 0.7(開始値) + 0.2(変化幅) * scalar" = 0.82～0.9 ← 誤
    # 修正1：scalar>±0.6 を "return 開始値 + (abs(scalar) - 0.6(範囲)) / 範囲量 * 変化幅"
    # 修正2：scalar>±0.1 を "return 開始値 + (abs(scalar) - 0.1(範囲)) / 範囲量 * 変化幅"
    # タスク等に応じた調整のため３段階で適用しておく(上記を参考に調整してください／現状はshadow-effect反映)
    def _decide_ratio(self, scalar):
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

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # EMA更新・スカラー生成(EMA差分からスカラーを生成しスパイク比率を決定)
                ema = self._update_ema(state, loss_val)
                scalar = self._compute_scalar(ema)
                ratio = self._decide_ratio(scalar)

                # shadow_param：必要時のみ更新(スパイク部分に現在値を5%ずつ追従させる動的履歴)
                if ratio > 0:
                    if 'shadow' not in state:
                        state['shadow'] = p.clone()
                    else:
                        p.mul_(1 - ratio).add_(state['shadow'], alpha=ratio)
                        state['shadow'].lerp_(p, 0.05)
                
                # スカラー生成：短期と長期EMAの差分から信号を得る(高ぶりの強さ)
                # 混合比率：スカラーが閾値を超える場合にのみ計算される(信頼できる感情信号かどうかの選別)
                # → スカラー値が小さい場合は ratio = 0 となり、shadow混合は行われない
                # → 信頼できる強い差分のときのみ感情機構が発動する(暗黙の信頼度判定)                
                
                # 1次・2次モーメントを使った勾配補正(decoupled weight decay 構造に近い)
                exp_avg = state.setdefault('exp_avg', torch.zeros_like(p))
                exp_avg_sq = state.setdefault('exp_avg_sq', torch.zeros_like(p))
                beta1, beta2 = group['betas']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr']
                if group['weight_decay']:
                    p.add_(p, alpha=-group['weight_decay'] * step_size)
                p.addcdiv_(exp_avg, denom, value=-step_size)

        # 感情機構の発火が収まり"十分に安定"していることを外部伝達できる(自動停止ロジックではない)
                # Early Stop用 scalar 記録(バッファ共通で管理/最大32件保持/動静評価)
                hist = self.state.setdefault('scalar_hist', [])
                hist.append(scalar)
                if len(hist) >= 33:
                    hist.pop(0)

        # Early Stop判断(静けさの合図)
        if len(self.state['scalar_hist']) >= 32:
            buf = self.state['scalar_hist']
            avg_abs = sum(abs(s) for s in buf) / len(buf)
            std = sum((s - sum(buf)/len(buf))**2 for s in buf) / len(buf)
            if avg_abs < 0.05 and std < 0.005:
                self.should_stop = True  # 💡 外部からこれを見て判断可

        # 32ステップ分のスカラー値の静かな条件を満たした時"フラグ" should_stop = True になるだけ

        return loss

"""
 https://github.com/muooon/EmoNavi
 An emotion-driven optimizer that feels loss and navigates accordingly.
 Don't think. Feel. Don't stop. Keep running. Believe in what's beyond.
"""
