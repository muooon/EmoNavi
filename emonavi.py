import torch
from torch.optim import Optimizer
import math

class EmoNavi(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def _update_ema(self, state, loss_val):
        ema = state.setdefault('ema', {})
        ema['short'] = 0.3 * loss_val + 0.7 * ema.get('short', loss_val)
        ema['long']  = 0.01 * loss_val + 0.99 * ema.get('long', loss_val)
        return ema

    def _compute_scalar(self, ema):
        diff = ema['short'] - ema['long']
        return math.tanh(5 * diff)

    def _decide_ratio(self, scalar):
        if scalar > 0.6:
            return 0.7 + 0.2 * scalar
        elif scalar < -0.6:
            return 0.1
        elif abs(scalar) > 0.3:
            return 0.3
        return 0.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        loss_val = loss.item() if loss is not None else 0.0

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # EMA更新・スカラー生成
                ema = self._update_ema(state, loss_val)
                scalar = self._compute_scalar(ema)
                ratio = self._decide_ratio(scalar)

                # shadow_param：必要時のみ更新
                if ratio > 0:
                    if 'shadow' not in state:
                        state['shadow'] = p.data.clone()
                    else:
                        p.data.mul_(1 - ratio).add_(state['shadow'], alpha=ratio)
                        state['shadow'].lerp_(p.data, 0.05)

                # AdamW形式のパラメータ更新
                exp_avg = state.setdefault('exp_avg', torch.zeros_like(p.data))
                exp_avg_sq = state.setdefault('exp_avg_sq', torch.zeros_like(p.data))
                beta1, beta2 = group['betas']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr']
                if group['weight_decay']:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * step_size)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Early Stop用 scalar 記録（バッファ共通で管理）
                hist = self.state.setdefault('scalar_hist', [])
                hist.append(scalar)
                if len(hist) > 32:
                    hist.pop(0)

        # Early Stop判断（Refの静かな合図）
        if len(self.state['scalar_hist']) >= 32:
            buf = self.state['scalar_hist']
            avg_abs = sum(abs(s) for s in buf) / len(buf)
            std = sum((s - sum(buf)/len(buf))**2 for s in buf) / len(buf)
            if avg_abs < 0.05 and std < 0.005:
                self.should_stop = True  # 💡 外部からこれを見て判断可

        return loss

#    https://github.com/muooon/EmoNavi
#    An emotion-driven optimizer that feels loss and navigates accordingly.
#    Don't think. Feel. Don't stop. Keep running. Believe in what's beyond.
