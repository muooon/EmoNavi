import torch
from torch.optim import Optimizer
import math

"""
EmoZeal v2.0 (250815) shadow-system v2.0 scalar-switch v2.0
AMP対応完了(202507) p.data -> p 修正済み
memo : "optimizer = EmoNeco(model.parameters(), lr=1e-3, use_shadow=True)"
optimizer 指定の際に True にすることで shadow をオンにできる
emosens shadow-effect v1.0 反映 shadow-system、scalar-switch 修正
"""

# Soft Sign 関数
def softsign(x): 
    return x / (1 + x.abs())
    
class EmoZeal(Optimizer):
    # クラス定義＆初期化 🔸Shadow True(有効)/False(無効) 切替え
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.01, use_shadow: bool = False): 
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super().__init__(params, defaults)

        self.alpha_prev = getattr(self, 'alpha_prev', 1.0)
        self._init_lr = lr 
        self.should_stop = False # 停止フラグの初期化
        self.use_shadow = use_shadow # 🔸shadowの使用フラグを保存

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

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # 感情EMA更新・スカラー生成 (既存ロジックを維持)
                ema = self._update_ema(state, loss_val)
                scalar = self._compute_scalar(ema)
                ratio = self._decide_ratio(scalar) # 🔸use_shadow に応じて ratio が 0 になる

                # shadow_param：必要時のみ更新 (既存ロジックを維持)
                # 🔸self.use_shadow が True で、かつ ratio > 0 の場合のみ shadow を更新
                if self.use_shadow and ratio > 0: 
                    if 'shadow' not in state:
                        state['shadow'] = p.clone()
                    else:
                        p.mul_(1 - ratio).add_(state['shadow'], alpha=ratio)
                        state['shadow'].lerp_(p, 0.05)
                
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
                    eps = group['eps'] 
                    lr = group['lr']   
                    exp_avg = state.setdefault('exp_avg', torch.zeros_like(p))
                    blended_grad = grad.mul(1 - beta1).add_(exp_avg, alpha=beta1)
                    grad_norm = torch.norm(grad, dtype=torch.float32)
                    # > abs 0.6 Cautious (過適合や崩壊傾向を慎重に)
                    # > abs 0.1 SoftSign+NormEPS (揺れを滑らかに)
                    # 削除：それ以外 SoftSign (ゆっくり滑らかに)
                    # p - lr * softsign(blended_grad) (from softsign)
                    # p - lr * direction * mask (from Cautious)
                    # safe_norm 極値のブレンド勾配に対するスケーリング
                    if abs(scalar) > 0.6:
                        direction = blended_grad.sign()    # 勾配方向の符号 Cautious 処理
                        mask = (direction == grad.sign())  # 過去の勾配と方向が一致する部分のみ更新
                        p.add_(direction * mask, alpha = -lr)  # Cautious 更新
                    elif abs(scalar) > 0.1:
                        safe_norm = grad_norm + eps
                        modified_grad = softsign(blended_grad) * safe_norm
                        p.add_(-lr * modified_grad) 
                    
                    state.setdefault('exp_avg_r', torch.zeros_like(r_sq)).mul_(beta1).add_(torch.sqrt(r_sq), alpha=1 - beta1)
                    state.setdefault('exp_avg_c', torch.zeros_like(c_sq)).mul_(beta1).add_(torch.sqrt(c_sq), alpha=1 - beta1)
                    
                    # 再構築した近似勾配の平方根の積で正規化
                    # これにより2次モーメントのような役割を果たす
                    denom = torch.sqrt(state['exp_avg_r'] * state['exp_avg_c']) + eps
                    
                    # 最終的な更新項を計算
                    update_term = grad / denom

                # 1次元(ベクトル)の勾配補正(decoupled weight decay 構造に近い)
                else:
                    exp_avg = state.setdefault('exp_avg', torch.zeros_like(p))
                    exp_avg_sq = state.setdefault('exp_avg_sq', torch.zeros_like(p))
                    beta1, beta2 = group['betas']
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    update_term = exp_avg / denom

                # 最終的なパラメータ更新 (decoupled weight decayも適用)
                p.add_(p, alpha=-group['weight_decay'] * group['lr'])
                p.add_(update_term, alpha=-group['lr'])

                # --- Early Stop ロジック (既存ロジックを維持) ---
                hist = self.state.setdefault('scalar_hist', [])
                hist.append(scalar)
                if len(hist) >= 33:
                    hist.pop(0)

        # Early Stop判断
        if len(self.state['scalar_hist']) >= 32:
            buf = self.state['scalar_hist']
            avg_abs = sum(abs(s) for s in buf) / len(buf)
            std = sum((s - sum(buf)/len(buf))**2 for s in buf) / len(buf)
            if avg_abs < 0.05 and std < 0.005:
                self.should_stop = True

        return loss

"""
 https://github.com/muooon/EmoNavi
 Zeal is inspired by Adafactor, and EmoFact,  
 and its VRAM-friendly design is something everyone loves.
"""