import torch
from torch.optim import Optimizer
import math
from typing import Tuple, Callable, Union

"""
AMP対応完了(202507) p.data -> p 修正済み
memo : "optimizer = EmoNeco(model.parameters(), lr=1e-3, use_shadow=False)"
optimizer 指定の際に False にすることで shadow をオフにできる
"""

# Helper function (Lynx)
def exists(val):
    return val is not None
# Soft Sign 関数
def softsign(x):
    return x / (1 + x.abs())

class EmoNeco(Optimizer):
    # クラス定義＆初期化 -  🔸Shadow True(有効)/False(無効) 切替え
    def __init__(self, params: Union[list, torch.nn.Module], lr=1e-3, betas=(0.9, 0.99), 
    # neco用ベータ･互換性の追加(neco用beta1･beta2)
                 eps=1e-8, weight_decay=0.01, decoupled_weight_decay: bool = False, use_shadow: bool = True): 

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
        # ウェイト減衰のため保存
        self._init_lr = lr
        self.decoupled_wd = decoupled_weight_decay
        self.should_stop = False # 停止フラグの初期化
        self.use_shadow = use_shadow # 🔸shadowの使用フラグを保存

    # 感情EMA更新(緊張と安静)
    def _update_ema(self, state, loss_val):
        ema = state.setdefault('ema', {})
        ema['short'] = 0.3 * loss_val + 0.7 * ema.get('short', loss_val)
        ema['long']  = 0.01 * loss_val + 0.99 * ema.get('long', loss_val)
        return ema

    # 感情スカラー値生成(EMA差分、滑らかな非線形スカラー、tanh 5 * diff で鋭敏さ強調)
    def _compute_scalar(self, ema):
        diff = ema['short'] - ema['long']
        return math.tanh(5 * diff)

    # Shadow混合比率(> 0.6：70〜90%、 < -0.6：10%、 abs> 0.3：30%、 平時：0%)
    def _decide_ratio(self, scalar):
        # 🔸use_shadow が False の場合は常に比率を 0 にする
        if not self.use_shadow:
            return 0.0
        if scalar > 0.6:
            return 0.7 + 0.2 * scalar
        elif scalar < -0.6:
            return 0.1
        elif abs(scalar) > 0.3:
            return 0.3
        return 0.0

    # 損失取得(損失値 loss_val を数値化、感情判定に使用、存在しないパラメータ(更新不要)はスキップ)
    @torch.no_grad()
    def step(self, closure: Callable | None = None): # クロージャの型ヒントを追加
        loss = None
        if exists(closure): # 一貫性のためにexistsヘルパーを使う
            with torch.enable_grad():
                loss = closure()
        loss_val = loss.item() if loss is not None else 0.0

        for group in self.param_groups:
            # 共通パラメータ抽出
            lr, wd, beta1, beta2 = group['lr'], group['weight_decay'], *group['betas']
            
            # ウェイト減衰の処理を分離 (from lynx)
            _wd_actual = wd
            if self.decoupled_wd:
                _wd_actual /= self._init_lr # 非連結時ウェイト減衰調整

            for p in filter(lambda p: exists(p.grad), group['params']): # PGチェックにフィルタ

                grad = p.grad # PG直接使用(計算に".data"不要)
                state = self.state[p]

                # EMA更新・スカラー生成(EMA差分からスカラーを生成しスパイク比率を決定)
                ema = self._update_ema(state, loss_val)
                scalar = self._compute_scalar(ema)
                ratio = self._decide_ratio(scalar) # 🔸use_shadow に応じて ratio が 0 になる

                # shadow_param：必要時のみ更新(スパイク部分に現在値を5%ずつ追従させる動的履歴)
                # 🔸self.use_shadow が True で、かつ ratio > 0 の場合のみ shadow を更新
                if self.use_shadow and ratio > 0: 
                    if 'shadow' not in state:
                        state['shadow'] = p.clone()
                    else:
                        p.mul_(1 - ratio).add_(state['shadow'], alpha=ratio) 
                        state['shadow'].lerp_(p, 0.05) 
                        # 更新前 p で shadow 更新(現在値を5%ずつ追従)
                        # p.mul_(1 - ratio).add_(state['shadow'], alpha=ratio) 
                        # EmoNavi: p = p * (1-ratio) + shadow * ratio

                # --- Start Neco Gradient Update Logic ---
                
                # neco初期化(exp_avg_sq)
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p)
                exp_avg = state['exp_avg']

                # Stepweight decay (from lynx): p = p * (1 - lr * wd)
                # decoupled_wd 考慮 _wd_actual 使用(EmoNaviのwdは最後に適用)
                p.mul_(1. - lr * _wd_actual)

                # 勾配ブレンド
                # m_t = beta1 * exp_avg_prev + (1 - beta1) * grad
                blended_grad = grad.mul(1. - beta1).add_(exp_avg, alpha=beta1)
                grad_norm = torch.norm(grad, dtype=torch.float32) # 勾配ノルムの計算

                # scalar < -0.3 の場合のみ SoftSign、それ以外 Cautious (終盤や発散傾向をSSに)
                # p - lr * softsign(blended_grad) (from softsign)
                # p - lr * direction * mask (from Cautious)
                # safe_norm 極値のブレンド勾配に対するスケーリング
                if 0.3 < scalar <= 0.5:
                    safe_norm = grad_norm + eps
                    modified_grad = softsign(blended_grad) * safe_norm
                    p.add_(-lr * modified_grad) 
                elif scalar < -0.3:
                    p.add_(softsign(blended_grad), alpha = -lr)  # Soft Sign 処理
                else:
                    direction = blended_grad.sign()    # 勾配方向の符号 Cautious 処理
                    mask = (direction == grad.sign())  # 過去の勾配と方向が一致している部分のみ更新
                    p.add_(direction * mask, alpha = -lr)  # Cautious 更新

                # exp_avg = beta2 * exp_avg + (1 - beta2) * grad
                exp_avg.mul_(beta2).add_(grad, alpha = 1. - beta2)

                # --- End Neco Gradient Update Logic ---

                # Early Stop用 scalar記録(バッファ共通で管理/最大32件保持/動静評価)
                # この部分は p.state ではなく self.state にアクセスする
                hist = self.state.setdefault('scalar_hist', [])
                hist.append(scalar)
                if len(hist) >= 33:
                    hist.pop(0)

        # Early Stop判断(静けさの合図) This part is outside the inner loop
        if len(self.state['scalar_hist']) >= 32:
            buf = self.state['scalar_hist']
            avg_abs = sum(abs(s) for s in buf) / len(buf)
            std = sum((s - sum(buf)/len(buf))**2 for s in buf) / len(buf)
            if avg_abs < 0.05 and std < 0.005:
                self.should_stop = True # 外部からこれを見て判断可

        return loss

"""
 https://github.com/muooon/EmoNavi
 Neco was developed with inspiration from Lion, Tiger, Cautious, softsign, and EmoLynx 
 which we deeply respect for their lightweight and intelligent design.  
 Neco also integrates EmoNAVI to enhance its capabilities.
"""