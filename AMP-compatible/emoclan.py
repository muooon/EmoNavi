import torch
from torch.optim import Optimizer
import math
from typing import Callable, Union, Dict, Any, Tuple

"""
EmoClan v2.0 (250815) shadow-system v2.0 scalar-switch v2.0
AMP対応完了(202507) p.data -> p 修正済み
memo : "optimizer = EmoClan(model.parameters(), lr=1e-3, use_shadow=True)"
optimizer 指定の際に True にすることで shadow をオンにできる
emosens shadow-effect v1.0 反映 shadow-system、scalar-switch 修正
"""

# Helper function
def exists(val):
    return val is not None

class EmoClan(Optimizer):
    # クラス定義＆初期化 🔸Shadow True(有効)/False(無効) 切替え
    def __init__(self, params: Union[list, torch.nn.Module], 
                 lr: float = 1e-3, 
                 betas: Tuple[float, float] = (0.9, 0.999), 
                 eps: float = 1e-8, 
                 weight_decay: float = 0.01,
                 lynx_betas: Tuple[float, float] = (0.9, 0.99), # Lynx 固有の beta
                 decoupled_weight_decay: bool = False,
                 use_shadow: bool = False
                ):
        
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        # Lynx の betas もバリデーション
        if not 0.0 <= lynx_betas[0] < 1.0:
            raise ValueError(f"Invalid lynx_beta parameter at index 0: {lynx_betas[0]}")
        if not 0.0 <= lynx_betas[1] < 1.0:
            raise ValueError(f"Invalid lynx_beta parameter at index 1: {lynx_betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        lynx_betas=lynx_betas, decoupled_weight_decay=decoupled_weight_decay)
        super().__init__(params, defaults)
        
        self._init_lr = lr # decoupled weight decay のために保存 (Lynx用)
        self.should_stop = False # 全体の停止フラグ
        self.use_shadow = use_shadow # EmoClanインスタンス自身がuse_shadowを保持

    # --- 感情機構 (Emotion Mechanism) ---
    def _update_ema(self, param_state: Dict[str, Any], loss_val: float) -> Dict[str, float]:
        """損失値に基づいて短期・長期 EMA を更新"""
        # param_state は各パラメータの state['ema'] を保持する
        ema = param_state.setdefault('ema', {'short': loss_val, 'long': loss_val})
        ema['short'] = 0.3 * loss_val + 0.7 * ema['short']
        ema['long'] = 0.01 * loss_val + 0.99 * ema['long']
        return ema

    """EMA の差分から感情スカラー値を生成"""
    def _compute_scalar(self, ema: Dict[str, float]) -> float:
        diff = ema['short'] - ema['long']
        return math.tanh(5 * diff)

    """感情スカラーに基づいて Shadow の混合比率を決定"""
    # Shadow混合比率(> abs 0.6：60〜100%、 > abs 0.1：10〜60%、 平時：0%) emosens反映
    # 旧：Shadow混合比率(> 0.6：80〜90%、 < -0.6：10%、 abs> 0.3：30%、 平時：0%)
    # 説明：scalar>+0.6 は "return 0.7(開始値) + 0.2(変化幅) * scalar" = 0.82～0.9 ← 誤
    # 修正1：scalar>±0.6 を "return 開始値 + (abs(scalar) - 0.6(範囲)) / 範囲量 * 変化幅"
    # 修正2：scalar>±0.1 を "return 開始値 + (abs(scalar) - 0.1(範囲)) / 範囲量 * 変化幅"
    # タスク等に応じた調整のため３段階で適用しておく(上記を参考に調整してください／現状はshadow-effect反映)
    def _decide_ratio(self, scalar: float) -> float:
        if not self.use_shadow:
            return 0.0 # 🔸use_shadow が False の場合は常に比率を 0 にする
        if abs(scalar) > 0.6:
            return 0.6 + (abs(scalar) - 0.6) / 0.4 * 0.4 # 元 return 0.7 + 0.2 * scalar
        elif abs(scalar) > 0.1:
            return 0.1 + (abs(scalar) - 0.1) / 0.5 * 0.5 # 元 return 0.3
        return 0.0

    # --- 各最適化器のコアな勾配更新ロジック (プライベートメソッドとして統合) ---

    def _lynx_update(
        self, 
        p: torch.Tensor, 
        grad: torch.Tensor, 
        param_state: Dict[str, Any], 
        lr: float, 
        beta1: float, 
        beta2: float, 
        wd_actual: float
    ):
        """EmoLynx のコアな勾配更新ロジック"""
        # Stepweight decay: p = p * (1 - lr * wd)
        p.mul_(1. - lr * wd_actual)

        # Lynx 固有の EMA 状態は param_state に保持
        if 'exp_avg_lynx' not in param_state:
            param_state['exp_avg_lynx'] = torch.zeros_like(p)
        exp_avg = param_state['exp_avg_lynx']

        # 勾配ブレンド
        blended_grad = grad.mul(1. - beta1).add_(exp_avg, alpha=beta1)
        
        # 符号ベースの更新
        p.add_(blended_grad.sign_(), alpha = -lr)

        # exp_avg 更新
        exp_avg.mul_(beta2).add_(grad, alpha = 1. - beta2)

    def _navi_update(
        self, 
        p: torch.Tensor, 
        grad: torch.Tensor, 
        param_state: Dict[str, Any], 
        lr: float, 
        betas: Tuple[float, float], 
        eps: float, 
        weight_decay: float
    ):
        """EmoNavi のコアな勾配更新ロジック"""
        beta1, beta2 = betas

        exp_avg = param_state.setdefault('exp_avg_navi', torch.zeros_like(p))
        exp_avg_sq = param_state.setdefault('exp_avg_sq_navi', torch.zeros_like(p.to(torch.float32)))

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad.to(torch.float32), grad.to(torch.float32), value=1 - beta2)
        denom = exp_avg_sq.sqrt().add_(eps)

        # Weight decay (標準的手法)
        if weight_decay:
            p.mul_(1 - lr * weight_decay) 

        p.addcdiv_(exp_avg, denom, value=-lr)

    def _fact_update(
        self, 
        p: torch.Tensor, 
        grad: torch.Tensor, 
        param_state: Dict[str, Any], 
        lr: float, 
        betas: Tuple[float, float], # beta2 は現状使われないが互換性のため残す (1D勾配で使用)
        eps: float, 
        weight_decay: float
    ):
        """EmoFact のコアな勾配更新ロジック (Adafactor ライク)"""
        beta1, beta2 = betas

        if grad.dim() >= 2:
            # 行と列の2乗平均を計算 (分散の軽量な近似)
            # gradをfloat32にキャストして計算することで数値安定性を高める
            r_sq = torch.mean(grad.to(torch.float32) * grad.to(torch.float32), dim=tuple(range(1, grad.dim())), keepdim=True).add_(eps)
            c_sq = torch.mean(grad.to(torch.float32) * grad.to(torch.float32), dim=0, keepdim=True).add_(eps)

            param_state.setdefault('exp_avg_r_fact', torch.zeros_like(r_sq)).mul_(beta1).add_(torch.sqrt(r_sq), alpha=1 - beta1)
            param_state.setdefault('exp_avg_c_fact', torch.zeros_like(c_sq)).mul_(beta1).add_(torch.sqrt(c_sq), alpha=1 - beta1)
            
            # 再構築した近似勾配の平方根の積で正規化
            denom = torch.sqrt(param_state['exp_avg_r_fact'] * param_state['exp_avg_c_fact']).add_(eps)
            update_term = grad / denom # grad は元の型（float16またはfloat32）

        else: # 1次元(ベクトル)の勾配補正
            exp_avg = param_state.setdefault('exp_avg_fact', torch.zeros_like(p))
            exp_avg_sq = param_state.setdefault('exp_avg_sq_fact', torch.zeros_like(p.to(torch.float32)))
            
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad.to(torch.float32), grad.to(torch.float32), value=1 - beta2)
            denom = exp_avg_sq.sqrt().add_(eps)
            update_term = exp_avg / denom

        # 最終的なパラメータ更新 (decoupled weight decayも適用)
        # decoupled_weight_decay は __init__ でグループにdefaultsとして渡されているが、
        # ここではfactorロジック自体がweight_decayを受け取る形式
        p.mul_(1 - weight_decay * lr) 
        p.add_(update_term, alpha=-lr)


    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()
        loss_val = loss.item() if loss is not None else 0.0

        # 全体の scalar_hist を EmoClan インスタンスで管理
        global_scalar_hist = self.state.setdefault('global_scalar_hist', [])
        
        # 全体としての感情EMA状態を self.state に保持し、現在の感情スカラーを計算
        global_ema_state = self.state.setdefault('global_ema', {'short': loss_val, 'long': loss_val})
        global_ema_state['short'] = 0.3 * loss_val + 0.7 * global_ema_state['short']
        global_ema_state['long'] = 0.01 * loss_val + 0.99 * global_ema_state['long']
        current_global_scalar = self._compute_scalar(global_ema_state)
        
        # global_scalar_hist に現在の感情スカラーを追加
        global_scalar_hist.append(current_global_scalar)
        if len(global_scalar_hist) >= 33:
            global_scalar_hist.pop(0)


        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            eps = group['eps']
            decoupled_wd = group['decoupled_weight_decay']
            
            lynx_beta1, lynx_beta2 = group['lynx_betas']
            navi_fact_betas = group['betas'] # Navi/Fact 共通の beta を使用 (デフォルトの betas)
            
            # Lynx の decoupled_wd のための _wd_actual 計算
            _wd_actual_lynx = wd
            if decoupled_wd:
                _wd_actual_lynx /= self._init_lr

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                param_state = self.state[p] # 各パラメータごとの状態

                # --- 各パラメータごとの感情機構の更新と Shadow 処理 ---
                # 各パラメータの state['ema'] は、それぞれの loss_val (全体で共通) を元に更新される
                # ただし、現状の loss_val はクロージャから受け取った単一の値なので、
                # 各パラメータ固有の「感情」を定義するより、全体としての感情が使われることになる。
                # use_shadow が True の場合にのみ Shadow 関連の処理を実行
                if self.use_shadow:  
                    param_ema = self._update_ema(param_state, loss_val) 
                    param_scalar = self._compute_scalar(param_ema) # 各パラメータ固有のスカラー

                    ratio = self._decide_ratio(param_scalar) # 各パラメータ固有の ratio

                    if ratio > 0:
                        if 'shadow' not in param_state:
                            param_state['shadow'] = p.clone()
                        else:
                            # Shadow を現在値にブレンド
                            p.mul_(1 - ratio).add_(param_state['shadow'], alpha=ratio)
                        # Shadow を現在値に追従させる
                        param_state['shadow'].lerp_(p, 0.05)

                # --- 最適化器の選択と勾配更新 ---
                # 現在のglobal_scalar_histに記録された全体としての感情スカラーに基づいてフェーズを判断
                # global_scalar > abs 0.6 の範囲は Lynx
                # global_scalar > abs 0.3 の範囲は Fact
                # global_scalar < abs 0.3 の範囲は Navi
                if abs(current_global_scalar) > 0.6: # 序盤・過学習・発散時
                    self._lynx_update(p, grad, param_state, lr, lynx_beta1, lynx_beta2, _wd_actual_lynx)
                elif abs(current_global_scalar) > 0.3: # 終盤・過学習・発散傾向時
                    self._fact_update(p, grad, param_state, lr, navi_fact_betas, eps, wd)
                else: # -0.3 <= current_global_scalar <= 0.3 の中盤･平時(安定期)
                    self._navi_update(p, grad, param_state, lr, navi_fact_betas, eps, wd)

        # Early Stop判断
        # global_scalar_hist の評価
        if len(global_scalar_hist) >= 32:
            buf = global_scalar_hist
            avg_abs = sum(abs(s) for s in buf) / len(buf)
            std = sum((s - sum(buf)/len(buf))**2 for s in buf) / len(buf)
            if avg_abs < 0.05 and std < 0.005:
                self.should_stop = True # 外部からこれを見て判断可

        return loss

"""
 Emoシリーズは、Adam、Adafactor、Lion、Tiger、等から多くを学びました。  
 この開発において先人たちの知見に深く感謝しつつ今後も新しい可能性を探究します。 
 The Emo series has learned much from Adam, Adafactor, Lion, and Tiger.  
 Rather than being their successors,  
 In its development, we deeply appreciate the insights of those who came before us—and continue to explore new possibilities beyond them.  
"""