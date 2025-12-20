import torch
from torch.optim import Optimizer
import math
from collections import deque

"""
EmoFact v3.6.1 (251220) shadow-system v3.1 -moment v3.1 emoDrive ｖ3.6
(v1.0)AMP対応完了(250725) p.data -> p 修正済み／低精度量子化への基本対応／低精度補償は別
(v2.0)shadow-system 微調整／３段階補正を連続的に滑らかに／派生版では以下の切替も可能
optimizer 指定の際に True / False で shadow を切替できる(現在 False)
(v3.0)emosens shadow-effect v1.0 反映した動的学習率と shadow-system 切替をデフォルト化
(v3.1)通常未使用の shadow 更新速度 (lerp) を倍化し信頼度で動的制御／coeff 活用(急変･微動)
動的学習率や感情スカラー値など TensorBoard 連携可 (現在 writer=None)／外部設定必要
全体の効率化や可読性を向上(emaやスカラーの多重処理を省く等、動的学習率のスケールや状態の見直し等、含む)
(v3.6)-Final- emoDrive v3.6 により信頼度に応じ学習率を大きく増減させることにした(emo系の完成版)
"""

class EmoFact(Optimizer):
    # クラス定義＆初期化
    def __init__(self, params, 
                 lr=1e-3, 
                 eps=1e-8, 
                 betas=(0.9, 0.995), 
                 weight_decay=0.01, 
                 use_shadow:bool=False, 
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
        ema['medium'] = 0.05 * loss_val + 0.95 * ema.get('medium', loss_val)
        ema['long'] = 0.01 * loss_val + 0.99 * ema.get('long', loss_val)
        return ema

    # 感情スカラー値生成(EMA差分、滑らかな非線形スカラー、tanh(diff) は ±1.0 で有界性)
    # 係数"1"：ema差分 のスケール調整処理に活用(感度調節係数)／通常は1(タスクに応じ調整可(非推奨))
    # scale_base：Loss値とema値の乖離を修正(分母 ema(long) 「改善率」共通化/loss種に非依存)
    # 1e-5(デフォルト)／1e-6(感度向上)／1e-4(安定性向上)：分母を０にせず安定させる
    # トラウマ的反応や慣れによる鈍化で安定性向上(ema-medium 安定と急変を信頼度で感知)
    def _compute_scalar(self, ema):
        scale_base_l = max(ema['long'], 1e-5)
        scale_base_m = max(ema['medium'], 1e-5)
        diff_l = (ema['long'] - ema['short']) / scale_base_l
        diff_m = (ema['long'] - ema['short']) / scale_base_m
        # longが十分静かなら、常にlongを優先
        if abs(diff_l) < 0.05:
            return math.tanh(diff_l)
        # longが静かでない時のみ、mediumの静けさを条件付きで採用
        if abs(diff_m) * scale_base_m < abs(diff_l) * scale_base_l:
            return math.tanh(1 * diff_m)
        else:
            return math.tanh(1 * diff_l)

    # アーリーストップ専用(静けさ判定の感情スカラ生成)
    def _early_scalar(self, ema):
        scale_base_l = max(ema['long'], 1e-5)
        diff = (ema['long'] - ema['short']) / scale_base_l
        return math.tanh(1 * diff)

    # 論文通りの抑制則/急変時は強抑制/悪化時は微減速/平時は無介入で収束を安定させる
    # 区分別けは現状では無意味ですが後々にカスタマイズしやすい形式として整理してあります
    def _decide_coeff(self, scalar):  # B <= x <= A: 等も可能
        if abs(scalar) > 0.75:
            return 1.0 - abs(scalar)  # 緊急｜急制動｜tanh 0.97(0.03)
        elif abs(scalar) > 0.50:
            return 1.0 - abs(scalar)  # 急変｜強抑制｜tanh 0.55(0.45)
        elif abs(scalar) > 0.25:
            return 1.0 - abs(scalar)  # 悪化｜微減速｜tanh 0.26(0.74)
        else:
            return 1.0                # 平時｜無介入｜他(常に 1.0 を返す)

    # (重要)現在は shadow-effect を参考に得た動的フィルタ効果の近似により use_shadow=False です
    # しかし全機能は shadow なしで全て成立します／コード確認時はこの shadow を考慮外として無視してください

    # Shadow混合比 ３段階構成 タスクに応じ調整可、以下を参考に 開始値・範囲量･変化幅を調整
    # return 開始値 + ((scalar) - 閾値) / 範囲量 * 変化幅 も可能(特殊用途向け)
    def _decide_ratio(self, scalar):
        if not self.use_shadow:
            return 0.0  # 🔸use_shadow = False のとき常に比率を 0 にする
        if abs(scalar) > 0.625:
            return 1.0 - abs(scalar)  # 急変｜強抑制｜tanh 0.73(0.27)
        else:
            return 0.0  # return<0 の場合は leap 専用(書き戻しはしないが履歴更新のみ)

    # 損失取得(損失値 loss_val を数値化、感情判定に使用、存在しないパラメータ(更新不要)はスキップ)
    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        loss_val = loss.item() if loss is not None else 0.0

        # EMA更新・スカラー生成(EMA差分からスカラーを生成しスパイク比率等を決定)
        ema = self._update_ema(self.state, loss_val)
        early_scalar = self._early_scalar(ema)
        scalar = self._compute_scalar(ema)
        coeff = self._decide_coeff(scalar)
        ratio = self._decide_ratio(scalar)
        trust = math.copysign((1.0 - abs(scalar)), scalar)
        emoDpt = 8.0 * abs(trust)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # 動的学習率補正により shadow 形成を信頼度で調整(trustは正値化(負にならない))
                # shadow：必要時のみ(スパイクp部分に現在値を最大10%追従させる動的履歴更新)
                # 混合比率：スカラーが閾値を超える場合にのみ計算される(信頼できる感情信号かどうかの選別)
                # 急変時は感情機構による shadow 混合で強く抑制する(急制動による安定性の確保)
                # 新 shadow-system は動的学習率と信頼度で協調し選択的スパース性も発揮する   
                if self.use_shadow :
                    if 'shadow' not in state: # 🔸shadow = False (デフォルト)
                        state['shadow'] = p.clone()
                    if ratio > 0: # 書き戻しと履歴更新(急変時の強い抑制と弱めの履歴更新)
                        p.mul_(1-ratio).add_(state['shadow'], alpha=abs(trust))
                    else: # 書き戻しせず履歴更新のみ：10%×trust
                        leap_ratio = 0.1 * abs(trust)
                        state['shadow'].lerp_(p, leap_ratio)  

                # emoDrive 作動域 (Turbo & Trust LR system)
                if 0.25 < abs(scalar) < 0.5:
                    emoDrive = emoDpt * (1.0 + 0.1 * trust)  # 加速／減速ゾーン補正
                elif abs(scalar) > 0.75:
                    emoDrive = coeff  # 緊急｜急制動｜tanh 0.97(0.03)
                else:
                    emoDrive = 1.0    # 無介入ゾーン

                # --- Start Gradient Update Logic ---
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
                    #update_term = grad / denom # sign化で１次ベクトルとのバランス改善
                    update_term = torch.sign(grad / denom)

                # 1次元(ベクトル)の勾配補正
                else:
                    beta1, beta2 = group['betas']
                    exp_avg_sq = state.setdefault('exp_avg_sq', torch.zeros_like(p))
                    exp_avg_sq.mul_(beta1).addcmul_(grad, grad, value=(1 - beta2))
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    #update_term = grad / denom # sign化で２次momentとのバランス改善
                    update_term = torch.sign(grad / denom)

                # 最終的なパラメータ更新 (decoupled weight decayも適用)
                step_size = group['lr']
                p.add_(p, alpha=-group['weight_decay'] * step_size)
                p.add_(update_term, alpha=-step_size * emoDrive)
                # --- End Gradient Update Logic ---

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
            self.writer.add_scalar("emoLR/base", step_size, self._step_count)
            self.writer.add_scalar("emoLR/Turbo", step_size * emoDrive, self._step_count)
            self.writer.add_scalar("emostate/emoDrive", emoDrive, self._step_count)
            self.writer.add_scalar("emostate/scalar", scalar, self._step_count)

        return

"""
 https://github.com/muooon/EmoNavi
 Fact is inspired by Adafactor, and emoairy, 
 and its VRAM-friendly design is something everyone loves.
"""