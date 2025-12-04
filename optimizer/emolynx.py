import torch
from torch.optim import Optimizer
import math
from typing import Tuple, Callable, Union
from collections import deque

"""
EmoLynx v3.3 (251202) shadow-system v3.0 -effect NoN -moment v3.0
(v1.0)AMP対応完了(250725) p.data -> p 修正済み／低精度量子化への基本対応／低精度補償は別
(v2.0)shadow-system 微調整／３段階補正を連続的に滑らかに／派生版では以下の切替も可能
optimizer 指定の際に True / False で shadow を切替できる(現在 False)
(v3.0)emosens shadow-effect v1.0 反映した動的学習率と shadow-system 切替をデフォルト化
(v3.1)通常未使用の shadow 更新速度 (lerp) を倍化し信頼度で動的制御／coeff 活用(急変･微動)
動的学習率や感情スカラー値など TensorBoard 連携可 (現在 writer=None)／外部設定必要
全体の効率化や可読性を向上(emaやスカラーの多重処理を省く等、動的学習率のスケールや状態の見直し等、含む)
(v3.3)トラウマ的反応や慣れによる鈍化で安定性向上(ema-medium 安定と急変を信頼度で感知)
完全自動学習率／目標減少率制御方式を導入／感情機構との相乗効果で急変時も鎮静化し安定進行
"""

# Helper function (Lynx)
def exists(val):
    return val is not None

class EmoLynx(Optimizer):
    # クラス定義＆初期化 lynx用ベータ･互換性の追加(lynx用beta1･beta2)
    def __init__(self, params: Union[list, torch.nn.Module], 
                 lr=1e-3, 
                 eps=1e-8,
                 lr_max=1e-3, 
                 lr_min=1e-8, 
                 betas=(0.9, 0.999), 
                 weight_decay=0.01, 
                 use_shadow:bool=False, 
                 writer=None): 
                     
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        
        super().__init__(params, defaults)
        self._init_lr = lr
        self.should_stop = False # 停止フラグの初期化
        self.use_shadow = use_shadow # 🔸shadow 使用フラグを保存
        self.writer = writer # 動的学習率や感情スカラー等を渡す
        self.eta = lr # 名目lrを初期値として利用(自己更新)
        self.k = 0.2 # 学習率自己更新の応答速度係数(比例制御の強さ)
        self.eps = 1e-8 # ゼロ割り防止の微小値(分母安定化)
        self.lr_min = 1e-8 # 学習率の下限(極端な縮小の防止)
        self.lr_max = 1e-3 # 学習率の上限(極端な拡大の防止)
        self.prev_loss = None # Loss初期化

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
            return 0.0 # 🔸use_shadow = False のとき常に比率を 0 にする
        if abs(scalar) > 0.75:
            return 0.75 # + ((scalar) - 0.75) / 0.4 * 0.4 # これはスカラーそのまま返す参考例
        elif abs(scalar) > 0.25:
            return -0.1 # return<0 の場合は leap 専用(書き戻しはしないが履歴更新のみ)
        return 0.0

    # 損失取得(損失値 loss_val を数値化、感情判定に使用、存在しないパラメータ(更新不要)はスキップ)
    @torch.no_grad()
    def step(self, closure: Callable | None = None): # クロージャの型ヒントを追加
        loss = None
        if exists(closure): # 一貫性のためにexistsヘルパーを使う
            with torch.enable_grad():
                loss = closure()
        loss_val = loss.item() if loss is not None else 0.0

        # EMA更新・スカラー生成(EMA差分からスカラーを生成しスパイク比率等を決定)
        ema = self._update_ema(self.state, loss_val)
        early_scalar = self._early_scalar(ema)
        scalar = self._compute_scalar(ema)
        coeff = self._decide_coeff(scalar)
        ratio = self._decide_ratio(scalar)

        # 目標減少率制御 ＋ eta_eff
        if self.prev_loss is None:
            self.prev_loss = loss_val # 初回は初期化のみ
            eta_eff = max(self.lr_min, min(self.lr_max, self.eta * coeff))
        else:
            delta = self.prev_loss - loss_val
            target_delta = max(1e-8, 0.01 * max(loss_val, 1e-8)) # １%固定
            # 学習率の自己更新(比例制御)
            self.eta *= math.exp(self.k * (delta - target_delta) / (abs(target_delta) + self.eps))
            # 感情スカラーで補正し最終ステップへ
            eta_eff = max(self.lr_min, min(self.lr_max, self.eta * coeff))

        for group in self.param_groups:
            step_size = eta_eff # 💡 group['lr'] は使わない
            # リンクス共通パラメータ抽出
            wd, beta1, beta2 = group['weight_decay'], *group['betas']

            # ウェイト減衰の処理を分離 (from lynx)
            _wd_actual = wd

            for p in filter(lambda p: exists(p.grad), group['params']): # PGチェックにフィルタ

                grad = p.grad # PG直接使用(計算に".data"不要)
                state = self.state[p]

                # 動的学習率補正により shadow 形成を信頼度で調整(coeffは正値(負にならない))
                # shadow：必要時のみ(スパイクp部分に現在値を最大10%追従させる動的履歴更新)
                # ratio <0：10%、0以外：10%×coeff、(0.25～0.75は10%、微動と急変は*coeff)
                # 微動時 coeff：1.0 固定なので結果的に微動時も 10% 履歴更新になる
                # 結果、微動時と安定時：10%、急変時：coeff、による履歴更新を行うことになる
                if self.use_shadow:
                    if 'shadow' not in state: # 🔸shadow = False (デフォルト)
                        state['shadow'] = p.clone()
                    if ratio > 0: # 書き戻しと履歴更新(急変時の強い抑制と弱めの履歴更新)
                        p.mul_(1 - ratio).add_(state['shadow'], alpha=coeff)
                    else: # 書き戻しせず履歴更新のみ：ratio<0：10%／0以外：10%×coeff
                        leap_ratio = 0.1 if ratio < 0 else 0.1 * coeff
                        state['shadow'].lerp_(p, leap_ratio)

                # 上記 shadow の説明：スカラー生成：短期と長期EMAの差分から信号を得る(高ぶりの強さ)
                # 混合比率：スカラーが閾値を超える場合にのみ計算される(信頼できる感情信号かどうかの選別)
                # 急変時は感情機構による shadow 混合で強く抑制する(急制動による安定性の確保)
                # 新しい shadow-system は動的学習率と協調することで選択的スパース性も発揮する

                # --- Start Lynx Gradient Update Logic ---
                # lynx初期化(exp_avg_sq)
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p)
                exp_avg = state['exp_avg']

                # Stepweight decay (from lynx): p = p * (1 - lr * wd)
                # decoupled_wd 考慮 _wd_actual 使用(EmoNaviのwdは最後に適用)
                p.mul_(1 - step_size * _wd_actual)
                beta1, beta2 = group['betas']

                # 勾配ブレンド
                # m_t = beta1 * exp_avg_prev + (1 - beta1) * grad
                blended_grad = grad.mul(1 - beta1).add_(exp_avg, alpha=beta1)

                # p: p = p - lr * sign(blended_grad)
                p.add_(blended_grad.sign_(), alpha = -step_size)

                # exp_avg = beta2 * exp_avg + (1 - beta2) * grad
                exp_avg.mul_(beta2).add_(grad, alpha = 1 - beta2)
                # --- End Lynx Gradient Update Logic ---

        self.prev_loss = loss_val

        # 感情機構の発火が収まり"十分に安定"していることを外部伝達できる(自動停止ロジックではない)
        # Early Stop用 scalar 記録(バッファ共通で管理/最大32件保持/動静評価)
        hist = self.state.setdefault('scalar_hist', deque(maxlen=32))
        hist.append(early_scalar)

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
            self.writer.add_scalar("emoLR", eta_eff, self._step_count)
            self.writer.add_scalar("etaLR", self.eta, self._step_count)
            self.writer.add_scalar("emoScalar", scalar, self._step_count)

        return loss

"""
 https://github.com/muooon/EmoNavi
 Lynx was developed with inspiration from Lion, Tiger, and emocats, 
 which we deeply respect for their lightweight and intelligent design.  
 Lynx also integrates EmoNAVI to enhance its capabilities.
"""