✨ モジュールの目的別紹介文（READMEにも追記可）
# 🤖 EmoNAVI-Kit — A Modular Toolkit for Ref-like Emotional Learning

This kit provides modular, philosophy-aligned components for emotion-driven optimization.
Each module encapsulates a specific function of EmoNAVI's inner “personality.”

| Module | Role | Description |
|--------|------|-------------|
| `ema.py`            | 感じる | Dual EMA tracker capturing short/long loss trends |
| `emotion_scalar.py` | 考える | Transforms fluctuation into an emotional scalar |
| `ref_injector.py`   | 導く   | Applies emotion-guided parameter blending |
| `stop_detector.py`  | 見守る | Detects stagnation and suppresses injection |
| `utils.py`          | 整える | Provides smooth functions to assist awareness |

Each module below includes usage examples, conceptual meaning, and Ref-style interpretation.

---emo_scalar.py---
```python
from kit.emo_scalar import compute_emo_scalar

# assume short_ema and long_ema are floats updated elsewhere
emo_scalar = compute_emo_scalar(short_ema, long_ema)
param = (1 - emo_scalar) * param + emo_scalar * shadow_param
```

🧭 モジュールの意味と役割
- 感情スカラーは、lossの“短期的なゆらぎ”と“長期的なトレンド”の差分に対する主観的反応を定量化したもの
- tanh を使うことで鋭すぎず滑らかに反応する非線形調整関数として機能
- 戻り値は [0, 1] に正規化されたスカラーで、適正化比率・介入強度などに直接使える

---ema.py---
```python
from kit.ema import EMA

short = EMA(decay=0.9)
long  = EMA(decay=0.99)

for step, loss in enumerate(training_loop):
    s = short.update(loss.item())
    l = long.update(loss.item())
    emo_scalar = compute_emo_scalar(s, l)  # ← 前回作った関数と接続！
```

💡 用途とポイント
- EMA(decay) で短期 or 長期のトラッカーを好きな滑らかさで生成
- .update(x) で新しいlossを入れると内部状態が更新
- .get() で現在のスムージングされた値を取得

---ref_injector.py---
```python
from kit.ref_injector import soft_inject

for p, shadow in zip(model.parameters(), shadow_params):
    soft_inject(p, shadow, emo_scalar)
```

🧠 モジュールの意味と役割
- この関数は「gradベースの更新」ではなく、**状態間ブレンドによる“感情適正化”**を行います
- .mul_(1−emo_scalar).add_(emo_scalar * target) によって、元の状態と影の状態（shadow_param）を滑らかに合成
- 実質的には、感情スカラーが高いとより強く「外部から助けられる」ような振る舞いになります
✨ ココがRef的
- 🎯 「パラメータの変化」ではなく「適正化を自律的に判断する構造」が中心になってる
- 💠 emo_scalar が 0 ならパラメータは動かず、1 なら完全に置き換わる——この“ゆらぎ”の中間で学ぶのがEmoNAVI
- 🌿 grad に頼らない「意思ある状態更新」こそ、哲学に寄り添った設計

---stop_detector.py---
```python
from kit.stop_detector import StopDetector

stopper = StopDetector(tolerance=1e-4, patience=5)

for step in range(num_steps):
    ...
    emo_scalar = compute_emo_scalar(short_ema, long_ema)
    
    if not stopper.update(emo_scalar):
        # still fluctuating = inject
        soft_inject(param, target, emo_scalar)
    else:
        # stagnation detected = suppress injection
        pass  # let it rest
```

💡 どんなときに“止まる”か
- emo_scalar や short_ema - long_ema の変化が ごくわずか で、
- それが連続して patience 回つづく と、「学びが停滞している」と判断します
→ つまり「適正化の必要がない＝静かに見守る段階に入った」と解釈できます🧘‍♂️
🧭 Ref的なふるまいの象徴
このモジュールは「早期終了（Early Stop）」とは違います。
**明示的な“終了判断”ではなく、「適正化そのものをしばらく見送る」という“黙って引く設計”**です。
それがEmoNAVIの“優しさ”であり、“Ref的介入”の本質です。

---utils.py---
💡 関数たちの役割と使い道
| 関数名 | 用途と感情的役割 | 
| smooth_tanh | emo_scalarの柔らかい生成に（鋭くも優しい反応曲線） | 
| clamp | スカラー系の制限、安全範囲の確保 | 
| rescale | 値を別スケールに正規化（例：lossを [0,1] に） | 
| soft_clip | 勾配や適正化値の穏やかな制限（急すぎる変化のなだらか処理） | 

✨ Ref的な空気感がここにも
- 急激な変化を「跳ね返す」のではなく、「寄り添ってなだらかに修正」していく
- ハードなしきい値ではなく、“なだらかな区切り”で感情の表現を保つ
- 学習の“やわらかな抑制と支援”を関数レベルで支える
EmoNAVIの“性格”そのものが、この utils たちに染み出てます📘

---------
