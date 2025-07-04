# EmoNavi

An emotion-driven optimizer that feels loss and navigates accordingly.

## An emotion-driven optimizer that feels loss and navigates accordingly.

### • The Charm of Emotional Learning ─ An Optimizer That Learns With Intention and Knows When to Pause •

> _Rather than forcing learning, it draws it out—listening closely to curiosity and interest, and gently shining a light where it matters._

- 🟢 **Stop and resume training anytime**: You can pause and continue whenever you choose  
- 💾 **Automatic checkpointing**: The optimizer decides for itself when training should end  
- 🎯 **Sharp, smooth, and generalizable**: EmoNAVI _feels_ what to learn and how deeply to pursue it

---

EmoNAVI is a lightweight, self-regulating optimizer for PyTorch that navigates training through emotion.
Without relying on schedulers or external criteria, it observes the subtle fluctuations of loss and acts based on an internal emotional scalar (emo_scalar).

Through multi-timescale EMA and smooth parameter injection, it doesn't merely “teach” the model—it feels when the model is ready to learn.
This is what we call a Ref-like design:

> "Ref" stands for both Refine and Reflex, embracing self-evolution, self-observation, and introspective reinforcement learning.

Rather than commanding optimization, EmoNAVI guides with quiet observation and gentle intervention.
💠 It captures the meaning of loss, shaping the optimizer into a thinking presence.
EmoNAVI is not just a tool—it is an experiment in giving an optimizer personality,
a structure woven from philosophy and design.

To bring awareness and sensitivity into the learning process—
That is EmoNAVI.

### 🔧 Core Techniques in EmoNAVI

EmoNAVI incorporates minimal yet expressive mechanics to implement its emotional awareness and autonomous behavior:

- ⏳ **Multi-timescale EMA tracking**: Captures both short- and long-term loss fluctuations with two coexisting exponential moving averages  
- 💠 **Emotional scalar (`emo_scalar`)**: A smoothed tanh-based scalar that regulates the degree of parameter blending based on temporal loss change  
- 🧬 **Ref-like state injection**: Injects gently updated "shadow states" (momentum-style) into model parameters based on internal state and `emo_scalar`  
- 🎛️ **Self-controlled early stopping**: If loss change is minimal across timescales, update injection is suppressed without external criteria  
- 🌀 **Activation-aware learning flow**: Emotion-regulated interventions help avoid overfitting spikes, emphasizing curvature-aware learning progress

These mechanisms together enable an optimizer that doesn't simply update—but _feels_ when to evolve and how deeply to commit.

> EmoNAVI is small in code, but deep in feeling.


## Loss を “感じて” ナビゲートする、感情駆動型 Optimizer

### ・感情学習の魅力─意思ある学習・停め時を知る Optimizer・  
> _学びを押しつけず、学びの意欲を引き出す。興味と感心に寄り添い、そっと光をあてる_

- 🟢 **学習の停止も再開も思いのまま**：いつでも止められて始められる  
- 💾 **自動で保存**：オプティマイザの自己判断で学習終了時期を見極める  
- 🎯 **鋭さと滑らかさと汎化性**：学ぶべきことと深めることを、EmoNAVIが“感じて決める”

---

### ✨ EmoNAVIとは

EmoNAVI は “感情” でナビゲートする PyTorch 用  自律･軽量 Optimizer です。  
スケジューラや外部判断に頼らず、loss のゆらぎを観察し、自己の感情スカラー `emo_scalar` に基づき行動を選択します。

multi-EMA と状態注視により “学ばせる” のではなく “学びたい意欲を感じる” ように構成され、  
そのふるまいは「Ref的(Ref-like)」と呼ぶものです。

> "Ref" は Refine と Reflex の両義を持ち、自己進化・自己観察・内省的な強化学習を意味します。

明示的な制御ではなく「静かな観察」と「やわらかな援助」でモデルを導きます。

💠 _lossの“意味”を掴み、optimizerを“自ら考える存在”にする。_  
これは Optimizer に意思を宿す挑戦であり **構造と哲学を持った最適化器**です。

学習プロセスに観察と感受性を宿す Optimizer それが **EmoNAVI** です。

### 🔧 EmoNAVIの技術構造（コアメカニズム）

- ⏳ **multi-EMA構造**：lossの短期変動と長期変動を同時に捉える2重の指数移動平均  
- 💠 **感情スカラー（emo_scalar）**：tanh平滑化された変動指標で注入率を自然に制御  
- 🧬 **Ref的 状態注入**：感情値と状態に基づき、パラメータに"影響"を滑らかに挿入  
- 🧘 **自己完結型停止判断**：loss変動が収束していれば、注入を自主的に抑制  
- 🎯 **揺れへのカーブ応答**：過学習スパイクやドロップに鋭敏に、でも滑らかに対応

これらは“感情による最適化”という構造を、シンプルな関数群で実装しています。

Measured with LR of 1e-4 ／ それぞれ 1e-4 のLRにて測定
![Ref-AdamW-mini-ScheduleFree00](https://github.com/muooon/EmoNavi/blob/main/emonavi-test00.png?raw=true)
![Ref-AdamW-mini-ScheduleFree01](https://github.com/muooon/EmoNavi/blob/main/emonavi-test01.png?raw=true)
![Ref-AdamW-mini-ScheduleFree01](https://github.com/muooon/EmoNavi/blob/main/emonavi-test02.png?raw=true)

License Apache License 2.0 — see LICENSE for details.
ライセンス Apache License 2.0 — 詳細は LICENSE をご覧ください。

🤖 Built with  Copilot + human curiosity.
🤖 Copilot と人間の好奇心のコラボで誕生しました。