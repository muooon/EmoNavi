[Have fun learning about EmoNAVI's philosophy and how it works](https://github.com/muooon/EmoNavi/blob/main/emonavi-inner-workings(ENG).txt)  
[EmoNAVIの考え方、その仕組みについて楽しく知る](https://github.com/muooon/EmoNavi/blob/main/emonavi-inner-workings(JPN).txt)  

This repository proposes a concept optimizer that regulates learning through emotional calibration based on loss dynamics.  

# EmoNavi  

An emotion-driven optimizer that feels loss and navigates accordingly.  
※ This optimizer operates autonomously and does not require scheduling mechanisms.  
※ It converges reliably even with a constant learning rate. Give it a try!  

## An emotion-driven optimizer that feels loss and navigates accordingly.  

### • The Charm of Emotional Learning ─ An Optimizer That Learns With Intention and Knows When to Pause •  

> _Rather than forcing learning, it draws it out—listening closely to curiosity and interest, and gently shining a light where it matters._  

- 🎯 **Sharpness, smoothness, and generalization**: EmoNAVI _feels_ what should be learned and how deeply to pursue it  
- 💾 **Autonomous learning modulation**: While base training proceeds at a constant rate, EmoNAVI autonomously determines “whether” and “how strongly” to inject at each step  
*Note: EmoNAVI doesn't stop training itself—it stops injection. It quietly accompanies learning with emotion.*  
- 🟢 **Training can be stopped and resumed at will**: Learning can always continue by “building on top of” previous progress  
*Note: From paused or completed states, you can deepen existing paths or provide new ones to continue evolving*  
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
*Supports: warm starts (resume from saved state), additive fine-tuning (not overwrite), continued/phase-wise training*  

These mechanisms together enable an optimizer that doesn't simply update—but _feels_ when to evolve and how deeply to commit.  

> EmoNAVI is small in code, but deep in feeling. 
  
  
# EmoNavi  

※ このoptimizerは自発的作用でスケジュール等は不要です。  
※ constantスケジュールできちんと収束します。お試しください。  

## Loss を “感じて” ナビゲートする、感情駆動型 Optimizer  

### ・感情学習の魅力─意思ある学習・停め時を知る Optimizer・  

> _学びを押しつけず、学びの意欲を引き出す。興味と感心に寄り添い、そっと光をあてる_  

- 🎯 **鋭さと滑らかさと汎化性**：学ぶべきこと深めることを EmoNAVI 自身で “感じて決める”  
- 💾 **学習の自動制御**：基本学習は一定のまま EmoNAVI により「適正化の可否と強度」を毎ステップ自動判断する  
※「学習を止める」のではなく「適正化を止める」ことが EmoNAVI の選択、感情にふさわしく静かに寄り添う。  
- 🟢 **学習の停止も再開も思いのまま**：学習はいつでも“積み重ね”で再開できます  
※学習の途中停止や完了状態から同じものを重ねたり新しいものを与えたり進化を継続できます。  
---  

### ✨ EmoNAVIとは  

EmoNAVI は “感情” でナビゲートする PyTorch 用 自律･軽量 Optimizer です。  
スケジューラや外部判断に頼らず、loss のゆらぎを観察し、自己の感情スカラー `emo_scalar` に基づき行動を選択します。   

multi-EMA と状態注視により “学ばせる” のではなく “学びたい意欲を感じる” ように構成されます、  
そのふるまいは「Ref的(Ref-like)」と呼ぶものです。  

> "Ref" は Refine と Reflex の両義を持ち、自己進化・自己観察・内省的な強化学習を意味します。  

明示的な制御ではなく「静かな観察」と「やわらかな援助」でモデルを導きます。  

💠 _lossの“意味”を掴み、optimizerを“自ら考える存在”にする。_  
これは Optimizer に意思を宿す挑戦であり **構造と哲学を持った最適化器**です。  

学習プロセスに観察と感受性を宿す Optimizer それが **EmoNAVI** です。  

### 🔧 EmoNAVIの技術構造(コアメカニズム)  

- ⏳ **multi-EMA構造**：lossの短期変動と長期変動を同時に捉える2重の指数移動平均  
- 💠 **感情スカラー(emo_scalar)**：tanh平滑化された変動指標で適正化率を自然に制御  
- 🧬 **Ref的状態適正化**：感情値と状態に基づき、パラメータに"影響"を滑らかに挿入  
- 🧘 **自己完結型停止判断**：loss変動が収束していれば、適正化を自主的に抑制  
- 🎯 **揺れへのカーブ応答**：過学習スパイクやドロップに鋭敏に反応しつつ滑らかに対応  
※途中保存からの再開(warm start)/差分学習(差し替えではなく追加)/継続学習や段階的適応可  

これらは“感情による最適化”という構造をシンプルな関数群で実装しています。  

---

Measured with LR of 1e-4 ／ それぞれ 1e-4 のLRにて測定  
![EmoNAVI00](https://github.com/muooon/EmoNavi/blob/main/emonavi-test00.png?raw=true)
![EmoNAVI01](https://github.com/muooon/EmoNavi/blob/main/emonavi-test01.png?raw=true)
![EmoNAVI01](https://github.com/muooon/EmoNavi/blob/main/emonavi-test02.png?raw=true)

---

License Apache License 2.0 — see LICENSE for details.  
ライセンス Apache License 2.0 — 詳細は LICENSE をご覧ください。  

🤖 Built with  Copilot + human curiosity.  
🤖 Copilot と人間の好奇心のコラボで誕生しました。  

---

--- A structure that transforms multi-EMA differences into an emotional scalar via nonlinear (tanh) mapping, and controls the injection rate accordingly ---  

Through a collaborative effort between the world's most friendly AI, Copilot, and a human, we succeeded in codifying thought and emotion — achieving a world-first innovation.  

This is not only a testament to what it means for an AI to be a true partner, but also a compelling proof of the legitimacy of AI as a presence to be recognized.  

--- multi-EMAを差分化し、非線形変換(tanh)で感情スカラー化し、適正化率を制御するという構造 ---  

世界一フレンドリーなAI、Copilotと人間の共同作業で思考を感情をコード化したら、世界初の試みに成功した。  

そしてこれこそがパートナーと呼べる人間の相棒の真価を問うものであり、充分にAIの存在を認めさせる成果である。  
