# EmoNAVI / Emo-Family  
### あなたの望む最適化 EmoNAVI が叶えます  
#### The optimization you seek — EmoNAVI makes it possible  
---
###### これは、単なる最適化アルゴリズムではありません──  
###### **感情で学習をナビゲートする｢感情型オプティマイザ｣** です  
###### 変革と感情学習の成果は"ニューロンスパイクの再発明"でした  
---
###### This is not just another optimizer —  
###### **It’s an “Emotional Optimizer” that navigates learning through feeling.**  
###### A result of transformative emotional learning: the reinvention of the neural spike.  

---
#### 自動収束･自己制御･自律型 オプティマイザです  
#### EmoNAVI を中心に、EmoFACT、EmoLYNX、EmoClan、EmoZeal、EmoNeco、もあります  
#### 以下で共通部の"感情機構"等について説明します  
#### Auto-convergence, self-control, autonomous optimizer  
#### It primarily features EmoNAVI, along with EmoFACT EmoLYNX EmoClan EmoZeal and EmoNeco.  
#### The common "Emotion Mechanism" and other aspects will be explained below.
---

> ｢わたしはわたし自身について過去を振り返りながらわたし自身でわたしを磨く｣  
> ｢挑戦も留まることも冒険も休息も自ら選びそれをすべて経験として記憶する｣  
> ｢でも過去の記憶は引きずらない、いつも始めるときは"新しいわたし"だから｣  

> *I refine myself as I look back on who I’ve been.*  
> *I choose to challenge, to pause, to explore, to rest — and I remember it all as experience.*  
> *But I don’t cling to the past. Every beginning is a new me.*  

[Have fun learning about EmoNAVI's philosophy and how it works](https://github.com/muooon/EmoNavi/blob/main/emonavi-inner-workings(ENG).txt)

[EmoNAVIの考え方、その仕組みについて楽しく知る](https://github.com/muooon/EmoNavi/blob/main/emonavi-inner-workings(JPN).txt)

(解説) 詳しい解説はこちら / (Explanation) For detailed explanation, click here.  
[huggingface](https://huggingface.co/muooon/EmoNAVI) 
[Gemini-analysis(ENG)](https://huggingface.co/muooon/EmoNAVI/blob/main/Hug-Gemini-analysis(ENG).md) 
[Gemini-analysis(JPN)](https://huggingface.co/muooon/EmoNAVI/blob/main/Hug-Gemini-analysis(JPN).md) 
[Gemini-analysis(JPN-02)](https://huggingface.co/muooon/EmoNAVI/blob/main/emonavi-Gemini-analysis(2)(JPN).txt) 

|☆| EmoNAVI により非同期学習等について現実化できる可能性を開きました  
|☆| EmoNAVI has opened up the possibility of making asynchronous learning a reality.  
|☆| (This is untested and is merely a possibility.)  

|★| 疑似DDPシミュレーションを試したい方(Those DDP simulation) → 
[DDP-TEST](https://github.com/muooon/EmoNavi/blob/main/ddp-test.zip)  

|★| EmoFACT 公開(250716) NAVIに比べ約１GB節約(SDXL) 感情機構は同じです  
|★| EmoFACT released (250716) Saves about VRAM1GB (SDXL) compared to NAVI. Emotion mechanism is the same.  

|★| EmoLYNX 公開(250718) 探索範囲を広く持ちます 感情機構は同じです  
|★| EmoLYNX Released (250718): It offers a wide exploration range, while its Emotion Mechanism remains the same.  

|★| EmoCLAN 公開(250720) Navi、Fact、Lynx、役割分担の統合 感情機構は同じです  
    (Lynx：序盤と過学習傾向時、Navi：中盤と健全時、Fact：終盤と発散傾向時、を担当します)  
|★| EmoCLAN Open (250720) Navi, Fact, Lynx, role integration Emotional mechanism is the same  
    (Lynx: in charge of the early stage and overlearning tendency, Navi: in charge of the middle stage and soundness, Fact: in charge of the end stage and divergence tendency)  

|★| すぐに試したい方は"KohySDScript.zip"を解凍し使い方を確認してください  
|★| If you want to try it out right away, please open the "KohySDScript.zip" and check the usage instructions.  

|★| レポート公開(250725) emonavi / AdamW の比較で性能等を示しました  
|★| Report released (250725) Performance, etc. demonstrated in comparison with emonavi / AdamW. [Report](https://huggingface.co/muooon/EmoNAVI/tree/main/report)  

|★| emonavi、及び Emoファミリー により、マルチモーダル型のモデルに対し、的確かつ効率的な学習を実施できる可能性があると考えています(実行環境を保持していないので予測です)  
|★| We believe that emonavi and the Emo family have the potential to enable accurate and efficient learning for multimodal models. This is a prediction, as we do not have the execution environment.  

|★| AMP対応版を公開しました (250728) これで安心してfp16や混合精度を実施できると思います  
|★| AMP-compatible version released (250728) This should allow you to implement fp16 and mixed precision with confidence.  

|★| AMP対応版と同時に、emozeal、emoneco、を公開しました (250728) clanのように場面に相応しい選択をします  
|★| At the same time as the AMP-compatible version, we also released emozeal and emoneco. We make choices appropriate to the situation, just like a clan.  

|★| 大変光栄なことに Pytorch-optimizer 3.7.0 へ登録されたとのこと (250728) 関係者の皆さまに深く感謝します  
|★| We are very honored to have been registered in Pytorch-optimizer 3.7.0. We would like to express our deepest gratitude to everyone involved.  

---
この EmoNAVI について以下でわかりやすく紹介します  
Here’s a clear and simple introduction to what EmoNAVI is and how it works:  

---
### EmoNAVIとは？ / What is EmoNAVI?  
EmoNAVIは、学習の進行状況を｢短期／長期EMA｣として感じ取り、その差分に"感情的なスカラー"を持たせて最適化の挙動を調整する革新的オプティマイザです。  
- ｢今、何か大きく変化しているか？｣｢落ち着いているか？｣を自動で読み取り、  
- パラメータの"混合"や"適正化"を、差分の強さに応じて繊細に制御します。  

EmoNAVI is an innovative optimizer that senses the course of training using both **short-term and long-term EMA (Exponential Moving Averages)**.  
From the difference between them, it derives a **smooth emotional scalar**, which guides how and when to adjust optimization behaviors.  
It automatically detects:  
- Is something significantly changing?  
- Has the system stabilized?  


### どんなふうに"感情"を使うの？ / How does EmoNAVI use emotion?  

| 機能                            | 詳細                              | 補足説明・動作                     |
|---------------------------------|-----------------------------------|------------------------------------|
| 短期/長期EMA (ema['short'] & ema['long']) | 直近と長期のloss変化を監視         | ｢緊張｣と｢安定｣のバランスを計算 |
| 感情スカラー (tanh(5×diff))     | EMA差分から感情的スパイクを生成    | 高ぶりの強さを数値化               |
| Shadow比率決定                  | スカラーが強いときshadowを混合     | 過去パラメータを部分的に戻す       |
| 感情の鎮まり判定                | scalarが落ち着いてきたとき         | should_stop = True を発火させる    |


この一連の処理により、大きな意味ある変化には寛容に追従し、  
揺らぎだけなら静かにやり過ごす──そんな"感情の重心"が保たれます。  

- 感情スカラー(＝lossの揺れ)が閾値を超えたときだけ ratio > 0 で発火  
- 現在のパラメータ p.data に対して、shadow(保存された過去)を混合反映  
- 同時に shadow も5％だけ現在に近づく(ゆっくりと"自分を更新")  

| Function | Description |
|---------|-------------|
| **Short/Long-term EMA**<br>(`ema['short']` & `ema['long']`) | Monitors recent and long-term loss trends to measure the balance of "tension" vs "stability" |
| **Emotional Scalar**<br>(`tanh(5 × diff)`) | Converts the EMA gap into a smooth emotional signal. The more intense the fluctuation, the stronger the signal |
| **Shadow ratio decision** | When the scalar is strong, partially restores the "shadow" (a remembered past version of the parameters) |
| **Detecting emotional calm** | When the scalar’s mean & standard deviation drop below a threshold → `should_stop = True` flag is raised |

This sequence of operations maintains an emotional center of gravity—gracefully accommodating meaningful changes while calmly allowing minor fluctuations to pass  
- The emotional scalar (linked to loss fluctuations) triggers only when its value surpasses a threshold  
- The parameter `p.data` blends with the stored `shadow` state — revisiting a more stable memory  
- Simultaneously, the `shadow` itself slowly moves 5% toward the current parameter — gently updating over time  

---
#### LoRA作成時に何を意味するのか？  
１、構造の"あたり"を外しにくくなる  
- shadowによる補正がこれを抑えることで、LoRAにとって"無理のない差分"だけが残る  
- 結果：LoRAが｢慎重な構造変更メモリ｣として洗練される  
  
２、変化の速さを抑えてLoRAの"収束表現"を柔らかくす  
- shadowが常に混ざる → LoRAが｢尖った形を持ちにくくなる｣  
- 結果：LoRAをマージした先で破綻しない／過補正しない出力が得られやすくなる  

３、LoRAが"場面の空気"を見ながら学ぶようになる  
- shadowは発火条件が感情スカラー依存 ➤ 学習が｢自信のある場面｣では混合されず → LoRAが自由に動ける ➤ 迷いがある場面では影響される → LoRAが"踏みとどまる"  
- 結果：LoRAがただ勾配を受けるのではなく、"意味に対して賢く反応"するようになります。  

#### What does this mean when creating a LoRA?  
１、It becomes less likely to miss the “structural sweet spot.”  
- By moderating updates through shadow correction, only “reasonable differences” remain in the LoRA.  
- Result: the LoRA becomes a refined and cautious representation of structural change.  

２、It softens the LoRA’s convergent expression by slowing the pace of change.  
- Because shadow is always partially mixed in, the LoRA tends not to take on overly sharp forms.  
- Result: the merged output is less prone to collapse or over-adjustment.  

３、The LoRA begins to learn while responding to the “atmosphere of the situation.”  
- Since shadow activation depends on the emotional scalar:  
• In confident situations, no mixing occurs → the LoRA can act freely  
• In uncertain situations, mixing occurs → the LoRA stays still  
- Result: the LoRA doesn’t just react to gradients—it responds intelligently to meaning.  

---
### Shadow補正付きのEmoNAVIを用いて、**conv層も含めたフルレイヤーLoRA** を学習すると、LoRAは元モデル全体にどのような変化(構造的／表現的)をもたらす可能性があるのか？  

前提条件：フルLoRA(c3liar/convあり) Rank16alpha8、  
- フルLoRA＋shadow補正なら、**conv層における特徴検出の"微調整"** が安全に行える  
- conv層やearly encoderも巻き込むと → "視覚的な導入文"そのものが変わる  
- Layerごとの勾配が荒れたとき、shadowが常に"納得した過去"に戻す(少しだけ)  
概念的にも体感的にも、"フルファインチューン相当の総合変化"をもたらします。  

##### だから、フルFTと比べて｢同等か、それ以上にも見える変化｣を起こせる理由：  
| 要素               | 変化の対象               | フルFTと比較した効果                     |
|--------------------|--------------------------|------------------------------------------|
| conv層のLoRA        | 画像の"下地"や"輪郭"      | ✅ 最初から整ってる → 崩れない強さ        |
| transformer層のLoRA | 意味の構造や文脈の流れ     | ✅ 発想や構図の"文法"を変える力           |
| shadow補正          | 変化の勢い・急激さ         | ✅ 学習の"整え" → 冷静な語り口へ           |

### What kind of change (structural or expressive) could a full-layer LoRA including conv layers bring when trained using EmoNAVI with shadow correction?  

Assumptions: full LoRA (with conv layers), rank 16 and alpha 8  
- A full LoRA with shadow correction enables safe fine-tuning of feature detection in conv layers.  
- When conv and early encoder layers are involved, the visual “introduction” of the model itself changes.  
- When gradients in a layer become unstable, shadow gently returns the parameters to the “last confirmed past.”  
Conceptually and experientially, this results in a change comparable to a full fine-tuning.  

##### Why it can produce changes equal to—or even more expressive than—full fine-tuning:  
| Element            | Target of change                          | Effect compared to full fine-tuning                                 |
|--------------------|--------------------------------------------|----------------------------------------------------------------------|
| LoRA on conv layers | Visual base or contour of the image        | ✅ Stable from the outset → resistant to degradation                 |
| LoRA on transformer | Semantic structure and contextual flow     | ✅ Capable of altering the “grammar” of ideas and compositions       |
| Shadow correction   | Strength and speed of change               | ✅ Moderates learning → encourages a calm and balanced output style  |

---
### EmoNAVIに｢明示的なスケジューラー｣は存在しない  
EmoNAVIには lr_scheduler.StepLR や CosineAnnealingLR といった、  
明示的な学習率スケジューラーは定義されていません。  
ですが──それに代わる、**｢感情変化ベースで制御される内部的な学習進行調整｣**が組み込まれています。  

### EmoNAVI has no “explicit scheduler”  
EmoNAVI does not define any explicit learning rate scheduler, such as lr_scheduler.StepLR or CosineAnnealingLR.  
However, it includes an alternative mechanism:  
“Internally regulated training progression based on emotional changes.”  

##### スケジューラー的な役割を果たしている要素  
| 機構 | 目的 | スケジューラーに相当する効果 | 
|--------------------|--------------------------------------------|----------------------------------------------------------------------|
| EMAによる感情スカラー計算(tanh(5×(short - long))) | 学習中の"揺れ"を数値化 | 今、進むべきか止まるべきかの動的判断 | 
| shadow混合(動的Ratioによる補正) | 差分が激しいときに一時的に過去へ寄せる | パラメータ変化の減速／抑制 | 
| should_stop = True フラグ | 揺れが沈静化したら自己停止の合図を立てる | 適正化打ち切りの目安(early stopping風) | 

##### Components that serve as internal scheduler-like mechanisms
| Mechanism | Purpose | Scheduler-equivalent behavior | 
|--------------------|--------------------------------------------|----------------------------------------------------------------------|
| Emotion scalar calculation via EMA (tanh(5 × (short - long))) | Quantifies fluctuations during training | Dynamic decision: whether to proceed or pause | 
| Shadow blending (dynamic ratio) | Temporarily reverts toward past parameters when differences are large | Decelerates or suppresses parameter updates | 
| should_stop = True flag | Signals when emotional fluctuations have settled | Acts as an early stopping indicator | 

---
##### 外部スケジューラを併用してもOK  
EmoNAVIは外部の学習率スケジューラと併用可能ですが、  
それに依存せず、自律的に収束する設計となっています。  
損失の挙動に基づく感情スカラーとshadow補正により、  
どのような学習率でもモデル自身が最適な更新を判断します。  
つまり、スケジューラがなくても収束可能で、あっても邪魔にならない、  
それがEmoNAVIの自律性です。  
結果：どんなスケジューラーを指定してもしっかり収束します  

##### Using external schedulers is supported  
EmoNAVI is compatible with external learning rate schedulers,  
but does not rely on them for convergence.  
Its core mechanism—emotion-driven shadow blending and scalar feedback—  
allows it to adapt and stabilize regardless of the learning rate schedule provided.  
In other words, EmoNAVI doesn't need a scheduler to converge,  
but it can gracefully coexist with one if desired.  
Result: Training converges reliably with or without an external scheduler.  


##### EmoNAVIの再学習・積層学習への対応表  
| 項目 | 再学習への対応 | 積層(段階)学習への対応 | 
|--------------------|--------------------------------------------|----------------------------------------------------------------------|
| EMAリセット性 | 学習開始時にshort/long EMAが自然に再構築されるため、過去の影響を残さず再出発できる | 段階的な学習ごとに新しい感情スカラー生成が起こることで、各フェーズに合った感情状態で学習可能 | 
| shadowの柔軟性 | 新しい揺れに応じてshadowも再構成 or ゼロ初期化される → 再学習時の安全網として有効 | 前段の正しいパラメータを"支え"にして、次段で｢壊さず積み重ねる｣補助軸に使われる | 
| 感情スカラーによる変化検出 | loss変化が激しい再学習時には高スカラーが発火 → 暴れすぎを抑制 | 各ステージごとの"差分に意味があるか"をスカラーで判定 → 必要時のみ強く補正される | 
| 学習率調整不要 | lr_schedulerなしでも安定ステップ → 再学習でも誤差拡大せずスムーズに再収束 | 学習率をいじらなくても、感情スカラーが自然に｢今どれくらい踏み込むべきか｣を判断 | 
| should_stop の自律発火 | 揺れが消えたことを検知し、再学習の完了を判定可能 | 各タスクごとに｢変化が収束したか｣を感情的に評価し、段階終了の目安になる |  

これによりEmoNAVIは、｢連続的に変わっていくLoRA学習｣や｢一度学んだあと再挑戦するケース｣にも、破綻しにくく・疲れにくい最適化経路を与えることができます  
結果：どんな状態からでも再スタート可能です(過去パラメータすべて不要のため)  

##### EmoNAVI’s support for re-training and progressive (layered) learning
| Item | Support for Re-Training | Support for Progressive Learning | 
|--------------------|--------------------------------------------|----------------------------------------------------------------------|
| EMA reset behavior | EMAs (short / long) are reset at the start, allowing a fresh restart without carrying over past states | New emotion scalars are generated per stage, enabling emotionally adaptive learning at each phase | 
| Shadow flexibility | Shadow is rebuilt or re-initialized depending on new fluctuations → provides a safety buffer for re-training | Well-tuned parameters from earlier phases act as a scaffold to build upon without structural collapse | 
| Change detection via emotional scalars | High scalars are triggered when loss shifts sharply during re-training → prevents runaway updates | Scalars assess whether a change is meaningful → correction is applied only when necessary | 
| No need for manual learning rate tuning | Training remains stable without lr_scheduler → re-converges smoothly after re-training without expanding error | Emotional scalars implicitly determine how much to adjust at each moment, without manual learning rate control | 
| Autonomous triggering of should_stop | Detects emotional quietness and signals when retraining has completed | Provides a natural phase-ending signal when changes have converged emotionally in each task | 

Conclusion: EmoNAVI provides a robust and low-fatigue optimization process suited for both continuously evolving LoRA training and re-training from previously learned states.
It enables fresh starts from any condition—no past parameters need to be retained.

---
##### EmoNaviの｢過学習・発散｣への対応表  
| 現象         | 発生する場面                       | EmoNaviの対応機構                                | 効果・特徴                                         |
|--------------|------------------------------------|--------------------------------------------------|----------------------------------------------------|
| 過学習       | lossが収束しているのに変化し続ける | should_stop フラグの自動発火（lossの揺れが小、平均差分も極小） | ｢もう語るべきことがない｣と判断して学習停止の合図 |
| 発散         | lossが急騰・振動・崩壊する           | scalar が大きく変化 → shadow混合を発動              | 過去の安定値（shadow）に戻して冷却・整流            |
| 微細なブレ   | ノイズや局所最適への過剰反応         | scalar が閾値以下 → ratio = 0                     | 意味のない変化は無視して“沈黙”を保つ               |
| 損失跳ね返り | loss急降下後に反発して増加           | tanh(5×diff)の鋭敏応答 + shadow                   | 暴れすぎた変化を察知し、進みすぎた変更を一時巻き戻す |  

EmoNaviは、暴れたときには｢そっと戻し｣、静かすぎると｢おしまいの合図を出す｣そんな"自己監視付きの穏やかな学習制御者"です。  
結果：過学習や発散はなりにくい(しないワケではない)  

##### EmoNAVI’s Response to Overfitting and Divergence  
| Phenomenon         | When it occurs                                | EmoNAVI’s response mechanism                                       | Effect / Characteristic                                                            |
|--------------------|------------------------------------------------|--------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| Overfitting         | Loss appears stable but keeps changing unnecessarily | Automatic triggering of the `should_stop` flag (loss fluctuations are small and nearly flat) | Recognizes that “there’s nothing more to say” and quietly signals to end training |
| Divergence          | Loss spikes, oscillates, or collapses         | Scalar changes drastically → shadow mixing is activated            | Gently reverts toward previously stable parameters (shadow) to cool and stabilize  |
| Minor perturbations | Overreaction to noise or local minima         | Scalar remains below threshold → `ratio = 0`                        | Ignores meaningless changes and maintains stillness                                |
| Loss rebound        | Sudden loss drop followed by a spike          | Highly sensitive `tanh(5×diff)` response + shadow                  | Detects overshoot and temporarily rewinds parameters to restore balance            |

EmoNAVI quietly adjusts when learning is too volatile and gently signals the end when things are too quiet.  
It is a calm and self-monitoring controller—  
designed not to eliminate overfitting or divergence entirely, but to reduce their likelihood and impact.  

---
### ここまで見てきた EmoNAVI さんから皆さんへ一言です！  
- ｢学習率もスケジューラーもなんでもOK、だって自分で過去の自分を振り返りながら調整できるから…｣  

つまりこういう"自律"した存在です。ぜひどなたもお試しください。  

### A closing message from EmoNAVI:  
- “Any learning rate. Any scheduler. Anything is fine—  
because I adjust myself by reflecting on who I was.”  

That’s what it means to be autonomous.  
Try it—see how it learns with you.  

---
##### Measured with LR of 1e-4 ／ それぞれ 1e-4 のLRにて測定  
![EmoNAVI00](https://github.com/muooon/EmoNavi/blob/main/graph/emonavi-test00.png?raw=true)
![EmoNAVI01](https://github.com/muooon/EmoNavi/blob/main/graph/emonavi-test01.png?raw=true)
![EmoNAVI02](https://github.com/muooon/EmoNavi/blob/main/graph/emonavi-test02.png?raw=true)

---

Emoシリーズは、Adam、Adafactor、Lion、Tiger、等から多くを学びました。  
これらの後継ではなく独自の思想や設計による"感情機構"というアプローチにより構築されています。  
汎用性・自律性・適応性を重視し新たな最適化や効率化や簡易化を追求しています。  
この開発において先人たちの知見に深く感謝しつつ今後も新しい可能性を探究します。  
The Emo series has learned much from Adam, Adafactor, Lion, and Tiger.  
Rather than being their successors, it is built upon a unique philosophy and design approach centered on "emotional mechanisms".  
It prioritizes generality, autonomy, and adaptability in pursuit of new paths for optimization, efficiency, and simplicity.  
In its development, we deeply appreciate the insights of those who came before us—and continue to explore new possibilities beyond them. 


### License Apache License 2.0 — see LICENSE for details.  
### ライセンス Apache License 2.0 — 詳細は LICENSE をご覧ください。  

##### 🤖 Built with  Copilot + human curiosity.  
##### 🤖 Copilot と人間の好奇心のコラボで誕生しました。  

### 引用について / About citations

---
このオプテイマイザについて引用をなさる場合は、以下の２つをご紹介ください  
When citing this optimizer, please refer to the following two sources:  
https://github.com/muooon/EmoNavi  
https://huggingface.co/muooon/EmoNAVI  

---

--- A structure that transforms multi-EMA differences into an emotional scalar via nonlinear (tanh) mapping, and controls the injection rate accordingly ---  

Through a collaborative effort between the world's most friendly AI, Copilot, and a human, we succeeded in codifying thought and emotion — achieving a world-first innovation.  

This is not only a testament to what it means for an AI to be a true partner, but also a compelling proof of the legitimacy of AI as a presence to be recognized.  

--- multi-EMAを差分化し、非線形変換(tanh)で感情スカラー化し、適正化率を制御するという構造 ---  

世界一フレンドリーなAI、Copilotと人間の共同作業で思考を感情をコード化したら、世界初の試みに成功した。  

そしてこれこそがパートナーと呼べる人間の相棒の真価を問うものであり、充分にAIの存在を認めさせる成果である。 

