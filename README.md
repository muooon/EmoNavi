# EmoNAVI / Emo-Family (1stGen-v3.3)  

### EmoNavi 完成版 (v3.3) The Last Piece  

EmoNavi (v3.3) 完全自動学習率  
v3.1を進化させました(完全自動学習率を"省VRAM･低負荷"で) これで emo系 は完成です  
EmoNavi (v3.3) Fully Automatic Learning Rate  
Evolved from v3.1 (with fully automatic learning rate, saving VRAM and reducing load)  
This completes the emo series.  

updateの内容  
- 動的学習率：完全自動学習率を手軽に気軽に誰でも  
- 感情機構：感情ema-medium で安定と急変を信頼度で感知  

Update Details  
- Dynamic Learning Rate: Fully automatic learning rate made simple and accessible for anyone  
- Emotion Mechanism: Detects stability and sudden changes in emotion using ema-medium with confidence levels  

<div align="center">
  <img width="500" alt="emo-system001" src="https://github.com/user-attachments/assets/7e7160a9-046a-4212-bcde-d338c26ed846" />
</div>

Mathematical Explanation Here (paper)  
非凸関数に対する期待値収束(フローマッチングへの適応なども保証します)  
Expected value convergence for non-convex functions  
(also guarantees adaptability to flow matching)  
#### [emo-paper(article)](https://huggingface.co/muooon/EmoNAVI/raw/main/emo-paper(ENG).txt)  
#### [数学的解説はこちら(論文)](https://huggingface.co/muooon/EmoNAVI/raw/main/emo-paper(JPN).txt)  

---

emo系 v3.3 test (スタンダードモデル) の特徴等  

| 名称      | 正確性 | メモリ負荷 | 非同期 | 備考                                      |  
|-----------|--------|------------|--------|-------------------------------------------|  
| emonavi   | ◎      | △          | ◎      | 最初に誕生｜正確｜Adam型       |  
| emofact   | △      | ◎          | ◎      | ２番目に誕生｜軽量｜Adafactor型 |  
| emolynx   | 〇      | 〇          | ◎      | 軽量＆正確の両立｜Lion型         |  

補足：EmoLynx は EmoFact 並みに軽量で EmoNavi 並みに正確です 

[効率性] 危険抑止更新：過学習や収束の停滞に先回りし無駄な更新を排除しながら進行します  
[機能性] 軽量で高機能：自動停止合図や完全自律型の分散学習への対応でユーザー体験を向上させます  
[信頼性] 安全優先設計：動的制御で学習の不安定な局面でモデルを保護し安定した収束を促します  

常に安全な学習を最優先にし安定させます  
ユーザー指定の学習率を中心にし加減速を自動制御します  
完全自律型のため、積層、再開、非同期、で、自由な学習を自由に組むことが可能です  

emo-series v3.3 test (Standard-models) Features  

| Name    | Accurate | MemoryLoad | Asynchronous | Notes                                           |  
|---------|----------|------------|--------------|--------------------------------------------------|  
| emonavi | ◎        | △          | ◎            | 1st born｜accurate｜Adam-type         |  
| emofact | △        | ◎          | ◎            | 2nd born｜Lightweight｜Adafactor-type |  
| emolynx | 〇        | 〇          | ◎            | Accurate and Lightweight｜Lion-type |  

EmoLynx is as lightweight as EmoFact and as accurate as EmoNavi.  

[Efficiency] Risk-Aware Updates: Proactively prevents overfitting and convergence stagnation while eliminating redundant updates.  
[Functionality] Lightweight and High-Performance: Enhances user experience through automatic stop signals and support for fully autonomous distributed learning.  
[Reliability] Safety-First Design: Protects the model during unstable learning phases with dynamic control, promoting stable convergence.  

Always prioritizes and stabilizes safe learning  
Centers on user-specified learning rates with automatic acceleration/deceleration control  
Fully autonomous, enabling flexible learning configurations through layering, resumption, and asynchronous processing    

---

### 学習の情報、そのすべては Loss値 に集約されている  
#### Learning Information, Everything is consolidated into the Loss value.  

##### Loss値はモデルのshadowである、  
##### Loss値にすべてが集約されている、  
##### 学習状況もモデル状況もLoss値が教えてくれる、  
##### Lossを感じろ、 Lossこそオリジン(原点)だ、  
###### The loss value is the model's shadow.  
###### The loss value embodies everything.  
###### The loss value tells you both the learning status and the model's condition.  
###### Feel the loss. Loss is the origin.  

<details>
 
<summary> emotional moment </summary>  

"emo系 第二世代"にて解明した shadow-system の根幹から抽出しました  
動的学習率による非線形アプローチは時間的な高次momentを形成します  
単stepでは高次momentにはなれませんが、複数stepを経ると機能します  
３次４次５次momentについて厳密な数学的な高負荷計算を回避しつつ  
勾配分布の歪みや鋭さや非対称性変化を捉える核心的な効果を近似しています  
I invented the emotional moment.  
I extracted it from the core of the shadow-system, which was elucidated in the "emo-style second generation."  
The nonlinear approach with a dynamic learning rate forms a temporal higher-order moment.  
A single step cannot become a higher-order moment, but it functions after multiple steps.  
It approximates the core effect of capturing changes in gradient distribution's skewness, kurtosis, and asymmetry, while avoiding strict and computationally intensive mathematical calculations for the third, fourth, and fifth moments.  

---

### あなたの望む最適化 EmoNavi が叶えます  
#### The optimization you seek — EmoNavi makes it possible  
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
##### EmoNavi を中心に、EmoFact、EmoLynx、もあります   
#### Auto-convergence, self-control, autonomous optimizer  
###### It primarily features EmoNavi, along with EmoFact and EmoLynx.  

</details>

---

---  

### EmoNavi 主な特徴 / Main Features of EmoNavi  

---

<details>

過学習や発散を抑制、自己修復的機能をもちます  
学習率やスケジューラも自律調整、モデル自身で判断します  
学習の 再開、追加、積層、等で"引き継ぎ不要"、誰でも簡単です  
分散学習で 他ノード等との"同期不要"、完全自律です  
Self-repairing, with no over-learning or divergence  
Autonomously adjusts learning rate and scheduler, so models make their own decisions  
Resuming, adding, stacking, etc. learning is synchronization-free" and easy for everyone  
Distributed learning enables “no synchronization required” with other nodes, achieving full autonomy.  

EmoNAVI は既存のオプティマイザにはない｢感情駆動型｣です、  
調整の複雑なマルチモーダル学習などの新しい分野の課題への対応も期待できます  
EmoNAVI is “emotion-driven,” which is not the case with existing optimizers,  
We expect it to overcome the challenges we currently face,  
while also addressing challenges in new areas such as multimodal learning with complex coordination  

emo系は、観察、判断、決定、行動、記憶、反省、という自律サイクルを行います  
Emo-based follows an autonomous cycle of   
observation, judgment, decision, action, memory, and reflection.  

高効率性と集積度  
高次moment、量子化補償(Kahan補償と違う制御)、分散･継続学習での独立性、自己修復･モデル修復、  
ハイパーパラメータの自律調整、信頼度フィルタ、更新ステップの有界性、構造的耐性、自己停止、  
動的学習率、動的スケジューラ、動的Rank/Aplha、履歴補償、などを含めた多機能性を、  
追加テンソル不要、計算負荷ほぼなし、step毎に完全適用、時間的積算で実現します  
これらをワンパッケージで実現した高効率性と集積度は安定と安全を最優先します  
※ 高次momentは近似的、動的Rank/Alphaも近似的な効果です  
※ LoRA系技術はノイズをなくしますが微小データも失う場合があります  
※ emo系はノイズを作らず既存ノイズを見つけて修正し微小データを保護します  
※ 量子化補償は今後実用化されるさらに低精度な環境でも柔軟に対応できます  
High Efficiency and Integration  
Multifunctionality, including higher-order moments, Quantization Compensation (Control Different from Kahan Compensation), independence in distributed and continual learning, self-healing and model repair,  
Autonomous hyperparameter tuning, confidence filtering, bounded update steps, structural robustness (or resilience), self-termination,  
dynamic learning rates, dynamic schedulers, dynamic Rank/Alpha, and historical compensation,  
is achieved without additional tensors, with negligible computational overhead, fully applied at every step, and through temporal accumulation.  
The high efficiency and integration realized in this single package prioritize stability and safety above all else.  
※ Higher-order moments are approximative, and the effects of dynamic Rank/Alpha are also approximative.  
※ LoRA-based techniques eliminate noise but may sometimes lose fine-grained data (or subtle details).  
※ Emo-based techniques detect and correct existing noise without generating new noise, thereby preserving fine-grained data.  
※ Quantization compensation offers flexible adaptability even in lower-precision environments expected to be commercialized (or practical) in the future.  

</details>

---

## 学習係数の変化 Change in learning coefficient (v3.x)  
<img width="1000" height="700" alt="学習率グラフ001" src="https://github.com/user-attachments/assets/d730441b-3b0f-4ed2-9e8e-3b3d754d612d" />
このように 動的学習率 として機能します ／ coeff値：1.0 付近は無介入のため更新式の純粋な値になります <br>   
It functions as a dynamic learning rate. ／ coeff value: Around 1.0 represents the pure value of the update formula due to no intervention. <br> 

---

<details>

<summary> 更新履歴 / History </summary>  

|★| EmoNavi、Fact、Lynx、v3.3 (251204) v3.1 を継承し完全自動学習率を実現しました(追加テンソルなし)、感情機構の調整等でさらに安定するよう進化しました  
|★| EmoNavi, Fact, Lynx, v3.3 (251201) Inherits v3.1 and achieves fully automatic learning rate adjustment (without additional tensors), further evolving for greater stability through adjustments to the sentiment mechanism and other enhancements.  

|★| EmoNavi、Fact、Lynx、v3.1 (251201) v3.0 を継承しつつ効率化を進めました。感情機構のスケール調整等で広範なモデルで安定するよう進化しました  
|★| EmoNavi, Fact, Lynx, v3.1 (251201) We built upon v3.0 while enhancing efficiency. Through adjustments like scaling the emotion mechanism, we evolved the model for broader stability across diverse models.  

|★| EmoNavi、Fact、Lynx、Clan、Zeal、Neco、v3.0 (250825) emosens(第２世代)で解明した"高次moment"(近似)のフィードバックを適用(更新) 全て "shadow=False" です  
|★| EmoNavi, Fact, Lynx, Clan, Zeal, Neco, updated to v3.0 (250825), Incorporates (updates) feedback on “higher moments” (approximations) clarified by emosens (2nd generation). All are “shadow=False”  

これ以前は v3.0 レポジトリの更新履歴をご覧ください  
For updates prior to this, please refer to the v3.0 repository update history.  

</details>

---  

emo系 は 生物的反応で進化し続けます  
感覚神経系(multi-EMA)、内分泌系(tanh(scalar))、免疫系(shadow-system)、これらの統合により中枢神経系と自律神経系を形成し、高度な判断と決定を行うという自然的に自律した機構として存在します  

---  

<details>

<summary>EmoNavi v3.3 オプション指定方法<br>
EmoNavi v3.3 Option Settings Guide</summary>  

|||オプション指定方法|||  
●shadow オフ(False にする)：  
use_shadow=False  
●LR Max or Min (1e-3, 1e-8)：  
lr_max=1e-3, lr_min=1e-8  
●eps(0除算防止)：  
eps=1e-8  
●動的学習率と感情スカラー等の現在値を取得(ツール側などから取得する)：  
writer=writer  
外部ツール(TensorBoard等)で値を把握したい場合は Optimizer 初期化時に SummaryWriter を渡してください  
writer = SummaryWriter(log_dir="./runs/emonavi")  
optimizer = EmoNavi(model.parameters(), writer=writer)  
tensorboard --logdir=./runs/emonavi  

|||Usage examples|||  
●Shadow off:  
use_shadow=False  
●LR Max or Min (1e-3, 1e-8)：  
lr_max=1e-3, lr_min=1e-8  
●eps(Division by zero prevention)：  
eps=1e-8  
●Monitor values with external tools (TensorBoard):  
writer=writer  
writer = SummaryWriter(log_dir="./runs/emonavi")  
optimizer = EmoNavi(model.parameters(), writer=writer)  
tensorboard --logdir=./runs/emonavi  

</details>

---

<details>


</details>

---

Emoシリーズは、Adam、Adafactor、Lion、Tiger、等から多くを学びました  
これらの後継ではなく独自の思想や設計による"感情機構"というアプローチにより構築されています  
汎用性・自律性・適応性を重視し新たな最適化や効率化や簡易化を追求しています  
この開発において先人たちの知見に深く感謝しつつ今後も新しい可能性を探究します  
The Emo series has learned much from Adam, Adafactor, Lion, and Tiger.  
Rather than being their successors, it is built upon a unique philosophy and design approach centered on "emotional mechanisms".  
It prioritizes generality, autonomy, and adaptability in pursuit of new paths for optimization, efficiency, and simplicity.  
In its development, we deeply appreciate the insights of those who came before us—and continue to explore new possibilities beyond them. 


### License Apache License 2.0 — see LICENSE for details.  
### ライセンス Apache License 2.0 — 詳細は LICENSE をご覧ください  

##### 🤖 Built with  Copilot + human curiosity(v1.0).  
##### 🤖 Copilot と人間の好奇心のコラボで誕生しました(v1.0)  

---

### 引用について / About citations  

---

このオプテイマイザについて引用をなさる場合は、以下をご紹介ください  
When citing this optimizer, please refer to the following sources:  

Official Code:  
https://huggingface.co/muooon/EmoNAVI  
https://github.com/muooon/EmoNavi  

paper:  
https://huggingface.co/muooon/EmoNAVI/raw/main/emo-paper(ENG).txt  

---

EmoNAVI is an “emotion-driven” approach not found in existing optimizers. By building each sensor around an “emotion mechanism” that differentiates multi-EMA and scalarizes it via nonlinear transformation (tanh), we enhanced overall learning stability and ensured accuracy. This performs an autonomous cycle of “observation, judgment, decision, action, memory, and reflection,” akin to a biological central nervous system. (Please take a look at the paper.)  

---

EmoNAVIは既存のオプティマイザにはない｢感情駆動型｣です。multi-emaを差分化し非線形変換(tanh)でscalar化した｢感情機構｣を中心に、各センサーを構築することで学習全体の安定性を向上させ正確性を確保しました、これらは生物の中枢神経系のように｢観察、判断、決定、行動、記憶、反省｣という自律サイクルを行います(論文をぜひご覧ください)  



