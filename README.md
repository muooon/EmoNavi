### EmoNAVI / Emo-Family (1stGen-v5.0) 廃止  
### 不安定のため廃止(Discontinued due to instability)  

お詫び：v4.0、v5.0、 では短期修正をしご迷惑ご心配をかけましたことお詫びします  
Apology: We apologize for the inconvenience and concern caused by the short-term fixes in v4.0, v5.0,.  

代わりに v3.1 Test を dev ブランチにて公開しました  
こちらでは emo系 初の 学習加速機能 を補正として実現しています  
その他の見直しや修正により進化したと思います  
(ご期待に沿えるものであれば嬉しいです)  
Instead, I've released v3.1 Test on the dev branch.  
This version implements the first learning acceleration feature in the emo series as a correction.  
I believe it has evolved through other revisions and fixes.  
(I hope it meets your expectations.)  


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

EmoNAVI is an “emotion-driven” approach not found in existing optimizers. By building each sensor around an “emotion mechanism” that differentiates multi-EMA and scalarizes it via nonlinear transformation (tanh), we enhanced overall learning stability and ensured accuracy. This performs an autonomous cycle of “observation, judgment, decision, action, memory, and reflection,” akin to a biological central nervous system. (Please take a look at the paper)

---

EmoNAVIは既存のオプティマイザにはない｢感情駆動型｣です。multi-emaを差分化し非線形変換(tanh)でscalar化した｢感情機構｣を中心に各センサーを構築することで学習全体の安定性を向上させ正確性を確保しました、これらは生物の中枢神経系のように｢観察、判断、決定、行動、記憶、反省｣という自律サイクルを行います(論文をぜひご覧ください)



