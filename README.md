# hand-detection-3class-MobilenetV2-SSDLite
MobilenetV2-SSDLiteで訓練した手検出(Open/Close/Pointer)のモデルです。

以下の3種類（開いた手(Open)、閉じた手(Close)、指さし(Pointer)）の検出を行います。

![2020-04-08 (7)](https://user-images.githubusercontent.com/37477845/78697893-71a72700-793c-11ea-8529-0764ed2f843e.png)

![2020-04-08 (8)](https://user-images.githubusercontent.com/37477845/78697903-75d34480-793c-11ea-8a65-c264b2358df1.png)

![2020-04-08 (10)](https://user-images.githubusercontent.com/37477845/78697913-7966cb80-793c-11ea-9742-531cf9522118.png)

# Requirement
 
* Tensorflow 1.14.0
* OpenCV 3.4.2(sample.pyを動かす場合のみ)

# Usage
 
サンプルの実行方法は以下です。
 
```bash
python sample.py
```

# Note

手検出用の「frozen_inference_graph.pb」もご自由に利用していただいて構いません。

訓練時のデータが少なかったため、学習データを追加し、訓練し直して精度アップ版に差し替える予定です。

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
object-detection-bbox-art is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
