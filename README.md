# my_esn
ESNの実装テスト

# 実行方法
  docker環境の作成
  `$python3 setup.py`
  docker環境へのマウント
  `$docker start -i my-esn`
  
  一次収束法(Adam)での実行
  `$python3 src/train.py`

  二次収束法(HessianFree法)での実行
  `$python3 src/hessian_train.py`

  遺伝的アルゴリズムでReservoir層の最適な結合を探索(デフォルトでHessianFree法) 
  `$python3 src/ga_train.py`

  計算グラフの表示
  `$tensorboard --logdir=./logs`

  図のプロット

# 内容
  出力層のunitが6で0から3が空間情報,3から4が時間情報を学習する
  

# プログラムの種類
　esn_model.py ESNのモデルを宣言，
    
    Binde_ESN_Best_Model　バイアス有り(入力層，Reservoir層)，重みの初期値をHe初期化で決定

    Binde_ESN_Model　バイアス無し，重みの初期値を標準偏差の正規分布

  Hessianfree.py HessianFree法のプログラム

  inputdata.py データセットの作成，cos波の時間，空間情報の分類

# ga_dataの種類

  best_sp_acc.csv 
  best_tp_acc.csv ５エポックでバイアスありHE初期化仕様

  75best_sp_acc　75エポックで同上

# gpuにおけるバージョンの変更

トムルファイルを変更してください

1080Ti

base_image = "pytorch/pytorch:1.4-cuda10.1-cudnn7-devel"
image = "hessian_free_leaky_esn:1.0"

を
3090で使うようになったので

base_image = "pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel"
image = "hessian_free_leaky_esn:1.0"

3090で実行したモデルを1080TIでは動かせません