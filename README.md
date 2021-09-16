# 内容
  遺伝的アルゴリズムとHessian-Free法を組み合わせて
  Echo State Networkの構造を探索します．
  誤差が少ないモデルを探索するだけでも
  機能局在がみられるモデル構造が発生する事(機能分化の誘発)
  が分かっています
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
  
# プログラムの種類
　esn_model.py ESNのモデルを宣言，
    
    Binde_ESN_Best_Model　バイアス有り(入力層，Reservoir層)，重みの初期値をHe初期化で決定

    Binde_ESN_Model　バイアス無し，重みの初期値を標準偏差の正規分布

  Hessianfree.py HessianFree法のプログラム

  inputdata.py データセットの作成，cos波の時間，空間情報の分類

  

# トムルファイルを用いたdocker環境の構築

  base_image = "pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel"
  image = "hessian_free_leaky_esn:1.0"

  で最新のプログラムが動きます

  dockerfileでダウンロードパッケージの追加が可能です