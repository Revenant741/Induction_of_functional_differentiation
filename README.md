# 内容
  遺伝的アルゴリズムとHessian-Free法を組み合わせて
  Echo State Networkの構造を探索します．
  誤差が少ないモデルを探索するだけでも
  機能局在がみられるモデル構造が発生する事(機能分化の誘発)
  が分かっています
# 実行方法
  docker環境の作成
  `$python3 setup.py`
  docker環境の確認
  `$docker ps -a`
  docker環境のスタート
  `$docker start [コンテナID]`
  docker環境に入る
  `$docker attach [コンテナID]`
  
  一次収束法(Adam)での実行
  `$python3 src/train.py`

  二次収束法(HessianFree法)での実行
  `$python3 src/hessian_train.py`

  遺伝的アルゴリズムでReservoir層の最適な結合を探索(デフォルトでHessianFree法) 
  `$python3 src/ga_train.py`
  
# プログラムの種類

  * model モデルを宣言するプログラム
    
    * esn_model.py esn_modelの宣言

      * Normal_Binde_ESN_Model  普通の拘束条件付きESN

      * Binde_ESN_Execution_Model  内部状態を観測出来るようにした拘束条件付きESN

  * Hessianfree.py HessianFree法のプログラム

  * inputdata.py データセットの作成，cos波の時間，空間情報の分類

  * my_def 自作の関数とかが入っている

    * Analysisi 保存プログラム

    * hessianfree https://github.com/fmeirinhos/pytorch-hessianfree から頂いたHessian-Free関数

    * import_data データ読み込み用の関数

    * Use_Model モデルデータと構造データの読み込み用の関数

  * plot 分析と描画の為の関数

    * ga_acc_loss_plot 遺伝的アルゴリズムの世代毎の精度の描画

    * atracter アトラクタの描画

    * directed_plot ニューロンと重みの接続状態の描画

    * func_loca_ana 分散と相互情報量の両方を描画　指定した個体番号と最高精度個体毎で見れる

    * ga_all_directed_plot 世代全てのニューロンと重みの接続状態の描画

    * ga_mutial_info_all 相互情報量の値の描画，世代毎と最高精度個体毎で見れる

    * heatmap_plot 相互情報量の値をヒートマップで見れる，ニューロン重なり度合いを見れる

    * kernel_map_plot カーネルマップで相互情報量の値が見れる

    * loss_and_acc_plot 学習個体の精度と誤差が見れる

    * mutual_info_plot 相互情報量の算出　最終世代全体と最終世代最高の個体の相互情報量が見れる

    * output_ana 各ニューロンの出力と正解ラベルとの違いがみれる

    * plot_binde　接続構造の行列をプリントしてくれる

    * var_plot 分散を根拠とした機能局在の分析，世代毎と個体毎で見れる

    * weight_conect_heat_map 重みの大きさをヒートマップで分析出来る

  * unimodified 使わなくなったプログラム

    * ga_train_pinpoint 部分的に接続構造の探索を出来るようにしたプログラム，重みの探索と接続構造の探索の影響度合いを分析出来る可能性を持つ

    * ga_train_restart 予期せぬ出来事が起き学習が止まってしまった時に指定した世代から探索をやり直す事ができるプログラム

  * ga_train.py 遺伝的アルゴリズムとHessian_free法を用いたESNの構造探索

  * hessian_train Hessian_free法を用いた重みの探索

  * train.py Adam等を用いた重みの探索
# トムルファイルを用いたdocker環境の構築

  base_image = "pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel"

  image = "hessian_free_leaky_esn:1.0"

  で最新のプログラムが動きます

  dockerfileでダウンロードパッケージの追加が可能です