## 環境構築

1. `docker build -t mmdet-test .` でdocker imageを作成(タグは任意)

2. `docker run -p 8501:8501 --rm -it -v ${PWD}:/work mmdet-test streamlit run object_detection_web_cam_app.py --server.address=0.0.0.0 --server.port=8501` を実行して `http://localhost:8501/` にアクセス(GPU搭載のマシンの場合は、`--gpus all` オプションをつける)
