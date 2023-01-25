# Final Project NLP-08

## develop_summary용 How to run

### Train
```commandLine
python train.py --config {config}
```
- config옵션을 주지 않을 경우, config/base.yaml이 실행됨.
- 실험 환경 세팅은 yaml 파일을 변경함.

### Inference
```commandLine
python test.py --config {config} --data {data_path} --model_path {model_path}
```
- config 옵션을 주지 않을 경우, config/base.yaml이 실행됨.
- data 옵션을 주지 않을 경우, config/base.yaml의 predict_path를 불러옴.
- model_path 옵션을 주지 않을 경우, config/base.yaml의 model_path를 불러옴.