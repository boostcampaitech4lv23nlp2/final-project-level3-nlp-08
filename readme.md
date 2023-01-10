# Final Project NLP-08

## develop_summary용 How to run

```
train.sh
get_model_binary.sh
inference.py
```
순서로 진행하면 됩니다.

train.sh: model의 하이퍼파라미터(hparams)와 model_binary를 default_root_dir에 저장
get_model_binary.sh: 하이퍼파라미터와 model_binary를 huggingface form의 from_pretrained method에 사용할 수 있게끔 config.json과 pytorch_model.bin 파일을 생성
inference.py: 코드 6번째 줄의 경로를 get_model_binary.sh의 output_dir에 저장된 모델로 수정해서 사용

각 sh file에 있는 대괄호 내의 내용은 사용자가 직접 수정하여 사용
