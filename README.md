# NAT_base

## Requirements
```
pytorch>=1.7.0
pytorch-lightning==1.2.10
transformers==4.14.1
```

pip install -r requirements.txt

## Data
- train_file : 학습데이터 위치
- valid_file : 검증데이터 위치
- src_lang과 tgt_lang 설정해 파일 선택


## How to Train
- Non-Autoregressive Translation 학습
- max_epochs : epoch 수
- warmup_ratio : warmup 학습 비율
- batch_size : batch 크기
- max_len : 문장의 최대 길이
- lr : learning rate
- gpus : gpu 수

```
sh run_train.sh
```

## How to Generate
- Non-Autoregressive Translation 생성
- hparams : yaml 파일
- model_binary : model 파일'
- testfile : test 파일
- outputfile : 생성할 결과 파일
- length_beam_size : length beam 크기
- src_lang과 tgt_lang 설정해 파일 선택
```
sh inference.sh
```
