<div align="center">

# Extration based Question Answering

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>


</div>

## Description
- [koelectra](https://github.com/monologg/KoELECTRA) 를 korquad v1 + custom kbo dataset 데이터셋에 fine-tuning하여 Question Answering
    - custom kbo dataset 구축은 여기를 참조해주세요.
    - question-answering 데이터셋 구축은 여기를 참조해주세요.
- pretrained electra-small-v3 모델에 start token, end token을 예측하는 custom cnn layer를 추가해서 학습했습니다.


```yaml
# default
python run.py

# train on CPU
python run.py trainer.gpus=0

# train on GPU
python run.py trainer.gpus=1

# train on sample configuration
python run.py experiment=example_simple.yaml
```



<br>
