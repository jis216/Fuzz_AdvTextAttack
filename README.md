# CS 239 Project: Fast Natural Language Adversarial Fuzzing

This is the repo for the class project of CS 239 Winter 2022

## Setup (without GPU)

```bash
conda create -y --name nlp_fuzz python=3.8.5
conda activate nlp_fuzz
cd /path/to/project/root/
pip install -r requirements.txt
pip install -e .
python -m spacy download en_core_web_lg
```

## Setup (with GPU)

```bash
conda create -y --name nlp_fuzz python=3.8.5
conda activate nlp_fuzz

cd /path/to/project/root/
# install gpu version with cuda 11.0
pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install -e .
python -m spacy download en_core_web_lg
```
