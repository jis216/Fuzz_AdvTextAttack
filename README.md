# CS 239 Project: Fuzz Natural Lanague Model with Adversarial Examples

This is the repo for the class project of CS 239 Winter 2022

## Setup (without GPU)

```bash
conda create -y --name nlp_fuzz python=3.8.5
conda activate nlp_fuzz
cd /path/to/project/root/
pip install -r requirements.txt
pip install -e .
cd TextAttack
pip install .[dev]
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
cd TextAttack
pip install .[dev]
python -m spacy download en_core_web_lg
```

## Activate Environment

```bash
conda activate nlp_fuzz
```

## Baseline Exeriments
The baseline experiments can all be exectued through running the following script files:

```bash
./a2t.sh
./bae.sh
./textfooler.sh
```

## Our method
To run our method, you can run the following command:

```bash
python main.py --dataset <imdb or yelp_polarity> [--phrase_off]
```
You can specify the dataset to run on, by default it will use yelp_polarity if no dataset option is specified. You can use the ``--phrase_off`` flag to turn off the phrase tokenization. Without this flag, by default it will use phrase tokenization.

You can also use ``python main.py -h`` to get the argument specific usage info. 


## Online Demo
You can run the Demo we've shown in the presentation in this [Colab Notebook](https://drive.google.com/file/d/1hW-PfcingOF1v57cjvX7ygZ-ZhhJoa9Z/view?usp=sharing).

Or just run the local Example.ipynb. Please switch to the correct kernel when running locally.
