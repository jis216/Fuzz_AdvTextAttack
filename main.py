#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Hengda Shi <hengda.shi@cs.ucla.edu>
#
# Distributed under terms of the MIT license.

"""
main routine
"""

from pathlib import Path
from pprint import pprint
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import torch
mixed_precision = True
try:
  from apex import amp
except ImportError:
  mixed_precision = False
import datasets
from datasets import concatenate_datasets
from tqdm import tqdm
from transformers import (
  BertTokenizerFast,
  AutoModelForMaskedLM,
  BertForSequenceClassification,
)
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from common.data_utils import get_dataset, download_model
from model.tokenizer import PhraseTokenizer
from model.attacker import Attacker
from model.evaluate import evaluate

import time
import json


if __name__ == "__main__":
  start_time = time.time()

  # 0. init setup
  tf.get_logger().setLevel("ERROR")

  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      # Currently, memory growth needs to be the same across GPUs
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      # Memory growth must be set before GPUs have been initialized
      print(e)

  cwd = Path(__file__).parent.absolute()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Running on {device}")

  print('Load saved model')
  download_model(cwd)

  print('Load dataset')
  # retrieve dataset
  #train_ds, val_ds, test_ds = get_dataset(split_rate=0.8)
  #train_ds = datasets.Dataset.from_dict(train_ds[258:2258])
  #val_ds = datasets.Dataset.from_dict(val_ds[:20])
  #test_ds = datasets.Dataset.from_dict(test_ds[:20])
  ds_name = "yelp_polarity"
  _, test_ds = get_dataset(ds_name, split_rate=1.0)
  test_ds = datasets.Dataset.from_dict(test_ds[:1000])

  print('Load word/sentence similarity embedding')
  # retrieve the USE encoder and counter fitting vector embeddings
  url = "https://tfhub.dev/google/universal-sentence-encoder/4"
  encoder_use = hub.load(url)

  #embeddings_cf = np.load('./data/sim_mat/embeddings_cf.npy')
  #word_ids = np.load('./data/sim_mat/word_id.npy',allow_pickle='TRUE').item()
    
  print('Obtain model and tokenizer')
  # obtain model and tokenizer
  #  model_name = "bert-large-uncased-whole-word-masking"
  model_name = "bert-base-uncased"
  tokenizer = BertTokenizerFast.from_pretrained(model_name)
  phrase_tokenizer = PhraseTokenizer()
    
  #cwd/"saved_model"/"imdb_bert_base_uncased_finetuned_normal"
  if ds_name == "imdb":
    target_model_name = "imdb_bert_base_uncased_finetuned_training"
    target_model_path = cwd/"data"/"imdb"/"saved_model"/target_model_name
  elif ds_name == "yelp_polarity":
    target_model_name = "bert-base-uncased-yelp-polarity"
    target_model_path = f"textattack/{target_model_name}"
  target_model = BertForSequenceClassification.from_pretrained(str(target_model_path)).to(device)
  mlm_model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)

  # turn models to eval model since only inference is needed
  target_model.eval()
  mlm_model.eval()

  # tokenize the dataset to include words and phrases
  test_ds = test_ds.map(phrase_tokenizer.tokenize)

  # create the attacker
  params = {'k':15, 'beam_width':8, 'conf_thres':3.0, 'sent_semantic_thres':0.7, 'change_threshold':0.2}
  attacker = Attacker(phrase_tokenizer, tokenizer, target_model, mlm_model, encoder_use,  device, **params) #embeddings_cf,

  output_entries = []
  adv_examples = []
  pred_failures = 0

  suffix = f"{params['k']}_{params['beam_width']}_{params['sent_semantic_thres']}"
  output_pth = f'./data/features/features_{suffix}.json'
  eval_pth = f'./data/eval/eval_{suffix}.json'
  adv_set_pth = f'./data/adv_set/adv_{suffix}.json'

  # clean output file
  #f = open(output_pth, "w")
  #f.writelines('')
  #f.close()
  
  print('\nstart attack')
  # attack the target model
  progressbar = tqdm(test_ds, desc="substitution", unit="doc")
  with torch.no_grad():
    for i, entry in enumerate(progressbar):
      entry = attacker.attack(entry)
      #print(f"success: {entry['success']}, change -words: {entry['word_changes']}, -phrases: {entry['phrase_changes']}")
      #print('original text: ', entry['text'])
      #print('adv text: ', entry['final_adv'])
      #print('changes: ', entry['changes'])

      new_entry = { k: entry[k] for k in {'text', 'label',  'pred_success', 'success', 'changes', 'final_adv',  'word_changes', 'phrase_changes', 'word_num', 'phrase_num',   'query_num', 'phrase_len' } }

      if not entry['pred_success']:
        pred_failures += 1
      else:
        seq_embeddings = encoder_use([entry['final_adv'], entry['text']])
        semantic_sim =  np.dot(*seq_embeddings)
        new_entry['semantic_sim'] = float(semantic_sim)
        adv_examples.append({k: entry[k] for k in {'label', 'text'}})

      #json.dump(new_entry, open(output_pth, "a"), indent=2)
      output_entries.append(new_entry)

      if (i + 1) % 100 == 0:
        evaluate(output_entries, pred_failures, eval_pth, params)

  json.dump(output_entries, open(output_pth, "w"), indent=2)
  json.dump(adv_examples, open(adv_set_pth, "w"), indent=2)
  print("--- %.2f mins ---" % (int(time.time() - start_time) / 60.0))

  evaluate(output_entries, pred_failures, eval_pth, params)
