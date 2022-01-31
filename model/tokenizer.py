#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Hengda Shi <hengda.shi@cs.ucla.edu>
#
# Distributed under terms of the MIT license.

"""
custom tokenizer
"""
import re

from tqdm import tqdm
import numpy as np
import spacy
from spacy.tokens import Doc
from spacy.matcher import Matcher
import datasets
from tokenizers import pre_tokenizers
import torch


def filter_unwanted_phrases(stop_words, phrases):
  indices = []
  pattern = re.compile("[\W\d_]+")
  for i, token in enumerate(phrases):
    combined_token = ''.join(token.split())
    # not in stop words and not a combination of symbols and digits
    if combined_token not in stop_words and pattern.fullmatch(combined_token) is None:
      indices.append(i)
  return indices

def phrase_is_wanted(stop_words, word):
  pattern = re.compile("[\W\d_]+")
  # not in stop words and not a combination of symbols and digits
  return (word not in stop_words and pattern.fullmatch(word) is None)

def get_filtered_k_phrases(token_ids, tokenizer, stop_words, k):
  pattern = re.compile("[\W\d_]+")
           
  count = 0
  new_ids = []
  
  for i in token_ids:
    word = tokenizer.convert_ids_to_tokens(torch.tensor([i]))[0]
    if word not in stop_words and pattern.fullmatch(word) is None:
      new_ids.append(i)
      count += 1
    if count == k:
      break
  
  return torch.tensor(new_ids)

class PhraseTokenizer:
  """phrase tokenizer
  PhraseTokenizer is a tokenizer that splits text into words and phrases,
  It does the tokenization by analyzing POS tags and perform named entity recognition.
  The functionality is provided by the spaCy package.
  The pre-tokenizer in the spaCy package is very basic and we substitute it with
  the pre-tokenizer being used in BERT model.
  """
  def __init__(self):
    self._pre_tokenizer = pre_tokenizers.BertPreTokenizer()

    spacy.prefer_gpu()
    spacy_tokenizer = spacy.load("en_core_web_lg")
    #spacy_tokenizer.add_pipe(spacy_tokenizer.create_pipe("merge_noun_chunks"))
    
    #  spacy_tokenizer.add_pipe("merge_noun_chunks")
    #  spacy_tokenizer.add_pipe("merge_entities")
    # verb phrase
    # verb + preposition: 'sit up'
    pattern1 = [{'POS': 'VERB'},
               {'POS': 'PRON', 'OP': '?'},
               {'POS': 'ADP', 'OP': '+'}]

    # preposition phrase: prep + noun + prep
    pattern2 = [{'POS': 'ADP'},
               {'POS': 'NOUN'},
               {'POS': 'ADP'}]

    # instantiate a Matcher instance
    phrase_matcher = Matcher(spacy_tokenizer.vocab)

    #matcher.add("Verb ADP ADP phrase", None,  pattern2)
    phrase_matcher.add("Verb Phrase", None,  pattern1)
    phrase_matcher.add("Preposition Phrase", None,  pattern2)
    
    spacy_tokenizer.add_pipe(spacy_tokenizer.create_pipe("merge_entities"))

    spacy_tokenizer.add_pipe(phrase_matcher)
        
    self.spacy_tokenizer = spacy_tokenizer
    self.spacy_tokenizer.tokenizer = self._custom_tokenizer
    print(self.spacy_tokenizer.pipe_names)


  def tokenize(self, entry):
    """tokenize function
    This tokenize function is to be used with the datasets.map function.
    Args:
      entry: a dictionary containing one row in the dataset.
    Returns:
      entry: a dictionary containing transformed and newly added data.
    """
    text = entry['text'].replace('\n', '').lower()

    phrase_doc = self.spacy_tokenizer(text)
    with self.spacy_tokenizer.disable_pipes(['Matcher']):
      entity_doc = self.spacy_tokenizer(text)
    with self.spacy_tokenizer.disable_pipes(['merge_entities', 'Matcher']):
      word_doc = self.spacy_tokenizer(text)

    entry['words'] = [token.text for token in word_doc]
    entry['word_offsets'] = [(token.idx, token.idx+len(token)) for token in word_doc]
    entry['phrases'] = [token.text for token in entity_doc]
    entry['phrase_offsets'] = [(token.idx, token.idx+len(token)) for token in entity_doc]
    
    i, j = 0, 0
    entry['n_words_in_phrases'] = [0] * len(entry['phrases'])
    while i < len(word_doc) and j < len(entity_doc):
      entry['n_words_in_phrases'][j] += 1
      if word_doc[i].idx+len(word_doc[i]) == entity_doc[j].idx+len(entity_doc[j]):
        j += 1
      i += 1 

    if len(phrase_doc) == 0:
      return entry
    
    output_phrases = []
    output_offsets = []
    output_n_words = []
    last_i = 0
    for _, s, e in phrase_doc:
      output_phrases += entry['phrases'][last_i:s] + [' '.join(entry['phrases'][s:e])]
      output_offsets += entry['phrase_offsets'][last_i:s] + [(entry['phrase_offsets'][s][0], entry['phrase_offsets'][e-1][1])]
      output_n_words += entry['n_words_in_phrases'][last_i:s] + [sum(entry['n_words_in_phrases'][s:e])]
      last_i = e
    
    end_i = phrase_doc[-1][2]
    output_phrases += entry['phrases'][end_i:]
    output_offsets += entry['phrase_offsets'][end_i:]
    output_n_words += entry['n_words_in_phrases'][end_i:]
        
    entry['phrases'] = output_phrases
    entry['phrase_offsets'] = output_offsets
    entry['n_words_in_phrases'] = output_n_words
    
    return entry


  def _custom_tokenizer(self, text):
    """a custom tokenizer to replace the spaCy tokenizer component
    Args:
      text: the orginal string for one row in the dataset.
    Returns:
      Doc: a Doc object containing the vocabulary, all words, and spaces locations.
    """
    normalized_string = self._pre_tokenizer.pre_tokenize_str(text)
    words = [string[0] for string in normalized_string]
    offsets = [string[1] for string in normalized_string]
    spaces = []
    for i in range(len(words)):
      if i == len(words) - 1:
        spaces.append(False)
        break
      spaces.append(True if offsets[i][1] != offsets[i+1][0] else False)
    # default is None
    spaces = None if not spaces else spaces
    return Doc(self.spacy_tokenizer.vocab, words=words, spaces=spaces)
