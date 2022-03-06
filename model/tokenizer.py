"""
custom tokenizer
"""
import re

from tqdm import tqdm
import numpy as np
import spacy
from spacy.tokens import Doc
from spacy.matcher import Matcher
from spacy.language import Language
from spacy.tokens import Token
import datasets
from tokenizers import pre_tokenizers
import torch

import benepar
import sys


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

@Language.factory("merge_phrases")
def create_entity_merger(nlp, name):
    return MergePhrases(nlp.vocab)

class MergePhrases:
  def __init__(self, vocab):    
    # instantiate a Matcher instance
    pattern = [[{'POS': 'VERB'}, {'POS': 'PRON', 'OP': '?'}, {'POS': 'ADP', 'OP': '+'}],
            [{'POS': 'ADV'}, {'POS': 'ADV'}, {'POS': 'SCONJ'}],
            [{'POS': 'ADP'}, {'POS': 'ADV'}, {'POS': 'ADV', 'OP': '+'}]]
    self.tag_matcher = Matcher(vocab)

    # matcher.add("Verb ADP ADP phrase", None,  pattern2)
    self.tag_matcher.add("Phrase",  pattern)

    if not Token.has_extension("is_phrase"):
        Token.set_extension("is_phrase", default=False)

  def detect_phrase_spans(self, doc):
    tag_matches = self.tag_matcher(doc)
    phrase_matches = []

    cur_sent_match_end = 0
    last_sent_match_end = 0
    
    for tree in doc.sents:
      tree_str = tree._.parse_string
      while cur_sent_match_end < len(tag_matches) and \
        tag_matches[cur_sent_match_end][-1] <= tree.end:
        cur_sent_match_end += 1

      nodes = tree_str.replace(')', '').split()[1:]

      closest_parent_VP = [0]*len(nodes)
      closest_parent_PP = [0]*len(nodes)

      for i, node in enumerate(nodes):
          if node == "(VP":
              closest_parent_VP[i] = i
          elif i > 0:
              closest_parent_VP[i] = closest_parent_VP[i-1]
      
      for i, node in enumerate(nodes):
          if node == "(PP":
              closest_parent_PP[i] = i
          elif i > 0:
              closest_parent_PP[i] = closest_parent_PP[i-1]

      for phrase_i, (_, start, end) in enumerate(tag_matches[last_sent_match_end:cur_sent_match_end]):
          words = list(doc[start:end])

          phrase_matches.append(tag_matches[phrase_i])

          node_idx = nodes.index(doc[start].text)
          parent_VP = closest_parent_VP[node_idx]
          parent_PP = closest_parent_PP[node_idx]
          for word in words[1:]:
            tree_index = nodes.index(word.text)

            if (closest_parent_VP[tree_index] == parent_VP and tree_index - parent_VP <= 5) \
              or closest_parent_PP[tree_index] == parent_PP:
              continue
            else:
              phrase_matches.pop()
              break
          
      last_sent_match_end = cur_sent_match_end

      # prevent overlaps
      '''
      i = 0
      while i < len(tag_matches)-1:
          # print(i, len(spans), range(len(spans) - 1))
          if tree_str.find(tag_matches[i][-1].text) >= tree_str.find(tag_matches[i + 1][0].text):
              print(tag_matches[i][-1].text, tag_matches[i + 1][0].text)
              second_label = nodes[nodes.index(tag_matches[i + 1][0].text)-1]

              if second_label == '(RP' or second_label == 'IN':
                  tag_matches.remove(tag_matches[i+1])
              elif tag_matches[i].text in tag_matches[i+1].text:
                  tag_matches.remove(tag_matches[i])
              elif tag_matches[i+1].text in tag_matches[i].text:
                  tag_matches.remove(tag_matches[i+1])
              else:
                  tag_matches.remove(tag_matches[i+1])
              continue
          i += 1

      print(spans)
      '''
    return phrase_matches
  
  def check_in_span(self, word_idx, span_chunks):
    for span in span_chunks:
        if word_idx >= span[0] and word_idx < span[1]:
            return True

    '''
    words = sentence[:-1].split()
    print(words)
    chunks = []
    word_idx = 0
    for word in words:
        if self.check_in_span(word_idx, span_chunks): 
            word_idx += (len(word) + 1)
            continue
        else: 
            chunks.append((word_idx, word_idx+len(word)))
            word_idx += (len(word)+1)
        
    phrase_offset_list = sorted(chunks + span_chunks) 
    '''
    return span_chunks
    
  def __call__(self, doc):
    phrase_matches = self.detect_phrase_spans(doc)
    spans = []  # Collect the matched spans here
    for _, start, end in phrase_matches:
      spans.append(doc[start:end])

    with doc.retokenize() as retokenizer:
      for span in spacy.util.filter_spans(spans):
        #if len(span) > 3:
        #  print(len(span), doc)
        retokenizer.merge(span)
        for token in span:
            token._.is_phrase = True  # Mark token as a phrase
    return doc

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

    #spacy.prefer_gpu()
    spacy_processor = spacy.load("en_core_web_lg")
    #spacy_tokenizer.add_pipe(spacy_tokenizer.create_pipe("merge_noun_chunks"))
    spacy_processor.add_pipe('benepar', config={'model': 'benepar_en3'})
    spacy_processor.add_pipe("merge_phrases")
        
    self.spacy_processor = spacy_processor
    self.spacy_processor.tokenizer = self._custom_tokenizer
    print(self.spacy_processor.pipe_names)


  def tokenize(self, entry, label_map=None):
    """tokenize function
    This tokenize function is to be used with the datasets.map function.
    Args:
      entry: a dictionary containing one row in the dataset.
    Returns:
      entry: a dictionary containing transformed and newly added data.
    """
    text = entry['text'].replace('\n', '').lower()

    phrase_doc = self.spacy_processor(text)
    #with self.spacy_tokenizer.disable_pipes(['Matcher']):
    #   entity_doc = self.spacy_tokenizer(text)
    with self.spacy_processor.disable_pipes(['merge_phrases']):
      word_doc = self.spacy_processor(text)

    entry['words'] = [token.text for token in word_doc]
    entry['word_offsets'] = [(token.idx, token.idx+len(token)) for token in word_doc]
    entry['phrases'] = [token.text for token in phrase_doc]
    entry['phrase_offsets'] = [(token.idx, token.idx+len(token)) for token in phrase_doc]
    
    i, j = 0, 0
    entry['n_words_in_phrases'] = [0] * len(entry['phrases'])
    while i < len(word_doc) and j < len(phrase_doc):
      entry['n_words_in_phrases'][j] += 1
      if word_doc[i].idx+len(word_doc[i]) == phrase_doc[j].idx+len(phrase_doc[j]):
        j += 1
      i += 1 

    '''
    #entity merging stuff
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

    '''
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
    return Doc(self.spacy_processor.vocab, words=words, spaces=spaces)
