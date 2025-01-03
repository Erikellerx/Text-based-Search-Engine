"""
Access and manage Bert reranker.
"""

# Copyright (c) 2024, Carnegie Mellon University.  All Rights Reserved.

import os
import re
import subprocess

import Util
import PyLu
import math
import numpy as np

from Idx import Idx
from QryParser import QryParser

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class RerankerBERTRR:
    
    def __init__(self, parameters):
        
        self._init_cache()
        self.parameters = parameters
        self.bert_modelPath = self.parameters['bertrr:modelPath'] 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.scoreAgg = self.parameters['bertrr:scoreAggregation']
        self.psgLen = int(self.parameters['bertrr:psgLen'])
        self.maxTitleLength = int(self.parameters['bertrr:maxTitleLength']) if 'bertrr:maxTitleLength' in self.parameters else 0
        
        self.bert_max_sequence_length = int(self.parameters['bertrr:maxSeqLength'])
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_modelPath)
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(
            self.bert_modelPath,
            num_labels=1).to(self.device)
        self.bert_model.eval()
            
    
    def rerank(self, queries, results):
        with open('OUTPUT_DIR/bertrr.log', 'w', encoding="utf-8") as f:
            reranked_results = dict()
            
            for qid in results:
                reranked_results[qid] = []
                for _, externalId in results[qid]:
                    internalId = self._get_cache('docid2internal', externalId, lambda: Idx.getInternalDocid(externalId))
                    
                    body_string = self._get_cache('docid2bodyString', internalId, lambda: Idx.getAttribute("body-string", internalId))
                    
                    title_string = None
                    if self.maxTitleLength > 0:
                        title_string = self._get_cache('docid2titleString', internalId, lambda: Idx.getAttribute("title-string", internalId))
                        
                    
                    passages = self._build_passage(body_string, title_string, self.scoreAgg)
                    
                    psgStride = int(self.parameters['bertrr:psgStride']) if 'bertrr:psgStride' in self.parameters else 1
                    for i, p in enumerate(passages):
                        f.write(f'{externalId}.{i*psgStride} {p}\n')

                    bert_score = []
                    for passage in passages:
                        bert_input = self.bert_prepare_tokenized_input(queries[qid], passage)
                        score = self.bert_score_sequence(bert_input)
                        bert_score.append(score)
                    
                    reranked_results[qid].append((self._agg_score(bert_score, self.scoreAgg), externalId))
            
            #sort the reranked results, if the score is the same, sort by externalId
            for qid in reranked_results:
                reranked_results[qid].sort(key=lambda x: (-x[0], x[1]))
            return reranked_results
                    
                
        
        
    
    def _build_passage(self, body_string, title_string, scoreAgg):
        
        
        if scoreAgg == 'firstp':
            body_list = body_string.split()
            passages = [' '.join(body_list[:self.psgLen])]
        
        elif scoreAgg == 'maxp' or scoreAgg == 'avgp':
            psgCnt = int(self.parameters['bertrr:psgCnt'])
            psgStride = int(self.parameters['bertrr:psgStride']) if 'bertrr:psgStride' in self.parameters else 1
            passages = []

            # Tokenizing body
            body_list = body_string.split()
           
            # Iterate through body to create passages with overlap
            for start in range(0, len(body_list), psgStride):
                end = start + self.psgLen
                if len(passages) >= psgCnt:
                    break
                passages.append(' '.join(body_list[start:end]))

            # Remove duplicate passages (cornor case)
            if len(passages) >= 2:
                if passages[-1] in passages[-2]:
                    passages = passages[:len(passages)-1]
                         
        else:
            raise Exception('Error: Unknown score aggregation method: ' + scoreAgg)
        
        if self.maxTitleLength == 0:
            return passages
        
        # add title
        title_list = title_string.split() if title_string else []
        max_title_len = min(self.maxTitleLength, len(title_list))
        passages = [' '.join(title_list[:max_title_len]) + ' ' + p for p in passages]
        
        return passages
        
        
        
    def _agg_score(self, scores, scoreAgg):
        
        if scoreAgg == 'firstp':
            assert len(scores) == 1
            return scores[0]
        elif scoreAgg == 'maxp':
            return max(scores)
        elif scoreAgg == 'avgp':
            return np.mean(scores)
        else:
            raise Exception('Error: Unknown score aggregation method: ' + scoreAgg)
    
    def bert_prepare_tokenized_input(self, query, doc):
        """
        Tokenize a (query, document) pair, convert to token ids, return as tensors.
        Input: query and document strings.
        Output: a dictionary of tensors that BERT understands.
            input_ids: The ids for each token. 
            token_type_ids: The token type (sequence) id of each token.
            attention_mask: For each token, mask(0) or don't mask(1). Not used.
        """
        bert_input =  self.bert_tokenizer.encode_plus(
                [query, doc],		# sequence_1, sequence_2
                add_special_tokens=True,	# Add [CLS] and [SEP] tokens?
                max_length=self.bert_max_sequence_length,
                truncation="only_second",	# If too long, truncate sequence_2
                return_tensors="pt")	# Return PyTorch tensors
        # Move tensors to GPU if available
        bert_input = {k: v.to(self.device) for k, v in bert_input.items()}
        #print(f"\tBERT input: {bert_input}")

        # Display WordPiece tokens. This is just FYI. It is not necessary.
        #tokens = self.bert_tokenizer.convert_ids_to_tokens(bert_input['input_ids'][0])
        #print(f"\tWordPiece tokens: {tokens}")

        return(bert_input)


    def bert_score_sequence(self, input_dict):
        """
        Score a (query, document) pair.
        Input: the tokenized sequence.
        Output: the reranking score.
        """
        with torch.no_grad():
            # Feed the tokenized sequence to the reranker for scoring. 
            outputs = self.bert_model(**input_dict) 
            #print(f"\tBERT Output: {outputs}")
            
            # Extract the classification score and transform to python float.
            score = outputs.logits.data.item()
            return(score)
    
    
    def _init_cache(self):
        self.cache = dict()
    
    
    def _get_cache(self, cacheName, key, value_fn):
        """
        Cache structure:

        Cache = {
            Cache Item: {
                key: value
            }
        }
        
        If cache item does not exist: open a new cache

        If key does not exist in cache item: call value_fn() and store the result

        Finally, return the result
        """
        if cacheName not in self.cache:
            self.cache[cacheName] = dict()
        
        cache = self.cache[cacheName]
        if key not in cache:
            cache[key] = value_fn()
        
        return cache[key]
        
    

if __name__ == "__main__":
    
    body = 'a b c d e f g h i j'
    title = None
    psgCnt = 4
    psgStride = 3
    psgLen = 5
    passages = []

    # Tokenizing title and body
    title_list = title.split() if title else []
    body_list = body.split()
    max_title_len = 0

    # Iterate through body to create passages with overlap
    for start in range(0, len(body_list), psgStride):
        end = start + psgLen
        if len(passages) >= psgCnt:
            break
        passage = title_list[:max_title_len] + body_list[start:end]
        passages.append(' '.join(passage))

    if len(passages) >= 2:
        if passages[-1] in passages[-2]:
            print(passages[:len(passages)-1])
    print(passages)
    
    