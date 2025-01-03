"""
Access and manage Diversity reranker.
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

class RerankerDiversity:
    """
    RerankerDiversity
    """

    def __init__(self, parameters):
        
        self._init_cache()
        self.parameters = parameters
        
        self.N = Idx.getNumDocs()
        self.intents_table, self.num_intents = self._parse_intent()
        self.score_table = dict()
        self.lambda_ = float(self.parameters["diversity:lambda"])
    def rerank(self, queries, results):
        
        #get the score for each doc of each intern
        for qid in results:
            self.score_table[qid] = dict()
            for score, externalId in results[qid]:
                internalId = self._get_cache('docid2internal', externalId, lambda: Idx.getInternalDocid(externalId))
                
                if "0" not in self.score_table[qid]:
                    self.score_table[qid]["0"] = []
                self.score_table[qid]["0"].append(score)
                
                for intent_id in self.intents_table[qid]:
                    intent = self.intents_table[qid][intent_id]
                    intent_score = self._bm25(internalId, intent)
                    if intent_id not in self.score_table[qid]:
                        self.score_table[qid][intent_id] = []
                    self.score_table[qid][intent_id].append(intent_score)
        # Scaling document scores
        for qid in self.score_table:
            # Check if any score is greater than 1.0 for this query and its intents
            should_scale = any(
                score > 1.0 
                for intent_id in self.score_table[qid] 
                for score in self.score_table[qid][intent_id]
            )
            
            if should_scale:
                max_sum = -1
                for intent_id in self.score_table[qid]:
                    if self.parameters["diversity:algorithm"] == "PM-2" and intent_id == "0":
                        continue

                    max_sum = max(max_sum, sum(self.score_table[qid][intent_id]))
                
                for intent_id in self.score_table[qid]:
                    self.score_table[qid][intent_id] = [score / max_sum for score in self.score_table[qid][intent_id]]
        
        reranked_results = dict()
        if self.parameters["diversity:algorithm"] == "xQuAD":
            for qid in results:
                reranked_results[qid] = self._xQuAD(qid, results[qid])
        elif self.parameters["diversity:algorithm"] == "PM-2":
            for qid in results:
                reranked_results[qid] = self._PM_2(qid, results[qid])
        else:
            raise Exception("Invalid diversity algorithm")
        
        return reranked_results
        
    def _xQuAD(self, qid, results):
        S = []
        R = set()
        id2idx = dict()
        
        for idx, (score, externalId) in enumerate(results):
            R.add(externalId)
            id2idx[externalId] = idx

        while len(S) < min(int(self.parameters["rerankDepth"]), len(results)):
            temp = []
            
            for externalId in R:
                idx = id2idx[externalId]
                
                #P(d|q)
                relevance_score = self.score_table[qid]["0"][idx]
                
                diversity_score = 0
                for intent_id in self.intents_table[qid]:
                    # P(q_i|q)
                    intent_weight = 1/self.num_intents[qid] 
                    # P(d|q_i)
                    intent_score = self.score_table[qid][intent_id][idx]
                    
                    # \prod(1 - P(d_j|q_i))
                    covered = 1
                    for _, convered_externalId in S:
                        covered *= (1 - self.score_table[qid][intent_id][id2idx[convered_externalId]])
                    
                    diversity_score += intent_weight * intent_score * covered

                final_score = (1 - self.lambda_) * relevance_score + self.lambda_ * diversity_score
                temp.append((final_score, externalId))
            
            temp.sort(key=lambda x: -x[0])
            d = temp[0]
            S.append(d)
            R.remove(d[1])
        
        return S
    
    
    def _PM_2(self, qid, results):
        S = []
        R = set()
        id2idx = dict()
        v, s = [], []
        for i in range(self.num_intents[qid]):
            v.append(1/self.num_intents[qid] * int(self.parameters["rerankDepth"]))
            s.append(0)
        
        for idx, (score, externalId) in enumerate(results):
            R.add(externalId)
            id2idx[externalId] = idx

        while len(S) < min(int(self.parameters["rerankDepth"]), len(results)):
            
            qt = [v[i] / (2 * s[i] + 1) for i in range(self.num_intents[qid])]
            
            # argmax the qt, if there are multiple max, choose the one with the smallest index
            max_qt = max(qt)
            priority_intent_id = str(qt.index(max_qt) + 1)
            #print(qid, priority_intent_id)
            assert priority_intent_id in self.intents_table[qid]
            
            temp = []
            for externalId in R:
                idx = id2idx[externalId]
                
                priority_score = 0
                covered = 0
                for intent_id in self.intents_table[qid]:
                    if intent_id == priority_intent_id:
                        priority_score = qt[int(intent_id) - 1] * self.score_table[qid][intent_id][idx]
                    else:
                        covered += qt[int(intent_id) - 1] * self.score_table[qid][intent_id][idx]
                
                final_score = self.lambda_ * priority_score + (1 - self.lambda_) * covered
                temp.append((final_score, externalId))
            
            temp.sort(key=lambda x: -x[0])
            d = temp[0]
            S.append(d)
            R.remove(d[1])
            
            #update s
            idx = id2idx[d[1]]
            total_intent = 0
            for intent_id in self.intents_table[qid]:
                total_intent += self.score_table[qid][intent_id][idx]
                
            for i in range(self.num_intents[qid]):
                if total_intent != 0:
                    s[i] += self.score_table[qid][str(i + 1)][idx] / total_intent
                else:
                    s[i] += self.score_table[qid]['0'][idx]

        # Ensure monotonically decreasing scores
        for i in range(1, len(S)):
            if S[i][0] >= S[i-1][0]:
                S[i] = (S[i-1][0] * 0.999, S[i][1])
        return S
    
    
    def _bm25(self, docid, queryStems, field = 'body'):
    
        term_vector = self._get_cache('term_vector', (docid, field), lambda: Idx.getTermVector(docid, field))
        if term_vector.stemsLength() == 0:
            return None
        
        score = 0.0
        for stem in queryStems:
            if stem in term_vector.stems:
                score += self._get_bm25_score(stem, term_vector, docid, field)
            
        return score



    def _get_bm25_score(self, stem, term_vector, docid, field):
        
        stem_idx = self._get_cache('docid-field-stem2idx', (docid, field, stem), lambda: term_vector.indexOfStem(stem))
        
        tf = self._get_cache('docid-field-stem2tf', (docid, field, stem), lambda: term_vector.stemFreq(stem_idx))
        df = self._get_cache('stem-docid-field2df', (stem, docid, field), lambda: term_vector.stemDf(stem_idx))
        idf = math.log((self.N + 1) / (df + 0.5))
        
        length = self._get_cache('docid-field2length', (docid, field), lambda: Idx.getFieldLength(field, docid))
        avg_length = self._get_cache('field2avg_length', field, lambda: Idx.getSumOfFieldLengths(field) / Idx.getDocCount(field))
        
        k_1 = float(self.parameters['BM25:k_1'])
        b = float(self.parameters['BM25:b'])
        
        return idf * (tf) / (tf + k_1 * ((1 - b) + b * length / avg_length))
    
    def _parse_intent(self):
        
        intents_file = self.parameters["diversity:intentsPath"]
        intents_table = dict()
        num_intents = dict()
        
        with open(intents_file, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                
                id, intent = line.split(":")
                q_id, intent_id = id.split(".")
                
                if q_id not in intents_table:
                    intents_table[q_id] = dict()
                intents_table[q_id][intent_id] = intent.lower()
        
        # break each into a list of words
        for q_id in intents_table:
            for intent_id in intents_table[q_id]:
                intent = intents_table[q_id][intent_id]
                intent_list = QryParser.tokenizeString(intent)
                intent_list = list(set(intent_list))
                intents_table[q_id][intent_id] = intent_list
            num_intents[q_id] = len(intents_table[q_id])
        
        
        return intents_table, num_intents
            
            
    
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
    
    
