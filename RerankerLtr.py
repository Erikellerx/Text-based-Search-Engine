"""
Access and manage a feature-based learning-to-rank (Ltr) reranker.
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

class RerankerLtr:
    """
    Access and manage a feature-based learning-to-rank (Ltr) reranker.
    """

    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, parameters):
        
        self._init_cache()

        # Store the parameters for the LTR reranker
        self.MAX_FEATURE_LEN = 20
        self.N = Idx.getNumDocs()
        self.parameters = parameters
        
        if 'ltr:featureDisable' in parameters:
            self.feature_disable = [int(x) for x in parameters['ltr:featureDisable'].split(',')]
        else:
            self.feature_disable = []
        
        

        # Initialization is a good time to create the model that will
        # be used for reranking. Cleaner code probably does this in a
        # separate function.
        # - Get training data from .trainQry and .trainQrels files
        train_qry, train_qrels = self._read_train_data(parameters['ltr:trainingQueryFile'],
                                                       parameters['ltr:trainingQrelsFile'])
        
        # - Generate feature vectors for each (qid, docid) tuple
        qid_docid = list(train_qrels.keys())
        train_features = self._generate_feature_vectors(train_qry, qid_docid)
        # - Possibly normalize vectors
        
        if self.parameters['ltr:toolkit'] == 'SVMRank':
            train_features = self._normalize4SVMrank(train_features)
        
        # - Write vectors to file
        self._write_feature_vectors(train_features, train_qrels, self.parameters['ltr:trainingFeatureVectorsFile'])
        
        # - Call the toolkit to train a model
        self.train()


    def __str__(self):
        """Human readable information about the object for debugging."""
        return(str(self.__dict__))
    
    def rerank(self, queries, results):
        """
        Update the results for a set of queries with new scores.

        queries: A dict of {qid : qString} pairs.
        results: A dict of {qid : [(score, externalId) ...]} tuples.
        """

        # Generate feature vectors for each (qid, docid) tuple
        qid_docid = []
        for qid in queries:
            for _, docid in results[qid]:
                qid_docid.append((int(qid), docid))
        queries = {int(qid):queries[qid] for qid in queries}
        test_features = self._generate_feature_vectors(queries, qid_docid)
        # Possibly normalize vectors
        if self.parameters['ltr:toolkit'] == 'SVMRank':
            test_features = self._normalize4SVMrank(test_features)

        # Write vectors to file
        self._write_feature_vectors(test_features, {t:0 for t in test_features}, self.parameters['ltr:testingFeatureVectorsFile'])

        # Call the toolkit to generate new scores
        self.inference()

        # Use the new scores to update results
        results = self._process_result(test_features)

        return {str(qid):results[int(qid)] for qid in results}
    
    def _generate_feature_vectors(self, train_qry, train_qrels):
        
        train_features = dict()
        for qid, docid in train_qrels:
            internal_doc_id = self._get_cache('docid2internal', docid, lambda: Idx.getInternalDocid(docid))
            external_doc_id = self._get_cache('internal2docid', internal_doc_id, lambda: Idx.getExternalDocid(internal_doc_id))
            
            feature = [None] * self.MAX_FEATURE_LEN
            
            # f1: Spam score for d (read from index).
            if 1 not in self.feature_disable:
                feature[0] = self._get_cache('id2spam', internal_doc_id, lambda: float(Idx.getAttribute('spamScore', internal_doc_id)))
            
            url = self._get_cache('id2url', internal_doc_id, lambda: Idx.getAttribute('rawUrl', internal_doc_id))
            # f2: Url depth for d (number of '/' in the rawUrl field).
            if 2 not in self.feature_disable:
                feature[1] = float(url.count('/'))
                
            # f3: FromWikipedia score for d (1 if the rawUrl contains "wikipedia.org", otherwise 0).
            if 3 not in self.feature_disable:
                feature[2] = 1.0 if 'wikipedia.org' in url else 0.0
            
            # f4: PageRank score for d (read from index).
            if 4 not in self.feature_disable:
                feature[3] = self._get_cache('id2pagerank', internal_doc_id, lambda: float(Idx.getAttribute('PageRank', internal_doc_id)))
            
            queryStems = QryParser.tokenizeString(train_qry[qid])
            
            for i, field in enumerate(['body', 'title', 'url', 'inlink']):
                
                # f5, f8, f11, f14: BM25 score for <q, d_field>.
                if 5 + 3 * i not in self.feature_disable:
                    feature[4 + 3 * i] = self.feature_bm25(internal_doc_id, queryStems, field)
                    
                # f6, f9, f12, f15: Query likelihood score for <q, d_field>.
                if 6 + 3 * i not in self.feature_disable:
                    feature[5 + 3 * i] = self.feature_query_likelihood(internal_doc_id, queryStems, field)
                
                # f7, f10, f13, f16: Term overlap score (also called Coordinate Match) for <q, d_field>.
                if 7 + 3 * i not in self.feature_disable:
                    feature[6 + 3 * i] = self.feature_coordinate_match(internal_doc_id, queryStems, field)
                
                # f17 inlink count
                if 17 not in self.feature_disable:
                    feature[16] = Idx.getFieldLength('inlink', internal_doc_id)
                    
                
                # f18 
                if 18 not in self.feature_disable:
                    feature[17] = self.feature_tf(internal_doc_id, queryStems, 'body', 'variance', idf=True)
                
                # f19
                if 19 not in self.feature_disable:
                     feature[18] = self._get_cache('docid-field2length', (internal_doc_id, 'body'), lambda: Idx.getFieldLength('body', internal_doc_id))
                
                # f20
                if 20 not in self.feature_disable:
                    feature[19] = self.feature_IncItc(internal_doc_id, queryStems, 'keywords')
                    
            
            feature = [str(x) if x is not None else None for x in feature]
            train_features[(qid, docid)] = feature
        
        return train_features
    
    
    def feature_queryTermDensity(self, docid, queryStems, field):
        
        term_vector = self._get_cache('term_vector', (docid, field), lambda: Idx.getTermVector(docid, field))
        if term_vector.stemsLength() == 0:
            return None
        
        docLength = self._get_cache('docid-field2length', (docid, field), lambda: Idx.getFieldLength(field, docid))
        score = 0.0
        for stem in queryStems:
            if stem in term_vector.stems:
                stem_idx = self._get_cache('docid-field-stem2idx', (docid, field, stem), lambda: term_vector.indexOfStem(stem))
                tf = self._get_cache('docid-field-stem2tf', (docid, field, stem), lambda: term_vector.stemFreq(stem_idx))
                score += tf 
        
        return score / docLength
    
    def feature_tf(self, docid, queryStems, field, type, idf = False):
        
        term_vector = self._get_cache('term_vector', (docid, field), lambda: Idx.getTermVector(docid, field))
        if term_vector.stemsLength() == 0:
            return None
    
        tf_list = []
        
        for stem in queryStems:
            if stem in term_vector.stems:
                stem_idx = self._get_cache('docid-field-stem2idx', (docid, field, stem), lambda: term_vector.indexOfStem(stem))
                if idf:
                    df = self._get_cache('stem-docid-field2df', (stem, docid, field), lambda: term_vector.stemDf(stem_idx))
                    idf = math.log(self.N / df)
                else:
                    idf = 1.0
                tf_list.append(self._get_cache('docid-field-stem2tf', (docid, field, stem), lambda: term_vector.stemFreq(stem_idx)) * idf)
        
        if len(tf_list) == 0:
            return None
        
        tf_list = np.array(tf_list)
        
        output = dict()
        output['min'] = min(tf_list)
        output['max'] = max(tf_list)
        output['sum'] = np.sum(tf_list)
        output['mean'] = np.mean(tf_list)
        output['variance'] = np.var(tf_list)
      
        return output[type]
    
    def feature_IncItc(self, docid, queryStems, field):
        
        term_vector = term_vector = self._get_cache('term_vector', (docid, field), lambda: Idx.getTermVector(docid, field))
        if term_vector.stemsLength() == 0:
            return None
        
        score = 0.0
        for stem in queryStems:
            if stem in term_vector.stems:
                stem_idx = self._get_cache('docid-field-stem2idx', (docid, field, stem), lambda: term_vector.indexOfStem(stem))
                tf = self._get_cache('docid-field-stem2tf', (docid, field, stem), lambda: term_vector.stemFreq(stem_idx))
                df = self._get_cache('stem-docid-field2df', (stem, docid, field), lambda: term_vector.stemDf(stem_idx))
                qtf = queryStems.count(stem)
                score += ((math.log(tf) + 1) * ((math.log(qtf) + 1) / math.log(self.N / df))) / \
                            (math.sqrt((math.log(tf) + 1) ** 2) * (math.sqrt((math.log(qtf) + 1) * math.log(self.N / df) ** 2)) )
            
        
        return score

    def feature_bm25(self, docid, queryStems, field):
        
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
        
        k_1 = float(self.parameters['ltr:BM25:k_1'])
        b = float(self.parameters['ltr:BM25:b'])
        
        return idf * (tf) / (tf + k_1 * ((1 - b) + b * length / avg_length))
        
    
    def feature_query_likelihood(self, docid, queryStems, field):
        
        term_vector = self._get_cache('term_vector', (docid, field), lambda: Idx.getTermVector(docid, field))
        if term_vector.stemsLength() == 0:
            return None
        
        score = 1
        missed = 0
        for stem in queryStems:
            
            ctf_q = self._get_cache('field-stem2ctf', (field, stem), lambda: Idx.getTotalTermFreq(field, stem))
            length_c = self._get_cache('field2length', field, lambda: Idx.getSumOfFieldLengths(field))
            p_mle_qiC = ctf_q / length_c
            
            if stem not in term_vector.stems:
                missed += 1
                tf_qd = 0
            else:
                stem_idx = self._get_cache('docid-field-stem2idx', (docid, field, stem), lambda: term_vector.indexOfStem(stem))
                tf_qd = self._get_cache('docid-field-stem2tf', (docid, field, stem), lambda: term_vector.stemFreq(stem_idx))
            lengh_d = self._get_cache('docid-field2length', (docid, field), lambda: Idx.getFieldLength(field, docid))
            
            mu = float(self.parameters['ltr:QL:mu'])
            p_qd = (tf_qd + mu * p_mle_qiC) / (lengh_d + mu)
            score *= p_qd
            
        if missed == len(queryStems):
            return 0.0
        return math.pow(score, 1 / len(queryStems))
    
    
    def feature_coordinate_match(self, docid, queryStems, field):
        
        term_vector = self._get_cache('term_vector', (docid, field), lambda: Idx.getTermVector(docid, field))
        if term_vector.stemsLength() == 0:
            return None
        
        ret = 0
        for stem in queryStems:
            if term_vector.stems and stem in term_vector.stems:
                ret += 1
        
        return float(ret)
    
    
    def train(self):
        
        if 'ltr:RankLib:metric2t' not in self.parameters:
            self.parameters['ltr:RankLib:metric2t'] = 'MAP'
        
        if self.parameters['ltr:toolkit'] == 'RankLib':
            # ListNet
            if self.parameters['ltr:RankLib:model'] == 7:
                PyLu.RankLib.main([
                '-train',   f"{self.parameters['ltr:trainingFeatureVectorsFile']}",
                '-ranker',  f"{self.parameters['ltr:RankLib:model']}",
                '-save',    f"{self.parameters['ltr:modelFile']}"])
            
            #Coordinate Ascent
            elif self.parameters['ltr:RankLib:model'] == 4:
                PyLu.RankLib.main([
                '-train',   f"{self.parameters['ltr:trainingFeatureVectorsFile']}",
                '-ranker',  f"{self.parameters['ltr:RankLib:model']}",
                '-save',    f"{self.parameters['ltr:modelFile']}",
                '-metric2t', f"{self.parameters['ltr:RankLib:metric2t']}"])
            else:
                raise ValueError(f"Invalid model: {self.parameters['ltr:RankLib:model']}")
        
        elif self.parameters['ltr:toolkit'] == 'SVMRank':
            self.parameters['ltr:svmRankLearnPath'] = self.parameters['ltr:svmRankLearnPath'].replace('/', os.sep)
            try:
                output = subprocess.check_output(
                    f"{self.parameters['ltr:svmRankLearnPath']} -c {self.parameters['ltr:svmRankParamC']} {self.parameters['ltr:trainingFeatureVectorsFile']} {self.parameters['ltr:modelFile']}",
                    stderr=subprocess.STDOUT,
                    shell=True).decode('UTF-8')
            except subprocess.CalledProcessError as e:
                print(e.output)
            
        else:
            raise ValueError(f"Invalid toolkit: {self.parameters['ltr:toolkit']}")
            
            
    def inference(self):
        
        if self.parameters['ltr:toolkit'] == 'RankLib':
            PyLu.RankLib.main([
                '-rank',    f"{self.parameters['ltr:testingFeatureVectorsFile']}",
                '-load',    f"{self.parameters['ltr:modelFile']}",
                '-score',   f"{self.parameters['ltr:testingDocumentScores']}"])
        
        elif self.parameters['ltr:toolkit'] == 'SVMRank':
            self.parameters['ltr:svmRankClassifyPath'] = self.parameters['ltr:svmRankClassifyPath'].replace('/', os.sep)
            try:
                output = subprocess.check_output(
                    f"{self.parameters['ltr:svmRankClassifyPath']} {self.parameters['ltr:testingFeatureVectorsFile']} {self.parameters['ltr:modelFile']} {self.parameters['ltr:testingDocumentScores']}",
                    stderr=subprocess.STDOUT,
                    shell=True).decode('UTF-8')
            except subprocess.CalledProcessError as e:
                print(e.output)
        
    def _read_train_data(self, train_qry_file, train_qrels_file):
        """
        Read training data from .trainQry and .trainQrels files.

        train_qry_file: Path to the .trainQry file.
        train_qrels_file: Path to the .trainQrels file.
        """

        # Read the training queries
        with open(train_qry_file, 'r') as f:
            train_qry = dict()
            while True:
                line = f.readline()
                if not line:
                    break
                qid, qry = line.strip().split(':')
                train_qry[int(qid)] = qry

        # Read the training qrels
        with open(train_qrels_file, 'r') as f:
            train_qrels = dict()
            while True:
                line = f.readline()
                if not line:
                    break
                qid, _, docid, rel = line.strip().split(' ')
                qid, rel = int(qid), int(rel)
                if rel == -2:
                    rel = 0
                train_qrels[(qid, docid)] = rel

        return(train_qry, train_qrels)

      
    def _write_feature_vectors(self, features, train_qrels, filepath):
        
        sorted_keys = sorted(features.keys())
        with open(filepath, 'w') as f:
            for qid, docid in sorted_keys:
                temp = []
                temp.append(str(train_qrels[(qid, docid)]))
                temp.append(f'qid:{qid}')
                
                assert len(features[(qid, docid)]) <= self.MAX_FEATURE_LEN
                for i in range(self.MAX_FEATURE_LEN):
                    features[(qid, docid)][i] = features[(qid, docid)][i] if features[(qid, docid)][i] is not None else 0.0
                    if i + 1 not in self.feature_disable:
                        temp.append(f'{i + 1}:{features[(qid, docid)][i]}')
                
                temp.append(f'# {docid}')
                f.write(' '.join(temp) + '\n')
                
                

    def _process_result(self, features):
        
        feature_keys = sorted(features.keys())
        with open(self.parameters['ltr:testingDocumentScores'], 'r') as f:
            result = dict()
            i = 0
            while True:
                line = f.readline()
                if not line:
                    break
                if self.parameters['ltr:toolkit'] == 'RankLib':
                    _ , _, score = line.strip().split('\t')
                    score = float(score)
                elif self.parameters['ltr:toolkit'] == 'SVMRank':
                    score = float(line.strip())
                else:
                    raise ValueError(f"Invalid toolkit: {self.parameters['ltr:toolkit']}")
                    
                qid, docid = feature_keys[i]
                if qid not in result:
                    result[qid] = []
                result[qid].append((score, docid))
                i += 1
                
        #sort based on the score
        for qid in result:
            result[qid] = sorted(result[qid], key=lambda x: x[0], reverse=True)
        return result
                    
    def _normalize4SVMrank(self, features):
        

        max_val_list = dict()
        min_val_list = dict()
        for qid, docid in features:
            feature = features[(qid, docid)]
            if qid not in max_val_list:
                max_val_list[qid] = [-1e9] * self.MAX_FEATURE_LEN
                min_val_list[qid] = [1e9] * self.MAX_FEATURE_LEN
            for i in range(len(feature)):
                if not feature[i]:
                    continue
                if i + 1 not in self.feature_disable: 
                    max_val_list[qid][i] = max(max_val_list[qid][i], float(feature[i]))
                    min_val_list[qid][i] = min(min_val_list[qid][i], float(feature[i]))
            
        for qid, docid in features:
            feature = features[(qid, docid)]
            for i in range(len(feature)):
                if not feature[i]:
                    continue
                if i + 1 not in self.feature_disable:
                    max_val = max_val_list[qid][i]
                    min_val = min_val_list[qid][i]
                    if max_val != min_val:
                        feature[i] = (float(feature[i]) - min_val) / (max_val - min_val)
                    else:
                        feature[i] = 0.0
                    
            
            features[(qid, docid)] = [str(x) if x is not None else None for x in feature]
        
        return features
            

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
        
        
    