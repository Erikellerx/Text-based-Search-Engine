"""
Get an initial ranking for a set of queries. The ranking may come
from an .inRank file or from a bag-of-words ranker (ranked and
unranked boolean, Indri, BM25).
"""

# Copyright (c) 2024, Carnegie Mellon University.  All Rights Reserved.

import heapq
import itertools

from collections import OrderedDict

import Util

from Idx import Idx
from QryParser import QryParser
from RetrievalModelUnrankedBoolean import RetrievalModelUnrankedBoolean
from RetrievalModelRankedBoolean import RetrievalModelRankedBoolean
from RetrievalModelBM25 import RetrievalModelBM25

class Ranker:
    """
    Get an initial ranking for a set of queries. The ranking may
    come from an .inRank file or from a bag-of-words ranker (ranked
    and unranked boolean, Indri, BM25).

    """

    class heap_item:
        """
        A utility class to provide control over heap __lt__.
        """

        def __init__(self, score, externalId):
            self.externalId = externalId
            self.score = score

        def __lt__(self, other):
            return(self.score < other.score or
                   (self.score == other.score and
                    self.externalId > other.externalId))



    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, parameters):
        self._model = None
        self._inRank_path = None
        self._max_results = 1000       		# default

        if 'outputLength' in parameters:
            self._max_results = parameters['outputLength']

        if 'retrievalAlgorithm' not in parameters:
            raise Exception('Error: Missing parameter retrievalAlgorithm.')

        if parameters['retrievalAlgorithm'] == 'UnrankedBoolean':
            self._model = RetrievalModelUnrankedBoolean(parameters)
        elif parameters['retrievalAlgorithm'] == 'RankedBoolean':
            self._model = RetrievalModelRankedBoolean(parameters)
        elif parameters['retrievalAlgorithm'] == 'BM25':
            self._model = RetrievalModelBM25(parameters)
        elif parameters['retrievalAlgorithm'] == 'inRankFile':
            self._inRank_path = parameters['inRankFile:Path']
        else:
            raise Exception('Error: Unknown retrievalAlgorithm: ' \
                            f'{parameters["retrievalAlgorithm"]}')


    def get_rankings(self, queries):
        """
        Get a list of rankings for a set of queries. Each ranking is
        a list of (score, externalId) tuples.

        queries: A dict of {query_id:query_string}.
        """
        if self._model is not None:
            return(self.get_rankings_bow(queries))
        elif self._inRank_path is not None:
            return(Util.read_rankings(self._inRank_path))
        else:
            raise Exception('Error: Ranker does not know how to rank')
                        

    def get_rankings_bow(self, queries):
        """
        Get a list of rankings for a set of queries. Each ranking is
        a list of (score, externalId) tuples.
        
        queries: A dict of {query_id:query_string}.
        """
        results = {}
        
        for qid, qString in queries.items():
            # Prepare to evaluate a query
            print(f'{qid}: {qString}')
            qString = f'{self._model.defaultQrySop}({qString})'
            q = QryParser.getQuery(qString)
            print(f'    ==> {str(q)}')
            q.initialize(self._model)
            result_heap = []		# A heap of max size n

            # Evaluate the query. Each pass of the loop finds
            # one matching document.
            while(q.docIteratorHasMatch(self._model)):
                docid = q.docIteratorGetMatch()
                score = q.getScore(self._model)
                q.docIteratorAdvancePast(docid)

                # Python heaps keep the smallest element at [0].
                # The most common case is that (score, docid) is not
                # in the top n. Do it first and efficiently.
                if len(result_heap) == self._max_results:
                    if result_heap[0].score > score:
                        continue

                # Maybe this (score, docid) needs to be saved.
                externalId = Idx.getExternalDocid(docid)

                if len(result_heap) < self._max_results:
                    heapq.heappush(result_heap,
                                   self.heap_item(score, externalId))
                    heapq.heapify(result_heap)
                else:
                    smallest = result_heap[0]
                    if (smallest.score < score or
                        (smallest.score == score  and
                         smallest.externalId > externalId)):
                        heapq.heapreplace(result_heap,
                                          self.heap_item(score, externalId))

            # Convert the heap into a list of (score, externalId),
            # then sort into ranking order and save it.
            results_qid = [(r.score, r.externalId) for r in result_heap]
            results_qid.sort(key=lambda r: (-r[0], r[1]))
            results[qid] = results_qid

        return(results)
