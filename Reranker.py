"""
Rerank initial rankings for a set of queries. The rankings may
come from an .inRank file or from a bag-of-words ranker (ranked
and unranked boolean, Indri, BM25).
"""

from RerankerPrf import RerankerPrf
from RerankerLtr import RerankerLtr
from RerankerBERTRR import RerankerBERTRR
from RerankerDiversity import RerankerDiversity


# Copyright (c) 2024, Carnegie Mellon University.  All Rights Reserved.

class Reranker:
    """
    Rerank initial rankings for a set of queries. The rankings may
    come from an .inRank file or from a bag-of-words ranker (ranked
    and unranked boolean, Indri, BM25).
    """

    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, parameters):
        self._model = None
        self._rerank_depth = parameters.get('rerankDepth', 1000)

        if 'rerankAlgorithm' not in parameters:
            raise Exception('Error: Missing parameter rerankAlgorithm.')
        
        models = {
            'prf': RerankerPrf,
            'ltr': RerankerLtr,
            'bertrr': RerankerBERTRR,
            'diversity': RerankerDiversity
        }
        if parameters['rerankAlgorithm'].lower() not in models:
            raise Exception('Error: Unknown rerankAlgorithm: ' \
                            f'{parameters["rerankAlgorithm"].lower()}')
        self._model = models[parameters['rerankAlgorithm'].lower()](parameters)
        
        self.parameters = parameters


    def rerank(self, queries, results):
        """
        Rerank a list of rankings for a set of queries. Each ranking is
        a list of (score, externalId) tuples.

        queries: A dict of {query_id: query_string}.
        results: A dict of {query_id: [(score, externalId)]}.
        """
        top_rankings = {qid:results[qid][:self._rerank_depth]
                        for qid in results}
        bot_rankings = {qid:results[qid][self._rerank_depth:]
                        for qid in results}
        top_rankings = self._model.rerank(queries, top_rankings)
        
        #fix for prf reranker problem by prof
        #if prf reranking, just return the top rankings
        if isinstance(self._model, RerankerPrf):
            return(top_rankings)
        
        #Merge the reranking (top) with the bottom of the
        # original ranking.
        for qid in results:
            q_min_score = top_rankings[qid][-1][0]
            bot_rankings[qid] = [(q_min_score-(1+i), bot_rankings[qid][i][1])
                                 for i in range(len(bot_rankings[qid]))] 
            
        results = {qid:top_rankings[qid] + bot_rankings[qid]
                    for qid in results}
        return(results)
