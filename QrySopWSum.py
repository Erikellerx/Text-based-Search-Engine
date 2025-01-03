import sys

from QrySop import QrySop
from RetrievalModelUnrankedBoolean import RetrievalModelUnrankedBoolean
from RetrievalModelRankedBoolean import RetrievalModelRankedBoolean
from RetrievalModelBM25 import RetrievalModelBM25


class QrySopWSum(QrySop):
    
    def __init__(self):
        QrySop.__init__(self)
        self.weight = None
        
    
    def docIteratorHasMatch(self, r):
        
        return self.docIteratorHasMatchMin(r)
    
    def getScore(self, retrievalModel):
        """
        Get a score for the document that docIteratorHasMatch matched.

        retrievalModel: retrieval model parameters

        Returns the document score.

        throws IOException: Error accessing the Lucene index
        """

        if isinstance(retrievalModel, RetrievalModelBM25):
            return self.__getScoreBM25(retrievalModel)
        else:
            raise Exception('{}.{} does not support {}'.format(
                self.__class__.__name__,
                sys._getframe().f_code.co_name,
                retrievalModel.__class__.__name__))
    
    def __getScoreBM25(self, r):
        """
        getScore for the BM25 retrieval model.

        r: The retrieval model that determines how scores are calculated.
        Returns the document score.
        throws IOException: Error accessing the Lucene index
        """
        if not self.weights:
            raise Exception('Weights not set')
        
        docid = self.docIteratorGetMatch()
        score = 0.0
        i = 0
        for q_i in self._args:
            if (q_i.docIteratorHasMatch(r) and
            q_i.docIteratorGetMatch() == docid):
                weight = self.weights[i] / sum(self.weights)
                score += q_i.getScore(r) * weight
            i += 1
        return score