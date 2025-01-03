"""
The SCORE operator for all retrieval models.
"""

# Copyright (c) 2024, Carnegie Mellon University.  All Rights Reserved.

import math
import sys

from Idx import Idx
from QrySop import QrySop
from RetrievalModelUnrankedBoolean import RetrievalModelUnrankedBoolean
from RetrievalModelRankedBoolean import RetrievalModelRankedBoolean
from RetrievalModelBM25 import RetrievalModelBM25


class QrySopScore(QrySop):
    """
    """

    # -------------- Methods (alphabetical) ---------------- #


    def __init__(self):
        QrySop.__init__(self)		# Inherit from QrySop

        self.N = Idx.getNumDocs()
        self.SumOfFieldLengths = dict()
        self.DocCount = dict()
        

    def docIteratorHasMatch(self, r):
        """
        Indicates whether the query has a match.
        r: The retrieval model that determines what is a match.

        Returns True if the query matches, otherwise False.
        """
        return(self.docIteratorHasMatchFirst(r))


    def getScore(self, r):
        """
        Get a score for the document that docIteratorHasMatch matched.
        
        r: The retrieval model that determines how scores are calculated.
        Returns the document score.
        throws IOException: Error accessing the Lucene index.
        """

        if isinstance(r, RetrievalModelUnrankedBoolean):
            return self.__getScoreUnrankedBoolean(r)
        elif isinstance(r, RetrievalModelRankedBoolean):
            return self.__getScoreRankedBoolean(r)
        elif isinstance(r, RetrievalModelBM25):
            return self.__getScoreBM25(r)
        else:
            raise Exception(
                '{} does not support the #SCORE operator.'.format(
                    r.__class__.__name__))


    def __getScoreUnrankedBoolean(self, r):
        """
        getScore for the Unranked retrieval model.

        r: The retrieval model that determines how scores are calculated.
        Returns the document score.
        throws IOException: Error accessing the Lucene index.
        """
        if not self.docIteratorHasMatchCache():
            return 0.0
        else:
            return 1.0
    
    def __getScoreRankedBoolean(self, r):
        """
        getScore for the Ranked retrieval model.

        r: The retrieval model that determines how scores are calculated.
        Returns the document score.
        throws IOException: Error accessing the Lucene index.
        """
        if not self.docIteratorHasMatchCache():
            return 0.0
        else:
            q = self._args[0]
            return q.docIteratorGetMatchPosting().tf
        
    
    def __getScoreBM25(self, r):
        """
        getScore for the BM25 retrieval model.

        r: The retrieval model that determines how scores are calculated.
        Returns the document score.
        throws IOException: Error accessing the Lucene index.
        """
        if not self.docIteratorHasMatchCache():
            return 0.0
        else:
            q = self._args[0]
            tf = q.docIteratorGetMatchPosting().tf
            df = q.getDf()
            idf = math.log((self.N + 1) / (df + 0.5))
            avg_doclen = self.__getSumOfFieldLengths(q._field) / self.__getDocCount(q._field)
            tfWeight = tf / (tf + r.k_1 * ((1 - r.b) + r.b * Idx.getFieldLength(q._field, q.docIteratorGetMatch()) / avg_doclen))
            userWeight = (r.k_3 + 1) * r.qtf / (r.k_3 + r.qtf)
            return idf * tfWeight * userWeight

    def __getSumOfFieldLengths(self, field):
        if field not in self.SumOfFieldLengths:
            ret =  Idx.getSumOfFieldLengths(field)
            self.SumOfFieldLengths[field] = ret
            return ret
        else:
            return self.SumOfFieldLengths[field]
    
    def __getDocCount(self, field):
        if field not in self.DocCount:
            ret = Idx.getDocCount(field)
            self.DocCount[field] = ret
            return ret
        else:
            return self.DocCount[field]


    def initialize(self, r):
        """
        Initialize the query operator (and its arguments), including any
        internal iterators.  If the query operator is of type QryIop, it
        is fully evaluated, and the results are stored in an internal
        inverted list that may be accessed via the internal iterator.

        r: A retrieval model that guides initialization.
        throws IOException: Error accessing the Lucene index.
        """
        q = self._args[ 0 ]
        q.initialize(r)
