"""
Create, access, and manipulate document score lists.
"""

# Copyright (c) 2024, Carnegie Mellon University.  All Rights Reserved.


import heapq

from Idx import Idx

class ResultHeap:
    """
    A ResultHeap stores the top n search results. The underlying
    data structure is a Python heap with a maximum size.
    """

    class Result:
        """
        A utility class to create an <externalDocid, score> object
        for the heap.
        """

        def __init__(self, externalId, score):
            self.externalId = externalId
            self.score = score

        def __lt__(self, other):
            return(self.score < other.score or
                    (self.score == other.score and
                      self.externalId < other.externalId))

        def __str__(self):
            return(f'{self.externalId} {self.score}')


    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, n):
        """Create an empty ResultHeap that can store n results."""
        self._max_heap_size = n
        self._heap = []


    def _full(self):
        """
        True iff the heap is full, otherwise False.
        """
        return(len(self._heap) == self._max_heap_size)
    
        
    def __len__(self):
        return(len(self._heap))


    def add(self, internalId, score):
        """
        Add a document to the ResultHeap.
        internalId: An internal document id.
        score: The document score.
        """

        # Do the most common case as efficiently as possible. This
        # works because the list is always kept in heap order.
        if self._full():
            if self._heap[ 0 ].score > score:
                return

        # Check whether this result needs to be added to the list.
        externalId = Idx.getExternalDocid(internalId)
        
        if not self._full():
            heapq.heappush(self._heap, self.Result(externalId, score))
            heapq.heapify(self._heap)
        else:
            # Decide whether to replace the smallest element. This can
            # be done with fewer lines of code, but it is called hundreds
            # of thousands of times, so be efficient as possible.
            smallest = self._heap[ 0 ]
            if (smallest.score < score or
                 (smallest.score == score  and
                   smallest.externalId < externalId)):
                smallest.score = score
                smallest.externalId = externalId
                heapq.heapify(self._heap)


    def get_ranking(self):
        """
        Get a ranked list of results in (score, externalId) order.
        """
        return(sorted(self._heap,
                        key=lambda x: (-x.score, x.externalId)))

