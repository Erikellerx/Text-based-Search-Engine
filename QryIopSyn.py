"""The SYN operator for all retrieval models."""

# Copyright (c) 2024, Carnegie Mellon University.  All Rights Reserved.

from InvList import InvList
from QryIop import QryIop

class QryIopSyn(QryIop):
    """The SYN operator for all retrieval models."""

    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self):
        """Create an empty SYN query node."""
        QryIop.__init__(self)		# Inherit from QryIop


    def evaluate(self):
        """"
        Evaluate the query operator; the result is an internal inverted
        list that may be accessed via the internal iterators.

        throws IOException: Error accessing the Lucene index.
        """

        # Create an empty inverted list.
        self.invertedList = InvList(self._field)

        if len(self._args) == 0:	# Should not occur if the
            return			# query optimizer did its job

        # Each pass of the loop adds 1 document to result inverted list
        # until all of the argument inverted lists are depleted.
        while True:

            # Locations will be merged for the minimum next document id.
            min_iDocid = None

            for q_i in self._args:
                if q_i.docIteratorHasMatch(None):
                    q_iDocid = q_i.docIteratorGetMatch()
                    if min_iDocid == None or min_iDocid > q_iDocid:
                        min_iDocid = q_iDocid;

            if min_iDocid == None:
                break			# All docids were processed. Done.
      
            # Create and save a new posting that is the union of the
            # posting lists for min_iDocid.  Locations that appear in
            # multiple arguments (e.g., #SYN(cat cat dog)) are fine.
            positions = []
            for q_i in self._args:
                if (q_i.docIteratorHasMatch(None) and
                    q_i.docIteratorGetMatch() == min_iDocid):
                    positions += q_i.docIteratorGetMatchPosting().positions
                    q_i.docIteratorAdvancePast(min_iDocid)

            positions = sorted(set(positions))	# sorted & unique
            self.invertedList.appendPosting (min_iDocid, positions)
