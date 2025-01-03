import sys

from InvList import InvList
from QryIop import QryIop


class QryIopNear(QryIop):
    
    
    def __init__(self, dist):
        super().__init__()
        
        self.dist = dist
    
    def evaluate(self):
        
        if len(self._args) == 0:	# Should not occur if the
            return			# query optimizer did its job
        
        self.invertedList = InvList(self._field)
        
        while self.docIteratorHasMatchAll(None):
            
            docid = self._args[0].docIteratorGetMatch()
            position = []
            
            while True:
                
                i = 0
                finished = False
                while i < len(self._args) - 1:
                    
                    curr_q_i = self._args[i]
                    next_q_i = self._args[i + 1]
                    
                    if not (curr_q_i.locIteratorHasMatch() and next_q_i.locIteratorHasMatch()):
                        finished = True
                        break
                    curr_loc_i = curr_q_i.locIteratorGetMatch()
                    next_loc_i = next_q_i.locIteratorGetMatch()
                    
                    distance = next_loc_i - curr_loc_i
                    
                    if distance > self.dist:
                        curr_q_i.locIteratorAdvancePast(curr_loc_i)
                        i = max(0, i - 1)
                    elif distance <= 0:
                        next_q_i.locIteratorAdvancePast(next_loc_i)
                    else:
                        i += 1
                        if i == len(self._args) - 1:
                            position.append(next_loc_i)
                            
                            for q_i in self._args:
                                temp_loc = q_i.locIteratorGetMatch()
                                q_i.locIteratorAdvancePast(temp_loc)
                if finished:
                    break
            
            if position:
                self.invertedList.appendPosting(docid, position)
            
            for i in range(len(self._args)):
                self._args[i].docIteratorAdvancePast(docid)
                             
                
                   
            
                
        
        
            
            