from RetrievalModel import RetrievalModel

class RetrievalModelBM25(RetrievalModel):
    
    def __init__(self, parameters):
        super().__init__()
        self.k_1 = parameters['BM25:k_1']
        self.b = parameters['BM25:b']
        self.k_3 = 0
        self.qtf = 1
        self.defaultQrySop = '#SUM'
        