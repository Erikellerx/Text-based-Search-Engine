import math
from Idx import Idx
from Ranker import Ranker



class RerankerPrf:
    
    def __init__(self,parameters):
        
        # prf parameters
        self.num_docs = parameters["prf:numDocs"]
        self.num_terms = parameters["prf:numTerms"]
        self.prfAlgorithm = parameters["prf:algorithm"]
        self.expansionField = parameters["prf:expansionField"] if "prf:expansionField" in parameters else "body"
        self.expansion_query_file = parameters["prf:expansionQueryFile"]
        self.defaultQrySop = '#SUM'
        
        # ranker parameters
        self.parameters = parameters
        
        # cache dicts
        self.df_dict = {}
        self.id_dict = {}
        self.term_vector_dict = {}
        self.N = Idx.getNumDocs()
        
    
    def rerank(self, queries, top_rankings):
        '''
        Rerank the top n (self.numTerms) docs for each query using the pseudo relevance feedback
        This method will generate the expanded query with length m (self.numTerms) and rerank the top n docs
        Expanded query will be written to the expansion query file (.qryOut)
        Expanded query: #SUM( t1 t2 ... tm ), where t1, t2, ... tm are the top m terms with highest weight
        
        Note: This prf method currently only support the Okapi BM25 retrieval model
        
        queries: dict, {qid: query}
        top_rankings: dict, {qid: [(score, external doc id)]}
        
        return reranked_result: dict, {qid: [(score, external doc id)]}
        '''
        self.qry_out = open(self.expansion_query_file, 'w')
        
        for qid in top_rankings:
            
            # Get the top n docs for each query and initialize the term weight dict
            top_n_docs = top_rankings[qid][:self.num_docs]
            term_weight = {}
            
            for doc in top_n_docs: 
                
                # Get the internal doc id and term vector of the doc
                internal_doc_id = self.__get_internel_doc_id(doc[1])     
                term_vector = self.__get_term_vector(internal_doc_id, self.expansionField)
                if term_vector.stemsLength() == 0:
                    #import pdb; pdb.set_trace()
                    continue
                
                for i, term in enumerate(term_vector.stems):
                    
                    # filter out non-ascii characters, terms with '.' or ','
                    if  term is None or '.' in term or ',' in term or \
                        not term.isascii():
                        continue  
                    
                    # Update the rdf of each term
                    # rdf = # of relevant docs containing term
                    if term not in term_weight:
                        term_weight[term] = 0
                    term_weight[term] += 1
                    self.__store_df(i, term, term_vector)

            # Calculate the weight of each term 
            for term in term_weight:
                if term_weight[term] == 0:
                    continue
                df = self.__get_df(term)
                term_weight[term] *= math.log((self.N - df + 0.5) / (df + 0.5))
            
            # Sort the terms by weight, if the weight is the same, sort by term on alphabetic order
            sorted_terms = sorted(term_weight.items(), key=lambda x: (x[1], x[0]))
            sorted_terms = reversed([sorted_terms[-i - 1][0] for i in range(min(self.num_terms, len(sorted_terms)))])
            
            # Generate the expanded query
            if self.expansionField == 'body':
                expanded_query = self.defaultQrySop + \
                                '( ' + ' '.join(sorted_terms) + ' )'
            else:
                expanded_query = self.defaultQrySop + \
                                '( ' + ' '.join([q + f'.{self.expansionField}' for q in sorted_terms]) + ' )'
                                
            self.qry_out.write(qid + ': ' + expanded_query + '\n')
            queries[qid] = expanded_query
        self.qry_out.close()
        
        # Rerank the expanded queries
        self.parameters['outputLength'] = self.parameters['rerankDepth']
        new_ranker = Ranker(self.parameters)
        reranked_result = new_ranker.get_rankings(queries)
        
        return reranked_result    
        
    def __get_internel_doc_id(self, external_doc_id):
        '''
        Get the internal doc id from external doc id
        if the external doc id is already in the cache,
        return the internal doc id from the cache
        
        external_doc_id: str, external doc id
        return: int, internal doc id
        
        '''
        if external_doc_id in self.id_dict:
            return self.id_dict[external_doc_id]
        else:
            internal_doc_id = Idx.getInternalDocid(external_doc_id)
            self.id_dict[external_doc_id] = internal_doc_id
            return internal_doc_id
        
    def __get_term_vector(self, internal_doc_id, field):
        '''
        Get the term vector from the cache
        if the term vector is not in the cache,
        get the term vector from the index and store it in the cache
        
        internal_doc_id: int, internal doc id
        field: str, field name
        return: TermVector
        '''
        if (internal_doc_id, field) in self.term_vector_dict:
            return self.term_vector_dict[(internal_doc_id, field)]
        else:
            term_vector = Idx.getTermVector(internal_doc_id, field)
            self.term_vector_dict[(internal_doc_id, field)] = term_vector
            return term_vector
    
    def __store_df(self, i, term, term_vector):
        '''
        Store the df of the term in the cache
        
        i: int, i th term in the term vector
        term: str, term
        term_vector: TermVector
        '''
        if term in self.df_dict:
            return
        df = term_vector.stemDf(i)
        self.df_dict[term] = df
 
    def __get_df(self, term):
        '''
        Get the df of the term from the cache
        
        term: str, term
        return: int, df of the term
        '''
        return self.df_dict[term]
            
            
            
            
            
            
            
        
        
                    
                

                    
            