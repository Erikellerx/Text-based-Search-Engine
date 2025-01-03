# Copyright (c) 2024, Carnegie Mellon University.  All Rights Reserved.

import cProfile
import json
import os
import sys

import Util

from Idx import Idx
from Ranker import Ranker
from Reranker import Reranker
from TeIn import TeIn
from Timer import Timer

# ------------------ Global variables ---------------------- #

usage = "Usage:  python QryEval.py paramFile\n\n";


# ------------------ Methods (alphabetical) ---------------- #

def main ():
    """The main function"""

    timer = Timer()
    timer.start()

    # Initialize the index and experiment parameters.
    parameters = readParameterFile()
    Idx.open(parameters['indexPath'])
    queries = Util.read_queries(parameters['queryFilePath'])
    teIn = TeIn(parameters['trecEvalOutputPath'],
                parameters['trecEvalOutputLength'])
    results = {}

    # In most search engines, the ranking pipeline would evaluate 1
    # query. The tools used in HW3-HW5 are faster and/or simpler if
    # called 1 time * n queries instead of n times * 1 query, so each
    # stage of our ranking pipeline evaluates all queries before
    # proceeding to the next stage. Rankers and rerankers return a
    # list of (qid [(score, externalId) ...]) tuples.

    # First stage ranker
    if "ranker" not in parameters:
        raise Exception('Error: Missing ranker parameters.')
    print(f'\n-- Ranker: {parameters["ranker"]["retrievalAlgorithm"]} --')
    ranker = Ranker(parameters['ranker'])
    results = ranker.get_rankings(queries)

    # Rerankers
    for reranker_name in [r for r in parameters.keys()
                          if r.startswith('reranker_') ]:
        print(f'\n-- {reranker_name}:',
              f'{parameters[reranker_name]["rerankAlgorithm"]} --\n')
        #pass bm25 k and b for diversity
        if parameters[reranker_name]["rerankAlgorithm"] == 'diversity':
            parameters[reranker_name]. update({'BM25:k_1': parameters['ranker']['BM25:k_1']})
            parameters[reranker_name]. update({'BM25:b': parameters['ranker']['BM25:b']})          
        reranker = Reranker(parameters[reranker_name])
        results = reranker.rerank(queries, results)

    # Write results to the trec_eval file
    for q in results:
        teIn.appendQuery(q, results[q], 'Erikellerx')
    
    # Clean up
    teIn.close()
    Idx.close()
    timer.stop()
    print('Time:  ' + str(timer))


def readParameterFile():
    """
    Get a dict that contains the contents of the parameter file.
    """

    # Remind the forgetful.
    if len(sys.argv) != 2:
        print(usage)
        sys.exit(1)

    if not os.path.exists(sys.argv[1]):
        print('No such file {}.'.format(sys.argv[1]))
        sys.exit(1)

    # Read the .param file into a dict. Values that look like numbers
    # are stored as numbers.
    with open(sys.argv[1]) as f:
        d = json.load(f)
    d = Util.str_to_num(d)

    return(d)


# ------------------ Script body --------------------------- #

import cProfile
import pstats

# cProfile.run('main()', 'main.profile')
# p = pstats.Stats('main.profile')
# p.strip_dirs().sort_stats('cumulative').print_stats()

main()
