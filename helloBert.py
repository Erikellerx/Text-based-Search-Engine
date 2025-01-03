"""
A simple example to illustrate the use of BERT for reranking documents.
"""

# Copyright (c) 2024, Carnegie Mellon University.  All Rights Reserved.

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from Idx import Idx

# ------------------ Configuration ------------------------- #

bert_max_sequence_length = 512				# Max WordPiece tokens
bert_modelPath = "INPUT_DIR/ms-marco-MiniLM-L-12-v2"	# Stored BERT model
indexPath = "INPUT_DIR/index-cw09"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------ Global variables ---------------------- #

bert_model = None
bert_tokenizer = None


# ------------------ Methods ------------------------------- #

def bert_prepare_tokenized_input(query, doc):
    """
    Tokenize a (query, document) pair, convert to token ids, return as tensors.
    Input: query and document strings.
    Output: a dictionary of tensors that BERT understands.
        input_ids: The ids for each token. 
        token_type_ids: The token type (sequence) id of each token.
        attention_mask: For each token, mask(0) or don't mask(1). Not used.
    """
    bert_input =  bert_tokenizer.encode_plus(
            [query, doc],		# sequence_1, sequence_2
            add_special_tokens=True,	# Add [CLS] and [SEP] tokens?
            max_length=bert_max_sequence_length,
            truncation="only_second",	# If too long, truncate sequence_2
            return_tensors="pt")	# Return PyTorch tensors
    # Move tensors to GPU if available
    bert_input = {k: v.to(device) for k, v in bert_input.items()}
    print(f"\tBERT input: {bert_input}")

    # Display WordPiece tokens. This is just FYI. It is not necessary.
    tokens = bert_tokenizer.convert_ids_to_tokens(bert_input['input_ids'][0])
    print(f"\tWordPiece tokens: {tokens}")

    return(bert_input)


def bert_score_sequence(input_dict):
    """
    Score a (query, document) pair.
    Input: the tokenized sequence.
    Output: the reranking score.
    """
    with torch.no_grad():
        # Feed the tokenized sequence to the reranker for scoring. 
        outputs = bert_model(**input_dict) 
        print(f"\tBERT Output: {outputs}")
        
        # Extract the classification score and transform to python float.
        score = outputs.logits.data.item()
        return(score)


# ------------------ Script body --------------------------- #

Idx.open(indexPath)

# Initialize BERT from a pretrained model checkpoint
bert_tokenizer = AutoTokenizer.from_pretrained(bert_modelPath)
bert_model = AutoModelForSequenceClassification.from_pretrained(
    bert_modelPath,
    num_labels=1).to(device)
bert_model.eval()

# Match a query to two documents.
query = "quit smoking"
print(f'QUERY:\t{query}\n')

doc = Idx.getAttribute("title-string", 304969)
print(f'DOC 304969:\t{doc}')
encoded_sequence = bert_prepare_tokenized_input(query, doc)
score = bert_score_sequence(encoded_sequence)
print(f'\t(q, d) score:\t{score}\n' )

doc = Idx.getAttribute("body-string", 288258)	
print(f'\nDOC 288258:\t{doc}')
encoded_sequence = bert_prepare_tokenized_input(query, doc)
score = bert_score_sequence(encoded_sequence)
print(f'\t(q, d) score:\t{score}\n' )
