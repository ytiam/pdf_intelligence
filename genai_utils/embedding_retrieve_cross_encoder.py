import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import gzip
import os
import torch
import pandas as pd
import re
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string
from tqdm.autonotebook import tqdm
import numpy as np

# if not torch.cuda.is_available():
#     print("Warning: No GPU found. Please add GPU to your notebook")

    
class sentence_transformer_paragraph_embedding():
    
    def __init__(self, data_path):
        from rank_bm25 import BM25Okapi
        self.df = pd.read_pickle(data_path)
        self.bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
        #Truncate long passages to 256 tokens
        self.bi_encoder.max_seq_length = 256
        self.top_k = 32
        
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.process()
        
    def striphtml(self, data):
        p = re.compile(r'<.*?>')
        return p.sub('', data)
    
    
    def bm25_tokenizer(self, text):
        tokenized_doc = []
        for token in text.lower().split():
            token = token.strip(string.punctuation)
            if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
                tokenized_doc.append(token)
        return tokenized_doc

    def search(self, query, best_n = 5):
        print("Input question:", query)
        ##### BM25 search (lexical search) #####
        bm25_scores = self.bm25.get_scores(self.bm25_tokenizer(query))
        top_n = np.argpartition(bm25_scores, -5)[-5:]
        bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
        bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
        ##### Semantic Search #####
        # Encode the query using the bi-encoder and find potentially relevant passages
        question_embedding = self.bi_encoder.encode(query, convert_to_tensor=True)
        question_embedding = question_embedding #.cuda()
        hits = util.semantic_search(question_embedding, self.corpus_embeddings, top_k = self.top_k)
        hits = hits[0]  # Get the hits for the first query

        ##### Re-Ranking #####
        # Now, score all retrieved passages with the cross_encoder
        cross_inp = [[query, self.passages[hit['corpus_id']]] for hit in hits]
        cross_scores = self.cross_encoder.predict(cross_inp)

        # Sort results by the cross-encoder scores
        for idx in range(len(cross_scores)):
            hits[idx]['cross-score'] = cross_scores[idx]
        hits = sorted(hits, key=lambda x: x['score'], reverse=True)
        hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
        return self.passages, hits, [self.passages[hit['corpus_id']].replace("\n", " ") for hit in hits[0:best_n]]
    
    def process(self):
        self.passages = self.df['context'].values.tolist()
        self.passages = [self.striphtml(i) for i in self.passages]
        self.corpus_embeddings = self.bi_encoder.encode(self.passages, convert_to_tensor=True, show_progress_bar=True)
        tokenized_corpus = []
        for passage in tqdm(self.passages):
            tokenized_corpus.append(self.bm25_tokenizer(passage))
        self.bm25 = BM25Okapi(tokenized_corpus)