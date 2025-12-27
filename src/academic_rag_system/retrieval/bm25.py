# BM25 Retriever module
import numpy as np
from typing import List, Tuple
from collections import Counter
import math


class BM25:
    """
    BM25 (Best Matching 25) sparse retrieval algorithm.
    
    Standard information retrieval method that uses term frequency
    and inverse document frequency for keyword-based matching.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 with standard parameters.
        
        Args:
            k1: Term frequency saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        self.corpus = None
        self.doc_lengths = None
        self.avg_doc_length = None
        self.doc_freqs = None
        self.idf = None
        self.num_docs = 0
        
    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace + lowercase tokenization."""
        return text.lower().split()
    
    def fit(self, corpus: List[str]):
        """
        Fit BM25 on a corpus of documents.
        
        Args:
            corpus: List of document strings
        """
        self.corpus = corpus
        self.num_docs = len(corpus)
        
        # Tokenize all documents
        tokenized_corpus = [self.tokenize(doc) for doc in corpus]
        
        # Calculate document lengths
        self.doc_lengths = np.array([len(doc) for doc in tokenized_corpus])
        self.avg_doc_length = np.mean(self.doc_lengths)
        
        # Calculate document frequencies (how many docs contain each term)
        self.doc_freqs = Counter()
        for doc in tokenized_corpus:
            unique_terms = set(doc)
            for term in unique_terms:
                self.doc_freqs[term] += 1
        
        # Calculate IDF for each term
        self.idf = {}
        for term, freq in self.doc_freqs.items():
            # Standard IDF formula with smoothing
            self.idf[term] = math.log((self.num_docs - freq + 0.5) / (freq + 0.5) + 1.0)
        
        # Store tokenized corpus for scoring
        self.tokenized_corpus = tokenized_corpus
        
    def score_document(self, query_tokens: List[str], doc_idx: int) -> float:
        """
        Calculate BM25 score for a single document.
        
        Args:
            query_tokens: Tokenized query
            doc_idx: Index of document to score
            
        Returns:
            BM25 score
        """
        doc_tokens = self.tokenized_corpus[doc_idx]
        doc_length = self.doc_lengths[doc_idx]
        
        # Count term frequencies in document
        term_freqs = Counter(doc_tokens)
        
        score = 0.0
        for term in query_tokens:
            if term not in self.idf:
                continue
                
            # Term frequency in document
            tf = term_freqs.get(term, 0)
            
            # BM25 formula
            idf = self.idf[term]
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: str, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for top-k documents using BM25.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            Tuple of (scores, indices) arrays
        """
        query_tokens = self.tokenize(query)
        
        # Score all documents
        scores = np.array([
            self.score_document(query_tokens, i) 
            for i in range(self.num_docs)
        ])
        
        # Get top-k indices
        top_k_indices = np.argsort(scores)[::-1][:k]
        top_k_scores = scores[top_k_indices]
        
        return top_k_scores, top_k_indices

