# Hybrid Retriever module
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
from collections import Counter
import math
from academic_rag_system.retrieval.bm25 import BM25 


class HybridRetriever:
    """
    Hybrid retrieval system combining dense (embedding) and sparse (BM25) retrieval.
    
    Uses Reciprocal Rank Fusion (RRF) to combine rankings from both methods.
    """
    
    def __init__(
        self,
        # embedding_model_name: str = 'sentence-transformers/multi-qa-mpnet-base-cos-v1',
        embedding_model_name: str = 'BAAI/bge-base-en-v1.5',
        use_cosine: bool = True,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            embedding_model_name: Name of sentence-transformer model
            use_cosine: If True, uses cosine similarity (recommended)
            bm25_k1: BM25 term frequency saturation parameter
            bm25_b: BM25 length normalization parameter
        """
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.use_cosine = use_cosine
        
        print("Initializing BM25...")
        self.bm25 = BM25(k1=bm25_k1, b=bm25_b)
        
        self.chunks = None
        self.dense_index = None
        self.embeddings = None
        
    def index_documents(self, chunks: List[str], show_progress: bool = True):
        """
        Index documents for both dense and sparse retrieval.
        
        Args:
            chunks: List of text chunks to index
            show_progress: Show progress bars
        """
        self.chunks = chunks
        print(f"\nIndexing {len(chunks)} documents...")
        
        # 1. Dense retrieval: Create embeddings
        print("Creating embeddings...")
        self.embeddings = self.embedding_model.encode(
            chunks, 
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        # 2. Normalize for cosine similarity
        if self.use_cosine:
            print("Normalizing embeddings for cosine similarity...")
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self.embeddings = self.embeddings / norms
            
            # Use Inner Product (equivalent to cosine with normalized vectors)
            dimension = self.embeddings.shape[1]
            self.dense_index = faiss.IndexFlatIP(dimension)
        else:
            # Use L2 distance
            dimension = self.embeddings.shape[1]
            self.dense_index = faiss.IndexFlatL2(dimension)
        
        self.dense_index.add(self.embeddings.astype('float32'))
        print(f"Dense index created with {self.dense_index.ntotal} vectors")
        
        # 3. Sparse retrieval: Fit BM25
        print("Fitting BM25 on corpus...")
        self.bm25.fit(chunks)
        print("BM25 fitted successfully")
        
        print("\n✅ Indexing complete!")
        
    def dense_search(self, query: str, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform dense (embedding-based) retrieval.
        
        Returns:
            Tuple of (scores, indices)
        """
        # Encode query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Normalize if using cosine
        if self.use_cosine:
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search
        scores, indices = self.dense_index.search(query_embedding.astype('float32'), k)
        
        return scores[0], indices[0]
    
    def sparse_search(self, query: str, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform sparse (BM25) retrieval.
        
        Returns:
            Tuple of (scores, indices)
        """
        return self.bm25.search(query, k)
    
    def reciprocal_rank_fusion(
        self,
        rankings: List[List[int]],
        scores_list: List[List[float]],
        k: int = 60
    ) -> List[Tuple[int, float]]:
        """
        Combine multiple rankings using Reciprocal Rank Fusion (RRF).
        
        RRF formula: score(d) = sum over all rankers of 1/(k + rank(d))
        where k is typically 60.
        
        Args:
            rankings: List of ranked document indices from different methods
            scores_list: List of scores from different methods (for breaking ties)
            k: Constant for RRF (default: 60, standard value)
            
        Returns:
            List of (doc_idx, fused_score) tuples, sorted by fused_score
        """
        rrf_scores = {}
        
        for ranking, scores in zip(rankings, scores_list):
            for rank, (doc_idx, score) in enumerate(zip(ranking, scores)):
                if doc_idx not in rrf_scores:
                    rrf_scores[doc_idx] = 0.0
                # RRF formula
                rrf_scores[doc_idx] += 1.0 / (k + rank + 1)
        
        # Sort by RRF score (descending)
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_results
    
    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        fusion_method: str = 'rrf',
        return_scores: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform hybrid search combining dense and sparse retrieval.
        
        Args:
            query: Search query
            k: Number of results to return
            dense_weight: Weight for dense retrieval (0-1)
            sparse_weight: Weight for sparse retrieval (0-1)
            fusion_method: 'rrf' (Reciprocal Rank Fusion) or 'weighted' (score-based)
            return_scores: If True, returns fused scores
            
        Returns:
            Tuple of (scores, indices) or just indices
        """
        # Retrieve from both methods (get more than k for better fusion)
        retrieve_k = k * 3
        
        dense_scores, dense_indices = self.dense_search(query, retrieve_k)
        sparse_scores, sparse_indices = self.sparse_search(query, retrieve_k)
        
        if fusion_method == 'rrf':
            # Reciprocal Rank Fusion
            fused_results = self.reciprocal_rank_fusion(
                [dense_indices.tolist(), sparse_indices.tolist()],
                [dense_scores.tolist(), sparse_scores.tolist()]
            )
            
            # Extract top-k
            top_k_results = fused_results[:k]
            indices = np.array([idx for idx, score in top_k_results])
            scores = np.array([score for idx, score in top_k_results])
            
        else:  # weighted fusion
            # Normalize scores to [0, 1] range
            dense_scores_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-8)
            sparse_scores_norm = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min() + 1e-8)
            
            # Combine all unique indices
            all_indices = list(set(dense_indices.tolist() + sparse_indices.tolist()))
            
            # Calculate weighted scores
            combined_scores = {}
            for idx in all_indices:
                score = 0.0
                if idx in dense_indices:
                    pos = np.where(dense_indices == idx)[0][0]
                    score += dense_weight * dense_scores_norm[pos]
                if idx in sparse_indices:
                    pos = np.where(sparse_indices == idx)[0][0]
                    score += sparse_weight * sparse_scores_norm[pos]
                combined_scores[idx] = score
            
            # Sort and get top-k
            sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
            indices = np.array([idx for idx, score in sorted_results])
            scores = np.array([score for idx, score in sorted_results])
        
        if return_scores:
            return scores, indices
        return indices
    
    def search(
        self,
        query: str,
        k: int = 3,
        method: str = 'hybrid',
        **kwargs
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Main search interface.
        
        Args:
            query: Search query
            k: Number of results
            method: 'hybrid', 'dense', or 'sparse'
            **kwargs: Additional arguments for specific methods
            
        Returns:
            Tuple of (retrieved_chunks, scores, indices)
        """
        if method == 'dense':
            scores, indices = self.dense_search(query, k)
        elif method == 'sparse':
            scores, indices = self.sparse_search(query, k)
        elif method == 'hybrid':
            scores, indices = self.hybrid_search(query, k, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'hybrid', 'dense', or 'sparse'")
        
        retrieved_chunks = [self.chunks[i] for i in indices]
        
        return retrieved_chunks, scores, indices
    
    def compare_methods(self, query: str, k: int = 5):
        """
        Compare retrieval results from all three methods.
        
        Useful for debugging and understanding method differences.
        """
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print(f"{'='*80}\n")
        
        methods = ['dense', 'sparse', 'hybrid']
        for method in methods:
            print(f"{method.upper()} RETRIEVAL:")
            print("-" * 80)
            chunks, scores, indices = self.search(query, k, method=method)
            
            for i, (chunk, score, idx) in enumerate(zip(chunks, scores, indices)):
                print(f"\nRank {i+1} | Score: {score:.4f} | Chunk ID: {idx}")
                print(f"Text: {chunk[:150]}...")
            
            print("\n")


# ========== USAGE EXAMPLE ==========
if __name__ == "__main__":
    
    # Sample chunks (replace with your actual chunks)
    sample_chunks = [
        "This study examines water access in rural southern Syria.",
        "Methods: We conducted household surveys across 500 farms.",
        "Data collection occurred between 2016 and 2017 in three districts.",
        "Results show that 65% of households lack clean water access.",
        "The methodology included structured interviews and water testing.",
        "Discussion: Water scarcity affects agricultural productivity significantly.",
    ]
    
    # Initialize hybrid retriever
    retriever = HybridRetriever(
        embedding_model_name='sentence-transformers/multi-qa-mpnet-base-cos-v1',
        use_cosine=True
    )
    
    # Index documents
    retriever.index_documents(sample_chunks)
    
    # Search using different methods
    query = "What methods were used for data collection?"
    
    print("\n" + "="*80)
    print("COMPARING RETRIEVAL METHODS")
    print("="*80)
    
    retriever.compare_methods(query, k=3)
    
    # Use hybrid search with custom weights
    print("\n" + "="*80)
    print("HYBRID SEARCH WITH CUSTOM WEIGHTS")
    print("="*80)
    
    chunks, scores, indices = retriever.search(
        query,
        k=3,
        method='hybrid',
        dense_weight=0.7,
        sparse_weight=0.3,
        fusion_method='rrf'
    )
    
    print(f"\nQuery: {query}\n")
    for i, (chunk, score, idx) in enumerate(zip(chunks, scores, indices)):
        print(f"Rank {i+1} | Score: {score:.4f} | Chunk {idx}")
        print(f"  {chunk}\n")

