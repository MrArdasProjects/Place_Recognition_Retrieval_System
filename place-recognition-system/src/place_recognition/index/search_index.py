"""Search index for efficient similarity-based retrieval."""

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from place_recognition.logging_config import get_logger

logger = get_logger(__name__)


class SearchIndex:
    """Cosine similarity-based retrieval index.
    
    Supports both NumPy brute-force and FAISS-based indexing.
    Uses inner product for L2-normalized embeddings (equivalent to cosine similarity).
    
    Args:
        embeddings: Gallery embeddings [N, feature_dim], L2-normalized
        records: List of manifest records corresponding to embeddings
        index_type: Index type ("flat" for now, future: "ivf", "hnsw")
        use_faiss: Whether to use FAISS library (faster for large datasets)
    
    Example:
        >>> index = SearchIndex(gallery_embeddings, gallery_records)
        >>> index.build()
        >>> results = index.search(query_embeddings, top_k=5)
    """
    
    def __init__(
        self,
        embeddings: np.ndarray,
        records: List[Dict],
        index_type: str = "flat",
        use_faiss: bool = False,
    ) -> None:
        if len(embeddings) != len(records):
            raise ValueError(
                f"Embeddings count ({len(embeddings)}) "
                f"!= records count ({len(records)})"
            )
        
        self.embeddings = embeddings
        self.records = records
        self.index_type = index_type
        self.use_faiss = use_faiss
        
        self.num_items = len(embeddings)
        self.feature_dim = embeddings.shape[1]
        
        self.faiss_index = None
        self.is_built = False
        
        # Statistics
        self.stats = {
            "build_time": 0.0,
            "total_queries": 0,
            "total_search_time": 0.0,
        }
        
        logger.info(f"SearchIndex initialized:")
        logger.info(f"  Gallery size: {self.num_items}")
        logger.info(f"  Feature dim: {self.feature_dim}")
        logger.info(f"  Index type: {self.index_type}")
        logger.info(f"  Use FAISS: {self.use_faiss}")
    
    def build(self) -> None:
        """Build the search index."""
        logger.info("Building search index...")
        start_time = time.time()
        
        if self.use_faiss:
            self._build_faiss_index()
        else:
            self._build_numpy_index()
        
        self.is_built = True
        self.stats["build_time"] = time.time() - start_time
        
        logger.info(f"Index built in {self.stats['build_time']:.3f}s")
    
    def _build_faiss_index(self) -> None:
        """Build FAISS index (inner product)."""
        try:
            import faiss
        except ImportError:
            logger.warning(
                "FAISS not installed. Falling back to NumPy. "
                "Install with: pip install faiss-cpu"
            )
            self.use_faiss = False
            self._build_numpy_index()
            return
        
        # Create flat inner product index
        # For L2-normalized vectors: inner product = cosine similarity
        if self.index_type == "flat":
            self.faiss_index = faiss.IndexFlatIP(self.feature_dim)
        else:
            raise ValueError(f"Index type '{self.index_type}' not supported with FAISS")
        
        # Add embeddings
        self.faiss_index.add(self.embeddings.astype(np.float32))
        
        logger.info(f"FAISS index built: {self.faiss_index.ntotal} vectors")
    
    def _build_numpy_index(self) -> None:
        """Build NumPy index (stores embeddings for brute-force search)."""
        # NumPy doesn't require explicit indexing
        # Just verify embeddings are valid
        if not np.all(np.isfinite(self.embeddings)):
            raise ValueError("Embeddings contain invalid values (inf/nan)")
        
        logger.info(f"NumPy index ready: {self.num_items} vectors")
    
    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 5,
        return_similarities: bool = True,
    ) -> List[List[Dict]]:
        """Search for most similar gallery images.
        
        Args:
            query_embeddings: Query embeddings [Q, feature_dim], L2-normalized
            top_k: Number of top results to return per query
            return_similarities: Whether to include similarity scores
        
        Returns:
            List of results for each query. Each result is a list of dicts with:
                - image_path: Path to gallery image
                - place_id: Place identifier
                - similarity: Cosine similarity score (if return_similarities=True)
                - rank: Rank in results (0-indexed)
        """
        if not self.is_built:
            raise RuntimeError("Index not built. Call build() first.")
        
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        
        num_queries = len(query_embeddings)
        top_k = min(top_k, self.num_items)  # Can't return more than we have
        
        logger.info(f"Searching {num_queries} queries for top-{top_k} results...")
        start_time = time.time()
        
        if self.use_faiss and self.faiss_index is not None:
            similarities, indices = self._search_faiss(query_embeddings, top_k)
        else:
            similarities, indices = self._search_numpy(query_embeddings, top_k)
        
        search_time = time.time() - start_time
        self.stats["total_queries"] += num_queries
        self.stats["total_search_time"] += search_time
        
        avg_time = search_time / num_queries * 1000  # ms per query
        logger.info(f"Search completed in {search_time:.3f}s ({avg_time:.2f}ms/query)")
        
        # Format results
        results = self._format_results(
            indices, similarities, return_similarities
        )
        
        return results
    
    def _search_faiss(
        self, query_embeddings: np.ndarray, top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search using FAISS index.
        
        Returns:
            Tuple of (similarities, indices)
        """
        similarities, indices = self.faiss_index.search(
            query_embeddings.astype(np.float32), top_k
        )
        return similarities, indices
    
    def _search_numpy(
        self, query_embeddings: np.ndarray, top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search using NumPy brute-force.
        
        Returns:
            Tuple of (similarities, indices)
        """
        # Compute cosine similarity: query @ gallery.T
        # Works because both are L2-normalized
        similarities = np.dot(query_embeddings, self.embeddings.T)
        
        # Get top-k indices for each query
        # argsort returns ascending order, so we reverse
        indices = np.argsort(similarities, axis=1)[:, ::-1][:, :top_k]
        
        # Get corresponding similarities
        batch_indices = np.arange(len(query_embeddings))[:, None]
        top_similarities = similarities[batch_indices, indices]
        
        return top_similarities, indices
    
    def _format_results(
        self,
        indices: np.ndarray,
        similarities: np.ndarray,
        return_similarities: bool,
    ) -> List[List[Dict]]:
        """Format search results into structured output.
        
        Args:
            indices: [num_queries, top_k] indices of retrieved items
            similarities: [num_queries, top_k] similarity scores
            return_similarities: Whether to include similarity scores
        
        Returns:
            List of results for each query
        """
        results = []
        
        for query_idx in range(len(indices)):
            query_results = []
            
            for rank, gallery_idx in enumerate(indices[query_idx]):
                record = self.records[gallery_idx]
                
                result = {
                    "image_path": record["image_path"],
                    "place_id": record["place_id"],
                    "rank": rank,
                }
                
                if return_similarities:
                    result["similarity"] = float(similarities[query_idx, rank])
                
                query_results.append(result)
            
            results.append(query_results)
        
        return results
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        stats = self.stats.copy()
        if stats["total_queries"] > 0:
            stats["avg_search_time"] = stats["total_search_time"] / stats["total_queries"]
        return stats
    
    def save(self, output_path: Path) -> None:
        """Save index to disk.
        
        Args:
            output_path: Path to save index (.npz file)
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(
            output_path,
            embeddings=self.embeddings,
            records=np.array(self.records, dtype=object),
            index_type=self.index_type,
            use_faiss=self.use_faiss,
        )
        
        logger.info(f"Index saved to: {output_path}")
    
    @classmethod
    def load(cls, index_path: Path) -> "SearchIndex":
        """Load index from disk.
        
        Args:
            index_path: Path to saved index (.npz file)
        
        Returns:
            Loaded SearchIndex instance
        """
        data = np.load(index_path, allow_pickle=True)
        
        embeddings = data["embeddings"]
        records = data["records"].tolist()
        index_type = str(data["index_type"])
        use_faiss = bool(data["use_faiss"])
        
        index = cls(
            embeddings=embeddings,
            records=records,
            index_type=index_type,
            use_faiss=use_faiss,
        )
        
        index.build()
        
        logger.info(f"Index loaded from: {index_path}")
        return index
