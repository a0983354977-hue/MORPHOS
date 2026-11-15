"""Associative Memory & State Retrieval Module (T3.2).

High-dimensional vector space similarity search over historical snapshots.
When the system encounters an anomalous state, this module retrieves the
most similar historical trajectories to inform prediction and diagnosis.

Key features:
  - Cosine / Euclidean distance in feature space
  - Efficient k-NN search over compressed history
  - Temporal coherence filtering (prefer temporally consistent matches)
  - Anomaly-triggered retrieval with match scoring
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional
import heapq


@dataclass
class RetrievalConfig:
    """Configuration for associative memory retrieval."""
    max_history: int = 8192          # max vectors in search index
    top_k: int = 5                   # number of nearest neighbors to return
    similarity_metric: str = "cosine"  # "cosine", "euclidean", "dot"
    temporal_weight: float = 0.1     # weight for temporal proximity bonus
    min_similarity: float = 0.3      # threshold for meaningful matches
    enable_temporal_filter: bool = True  # penalize temporally distant matches


@dataclass
class MatchResult:
    """Result of a similarity search."""
    index: int              # index in history
    step: int               # global simulation step
    similarity: float       # similarity score (0-1 for cosine)
    distance: float         # distance metric value
    feature_vector: np.ndarray  # the matched feature vector
    residual: float         # associated residual value
    temporal_bonus: float = 0.0  # temporal coherence bonus


class AssociativeMemory:
    """Associative memory for historical state retrieval.
    
    Stores compressed feature vectors with metadata and supports
    efficient similarity search for anomaly diagnosis.
    """
    
    def __init__(self, feature_dim: int, config: Optional[RetrievalConfig] = None):
        self.feature_dim = feature_dim
        self.cfg = config or RetrievalConfig()
        
        # --- Main storage (fixed capacity, ring buffer) ---
        self._capacity = self.cfg.max_history
        self._features = np.zeros((self._capacity, feature_dim), dtype=float)
        self._residuals = np.zeros(self._capacity, dtype=float)
        self._steps = np.zeros(self._capacity, dtype=int)
        self._head = 0
        self._size = 0
        self._total_added = 0
        
        # --- Normalized features cache for cosine similarity ---
        self._normed_features: Optional[np.ndarray] = None
        self._norms: Optional[np.ndarray] = None
        self._cache_valid = False
        
    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def add(self, feature: np.ndarray, residual: float, step: int) -> None:
        """Add a feature vector to the associative memory."""
        feat = np.asarray(feature, dtype=float).ravel()
        assert feat.shape[0] == self.feature_dim
        
        self._features[self._head] = feat
        self._residuals[self._head] = float(residual)
        self._steps[self._head] = int(step)
        self._head = (self._head + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)
        self._total_added += 1
        self._cache_valid = False  # invalidate norm cache
    
    def add_batch(self, features: np.ndarray, residuals: np.ndarray, 
                  steps: np.ndarray) -> None:
        """Add multiple feature vectors efficiently."""
        n = features.shape[0]
        for i in range(n):
            self.add(features[i], residuals[i], steps[i])
    
    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------
    def _ensure_norm_cache(self) -> None:
        """Compute and cache normalized features for cosine similarity."""
        if self._cache_valid:
            return
        if self._size == 0:
            self._normed_features = np.zeros((0, self.feature_dim))
            self._norms = np.zeros(0)
            self._cache_valid = True
            return
            
        # Get valid features in chronological order
        feats = self.get_all_features()
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        self._normed_features = feats / norms
        self._norms = norms.ravel()
        self._cache_valid = True
    
    def get_all_features(self) -> np.ndarray:
        """Get all stored features in chronological order (oldest first)."""
        if self._size == 0:
            return np.zeros((0, self.feature_dim))
        if self._size < self._capacity:
            return self._features[:self._size].copy()
        # Ring buffer: oldest is at head, newest at head-1
        return np.vstack([self._features[self._head:], self._features[:self._head]])
    
    def get_all_residuals(self) -> np.ndarray:
        """Get all residuals in chronological order."""
        if self._size == 0:
            return np.zeros(0)
        if self._size < self._capacity:
            return self._residuals[:self._size].copy()
        return np.hstack([self._residuals[self._head:], self._residuals[:self._head]])
    
    def get_all_steps(self) -> np.ndarray:
        """Get all steps in chronological order."""
        if self._size == 0:
            return np.zeros(0, dtype=int)
        if self._size < self._capacity:
            return self._steps[:self._size].copy()
        return np.hstack([self._steps[self._head:], self._steps[:self._head]])
    
    def query(self, query_vec: np.ndarray, 
              current_step: Optional[int] = None,
              top_k: Optional[int] = None,
              exclude_recent: int = 0) -> list[MatchResult]:
        """Find most similar historical states to query_vec.
        
        Args:
            query_vec: Feature vector to search for
            current_step: Current simulation step (for temporal filtering)
            top_k: Number of results (default from config)
            exclude_recent: Exclude the most recent N entries
            
        Returns:
            List of MatchResult sorted by similarity (best first)
        """
        if self._size == 0:
            return []
            
        k = top_k or self.cfg.top_k
        k = min(k, self._size - exclude_recent)
        if k <= 0:
            return []
            
        q = np.asarray(query_vec, dtype=float).ravel()
        assert q.shape[0] == self.feature_dim
        
        # Get valid features
        feats = self.get_all_features()
        residuals = self.get_all_residuals()
        steps = self.get_all_steps()
        
        # Exclude recent if requested
        if exclude_recent > 0:
            feats = feats[:-exclude_recent]
            residuals = residuals[:-exclude_recent]
            steps = steps[:-exclude_recent]
            if len(feats) == 0:
                return []
        
        # Compute similarity
        if self.cfg.similarity_metric == "cosine":
            scores = self._cosine_similarity(q, feats)
        elif self.cfg.similarity_metric == "euclidean":
            scores = -np.linalg.norm(feats - q, axis=1)  # negative distance
        elif self.cfg.similarity_metric == "dot":
            scores = feats @ q
        else:
            raise ValueError(f"Unknown metric: {self.cfg.similarity_metric}")
        
        # Apply temporal filtering if enabled
        if self.cfg.enable_temporal_filter and current_step is not None:
            temporal_dists = np.abs(steps - current_step)
            max_temporal = np.max(temporal_dists) if len(temporal_dists) > 0 else 1
            if max_temporal > 0:
                temporal_bonus = self.cfg.temporal_weight * (1.0 - temporal_dists / max_temporal)
                scores = scores + temporal_bonus
            else:
                temporal_bonus = np.zeros_like(scores)
        else:
            temporal_bonus = np.zeros_like(scores)
        
        # Get top-k indices
        # Use argpartition for efficiency
        if k < len(scores):
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        else:
            top_indices = np.argsort(scores)[::-1]
        
        # Build results
        results = []
        for idx in top_indices:
            sim = float(scores[idx])
            # Convert to 0-1 similarity for cosine
            if self.cfg.similarity_metric == "cosine":
                similarity = max(0.0, sim)  # cosine in [-1, 1], clip negative
            elif self.cfg.similarity_metric == "euclidean":
                similarity = 1.0 / (1.0 + abs(sim))  # convert distance to similarity
            else:
                similarity = max(0.0, sim)
            
            if similarity < self.cfg.min_similarity:
                continue
                
            results.append(MatchResult(
                index=int(idx),
                step=int(steps[idx]),
                similarity=similarity,
                distance=float(abs(scores[idx])),
                feature_vector=feats[idx].copy(),
                residual=float(residuals[idx]),
                temporal_bonus=float(temporal_bonus[idx])
            ))
        
        return results
    
    def _cosine_similarity(self, q: np.ndarray, feats: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between q and all features."""
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-12:
            return np.zeros(len(feats))
        
        self._ensure_norm_cache()
        if self._normed_features is not None and len(self._normed_features) == len(feats):
            return self._normed_features @ (q / q_norm)
        else:
            # Fallback: compute directly
            feat_norms = np.linalg.norm(feats, axis=1)
            feat_norms = np.maximum(feat_norms, 1e-12)
            return (feats @ q) / (feat_norms * q_norm)
    
    # ------------------------------------------------------------------
    # Anomaly-triggered retrieval
    # ------------------------------------------------------------------
    def retrieve_for_anomaly(self, current_state: np.ndarray, 
                             current_residual: float,
                             current_step: int,
                             residual_threshold: float = 2.0) -> list[MatchResult]:
        """Specialized retrieval when anomaly is detected.
        
        Args:
            current_state: Current feature vector
            current_residual: Current prediction residual
            current_step: Current simulation step
            residual_threshold: Multiplier for baseline residual to trigger
            
        Returns:
            Historical matches that resemble the anomalous state
        """
        # For anomaly retrieval, we search for similar STATES (not residuals)
        # but we also consider residual history to find analogous situations
        matches = self.query(current_state, current_step=current_step, 
                            top_k=self.cfg.top_k * 2)
        
        # Filter: prefer matches that had HIGH residuals (similar anomalous situations)
        # or matches that PRECEDED recovery (low residuals after high)
        if len(matches) > 0:
            match_residuals = np.array([m.residual for m in matches])
            # Boost matches with high residuals (similar anomalies)
            high_residual_mask = match_residuals > np.percentile(match_residuals, 75)
            for i, m in enumerate(matches):
                if high_residual_mask[i]:
                    m.similarity *= 1.2  # boost
            # Re-sort
            matches.sort(key=lambda x: x.similarity, reverse=True)
        
        return matches[:self.cfg.top_k]
    
    # ------------------------------------------------------------------
    # Statistics & inspection
    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        """Return memory statistics."""
        if self._size == 0:
            return {"size": 0, "capacity": self._capacity}
        
        residuals = self.get_all_residuals()
        return {
            "size": self._size,
            "capacity": self._capacity,
            "total_added": self._total_added,
            "residual_mean": float(np.mean(residuals)),
            "residual_std": float(np.std(residuals)),
            "residual_min": float(np.min(residuals)),
            "residual_max": float(np.max(residuals)),
            "step_range": (int(self.get_all_steps()[0]), int(self.get_all_steps()[-1])),
        }
    
    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        """Serialize for checkpointing."""
        return {
            "feature_dim": self.feature_dim,
            "capacity": self._capacity,
            "head": self._head,
            "size": self._size,
            "total_added": self._total_added,
            "features": self._features.copy(),
            "residuals": self._residuals.copy(),
            "steps": self._steps.copy(),
            "cfg": {
                "max_history": self.cfg.max_history,
                "top_k": self.cfg.top_k,
                "similarity_metric": self.cfg.similarity_metric,
                "temporal_weight": self.cfg.temporal_weight,
                "min_similarity": self.cfg.min_similarity,
                "enable_temporal_filter": self.cfg.enable_temporal_filter,
            }
        }
    
    def load_state_dict(self, d: dict) -> None:
        """Restore from checkpoint."""
        self.feature_dim = d["feature_dim"]
        self._capacity = d["capacity"]
        self._head = d["head"]
        self._size = d["size"]
        self._total_added = d["total_added"]
        self._features = np.asarray(d["features"], dtype=float)
        self._residuals = np.asarray(d["residuals"], dtype=float)
        self._steps = np.asarray(d["steps"], dtype=int)
        
        cfg = d["cfg"]
        self.cfg = RetrievalConfig(
            max_history=cfg["max_history"],
            top_k=cfg["top_k"],
            similarity_metric=cfg["similarity_metric"],
            temporal_weight=cfg["temporal_weight"],
            min_similarity=cfg["min_similarity"],
            enable_temporal_filter=cfg["enable_temporal_filter"],
        )
        self._cache_valid = False


def make_associative_memory(feature_dim: int,
                           max_history: int = 8192,
                           top_k: int = 5,
                           metric: str = "cosine",
                           temporal_weight: float = 0.1,
                           min_similarity: float = 0.3,
                           enable_temporal_filter: bool = True) -> AssociativeMemory:
    """Factory for backward-compatible creation."""
    config = RetrievalConfig(
        max_history=max_history,
        top_k=top_k,
        similarity_metric=metric,
        temporal_weight=temporal_weight,
        min_similarity=min_similarity,
        enable_temporal_filter=enable_temporal_filter,
    )
    return AssociativeMemory(feature_dim, config)