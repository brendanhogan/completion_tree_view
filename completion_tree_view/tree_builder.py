"""
Tree builder module for CompletionTreeView.

This module provides classes and functions to build a token prefix tree from
multiple model completions, with support for tracking statistical metrics.
"""

import json
from typing import Dict, List, Optional, Tuple, Union, Any
import math

class TreeNode:
    """Represents a node in the token prefix tree.
    
    Each node represents a token in the tree of completions, storing information
    about the token ID, how many completions pass through this node, and whether
    this node represents the end of a completion.
    
    Attributes:
        token_id: The ID of the token this node represents.
        unique_id: A unique identifier for this node instance.
        children: Dictionary mapping token IDs to child TreeNode instances.
        is_end_of_completion: Whether this node marks the end of a completion.
        score: Optional score for this completion (if this node is an end node).
        count: How many completions pass through or end at this node.
        structural_hash: Hash representing this node's subtree structure.
        descendant_leaf_count: Number of completion endpoints in this subtree.
        descendant_score_sum: Sum of scores of descendant completion endpoints.
    """
    
    def __init__(self, token_id: Any = None, unique_id: int = None):
        """Initialize a TreeNode.
        
        Args:
            token_id: The token ID this node represents.
            unique_id: A unique identifier for this node.
        """
        self.token_id = token_id
        self.unique_id = unique_id
        self.children: Dict[Any, TreeNode] = {}  # token_id -> TreeNode
        self.is_end_of_completion = False
        self.score: Optional[float] = None  # Score if it's an end node
        self.count = 0  # How many completions pass through/end here
        self.structural_hash: Optional[int] = None  # Computed hash
        # Statistics for descendants
        self.descendant_leaf_count = 0
        self.descendant_score_sum = 0.0


class CompletionTree:
    """A tree structure representing multiple token-based completions.
    
    This class builds and manages a prefix tree (trie) of completions, where each
    path from root to leaf represents a completion. It provides methods to compute
    structural hashes for identifying similar subtrees and statistics about scores.
    
    Attributes:
        root: The root node of the tree.
        has_scores: Whether score information is available for completions.
    """
    
    def __init__(self, 
                 completions: List[List[int]], 
                 scores: Optional[List[float]] = None):
        """Initialize a CompletionTree.
        
        Args:
            completions: List of completions, where each completion is a list of token IDs.
            scores: Optional list of scores for each completion. If provided, must be the
                   same length as completions. Scores should be in the range [0, 1] where
                   1 represents a "correct" or "good" completion.
        """
        self.has_scores = scores is not None
        self._next_id = 0
        
        # Build the tree from completions
        self.root = self._build_tree(completions, scores)
        
        # Compute hashes and statistics
        self._compute_structural_hashes(self.root)
        self._compute_leaf_stats(self.root)
    
    def _get_next_id(self) -> int:
        """Get the next unique ID for a node.
        
        Returns:
            A unique integer ID.
        """
        id_val = self._next_id
        self._next_id += 1
        return id_val
    
    def _build_tree(self, 
                    completions: List[List[int]], 
                    scores: Optional[List[float]]) -> TreeNode:
        """Build a prefix tree from a list of completions.
        
        Args:
            completions: List of completions, where each completion is a list of token IDs.
            scores: Optional list of scores for each completion.
            
        Returns:
            The root TreeNode of the constructed tree.
        """
        root = TreeNode("ROOT", self._get_next_id())
        root.count = len(completions)
        
        for i, tokens in enumerate(completions):
            node = root
            
            for token_id in tokens:
                if token_id not in node.children:
                    node.children[token_id] = TreeNode(token_id, self._get_next_id())
                node = node.children[token_id]
                node.count += 1
            
            # Mark as end of completion
            node.is_end_of_completion = True
            
            # Assign score if available
            if scores is not None and i < len(scores):
                node.score = scores[i]
                
        return root
    
    def _stable_hash(self, value: Any) -> int:
        """Create a stable hash for various types of values.
        
        Args:
            value: The value to hash.
            
        Returns:
            A stable hash value.
        """
        return hash(json.dumps(value, sort_keys=True))
    
    def _compute_structural_hashes(self, node: TreeNode) -> int:
        """Recursively compute and store structural hashes for the node and its descendants.
        
        This method uses post-order traversal to compute a hash for each node based on its
        token ID, completion status, score, and the hashes of its children.
        
        Args:
            node: The node to compute a hash for.
            
        Returns:
            The hash value for the node.
        """
        if node.structural_hash is not None:
            return node.structural_hash
        
        child_hashes = []
        sorted_children = sorted(node.children.items())
        for token_id, child_node in sorted_children:
            child_hashes.append((token_id, self._compute_structural_hashes(child_node)))
        
        # Hash components: token, end_flag, score, and children
        hash_components = (
            node.token_id,
            node.is_end_of_completion,
            node.score if node.is_end_of_completion else None,
            tuple(child_hashes)
        )
        
        node.structural_hash = self._stable_hash(hash_components)
        return node.structural_hash
    
    def _compute_leaf_stats(self, node: TreeNode) -> Tuple[int, float]:
        """Recursively compute descendant leaf statistics for a node.
        
        This method calculates for each node:
        1. The total number of descendant leaf nodes
        2. The sum of scores from descendant leaf nodes
        
        Args:
            node: The node to compute statistics for.
            
        Returns:
            A tuple of (total_leaves, score_sum).
        """
        if node.is_end_of_completion:
            # Base case: This node is a leaf
            total_leaves = 1
            score_sum = node.score if node.score is not None else 0.0
        else:
            # Recursive case: Sum stats from children
            total_leaves = 0
            score_sum = 0.0
            for child_node in node.children.values():
                child_total, child_sum = self._compute_leaf_stats(child_node)
                total_leaves += child_total
                score_sum += child_sum
        
        # Store computed stats on the node
        node.descendant_leaf_count = total_leaves
        node.descendant_score_sum = score_sum
        
        return total_leaves, score_sum
    
    def get_node_score_percentage(self, node: TreeNode) -> Optional[float]:
        """Calculate the percentage of "correct" (high scoring) descendant leaves.
        
        Args:
            node: The node to get score percentage for.
            
        Returns:
            A float between 0.0 and 1.0 representing the percentage of total score
            out of maximum possible score, or None if there are no descendants or scores.
        """
        if not self.has_scores or node.descendant_leaf_count == 0:
            return None
            
        # Calculate average score (between 0.0 and 1.0)
        return node.descendant_score_sum / node.descendant_leaf_count 