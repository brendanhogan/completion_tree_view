"""
CompletionTreeView - A library for visualizing token trees from language model completions.

This package provides tools to build and visualize token prefix trees from multiple
completions produced by language models, allowing for analysis of generation patterns
and decision paths.
"""

from completion_tree_view.tree_builder import CompletionTree, TreeNode
from completion_tree_view.plotter import plot_tree_pdf, plot_tree_html

__version__ = "0.1.0"

__all__ = [
    "CompletionTree", 
    "TreeNode",
    "plot_tree_pdf",
    "plot_tree_html"
] 