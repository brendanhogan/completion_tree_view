# CompletionTreeView

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A lightweight, focused library for visualizing language model completion patterns as token trees.

CompletionTreeView builds visual representations of token-based completions from language models, enabling researchers and developers to:

- Explore how language models generate text at the token level
- Compare the generation paths of multiple completions for the same prompt
- Visualize where different completions diverge or converge
- Analyze the relationships between completion quality (scores) and generation patterns

## Features

- **Interactive HTML Visualization** - Explore completion DAGs in your browser with zooming, panning and node selection
- **Static PDF Visualization** - Generate publication-ready PDF visualizations (requires Graphviz)
- **Path Merging** - Automatically detects and merges identical subtrees to create a directed acyclic graph (DAG)
- **Score Coloring** - Optionally visualize correctness or quality scores through node coloring
- **Lightweight & Focused** - Clean, well-documented code with minimal dependencies
- **Easy to Use** - Simple API that works with any tokenizer

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/CompletionTreeView.git
cd CompletionTreeView

# Install Python dependencies
pip install -r requirements.txt

# Optional: Install as a development package
pip install -e .
```

### Graphviz Installation (Required for PDF Generation)

For PDF visualization, you need to install both:
1. The Python `graphviz` package (included in requirements.txt)
2. The Graphviz system executable

**Install the Graphviz system executable:**

- **Ubuntu/Debian**: 
  ```
  sudo apt-get install graphviz
  ```

- **macOS**:
  ```
  brew install graphviz
  ```

- **Windows**:
  Download and install from the [Graphviz website](https://graphviz.org/download/)
  Then add the installation directory to your PATH

**Verify Installation:**
```
dot -V
```
If properly installed, this should display the Graphviz version.

## Quick Start

```python
from transformers import AutoTokenizer
from completion_tree_view import CompletionTree, plot_tree_pdf, plot_tree_html

# 1. Load a tokenizer (any tokenizer that can decode token ids)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 2. Prepare your completions as lists of token IDs
# Example: two completions with different tokens
completions = [
    [15496, 257, 3303, 12],     # "The answer is 56"
    [15496, 257, 11241, 2674]   # "The answer is wrong"
]

# 3. Optional: Provide scores for each completion (1.0 = correct/good)
scores = [1.0, 0.0]  # First completion is correct, second is incorrect

# 4. Create the completion tree
tree = CompletionTree(completions, scores)

# 5. Generate visualizations
plot_tree_html(tree, tokenizer, "my_tree.html")     # Always works
plot_tree_pdf(tree, tokenizer, "my_tree.pdf")       # Requires Graphviz
```

## Examples

The repository includes a working example:

- **[examples/math_example.py](examples/math_example.py)**: Demonstrates generating completions for a math problem using Qwen2.5-7B-Instruct, evaluating their correctness, and visualizing the results.

When you run this example, it generates:
1. A JSON file with all completions: `outputs/math_completions.json`
2. An interactive HTML visualization: `outputs/math_example.html`
3. A retro-futuristic art deco style PDF: `outputs/math_example.pdf` (if Graphviz is installed)

To run the example:

```bash
cd CompletionTreeView
python examples/math_example.py
```

## Visualization Examples

The library produces two types of visualizations:

### PDF Visualization (via Graphviz)

![Example PDF Visualization](https://via.placeholder.com/650x400?text=Example+PDF+Visualization)

### Interactive HTML Visualization (via vis.js)

![Example HTML Visualization](https://via.placeholder.com/650x400?text=Example+HTML+Visualization)

## Node Information

In both visualizations, nodes display the following information:

- **Token Text**: The decoded text of the token
- **T**: Token ID
- **N**: Number of completions passing through this node
- **L**: Number of completion endpoints (leaves) in this node's subtree
- **Score**: If scores are provided, the percentage of "correct" completions (based on scores)

Nodes are colored on a gradient from red (low score) to green (high score) if scores are provided.

## Documentation

### CompletionTree Class

```python
tree = CompletionTree(completions, scores=None)
```
- `completions`: List of completions, where each completion is a list of token IDs
- `scores`: Optional list of scores for each completion (between 0.0 and 1.0)

### Visualization Functions

```python
plot_tree_pdf(tree, tokenizer, output_filename, view=False, fail_silently=False)
```
- `tree`: A CompletionTree instance
- `tokenizer`: Tokenizer with a `decode()` method that converts token IDs to text
- `output_filename`: Path to save the PDF
- `view`: Whether to automatically open the PDF after creation
- `fail_silently`: If True, return False on error instead of raising exception

```python
plot_tree_html(tree, tokenizer, output_filename)
```
- `tree`: A CompletionTree instance
- `tokenizer`: Tokenizer with a `decode()` method that converts token IDs to text
- `output_filename`: Path to save the HTML file

## Use Cases

- **Research**: Analyze model behavior by visualizing completion patterns
- **Education**: Show students how LLMs generate text at the token level
- **Debugging**: Identify where models diverge from expected generation paths
- **Quality Analysis**: Visualize the relationship between generation paths and completion quality

## Troubleshooting

### PDF Generation Issues

If you encounter errors with PDF generation:

1. Ensure Graphviz is properly installed (both Python package and system executable)
2. Verify the Graphviz executable is in your PATH by running `dot -V`
3. If you don't need PDFs, you can use the HTML visualization which has no external dependencies
4. Add `fail_silently=True` to continue without errors if PDF generation fails

## Citation

If you use CompletionTreeView in your research, please cite:

```bibtex
@software{completiontreeview2023,
  author = {Brendan Hogan},
  title = {CompletionTreeView: A Tool for Visualizing Language Model Completion Trees},
  year = {2023},
}
```

## License

MIT License - see the [LICENSE](LICENSE) file for details. 