"""
Plotter module for CompletionTreeView.

This module provides functions to visualize CompletionTree objects as PDF files
using Graphviz or as interactive HTML using the vis.js library.
"""

import html
import json
import os
from typing import Dict, List, Optional, Tuple, Any, Set

# Try to import graphviz - it's required for PDF plotting
try:
    import graphviz
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False

from completion_tree_view.tree_builder import TreeNode, CompletionTree


def _get_color_for_score(score: Optional[float], 
                         html_format: bool = False) -> Dict[str, str]:
    """Get a color based on a node's score percentage.
    
    Creates a continuous gradient from red (0% correct) through
    orange, yellow, lime to green (100% correct) with unique colors
    for every small percentage change.
    
    Args:
        score: Score percentage between 0.0 and 1.0, or None.
        html_format: Whether to return HSL (for HTML) or hex (for PDF) format.
        
    Returns:
        Dictionary with 'background' and 'border' color values.
    """
    if score is None:
        # Neutral grey for nodes without scores
        if html_format:
            return {
                'background': "hsl(0, 0%, 85%)",
                'border': "hsl(0, 0%, 65%)"
            }
        else:
            return {
                'background': "#D8D8D8",  # Light grey
                'border': "#A9A9A9"       # Dark grey
            }
    
    # Clamp score between 0 and 1
    score = max(0.0, min(1.0, score))
    
    if html_format:
        # Continuous HSL gradient - more precise with decimal points
        # Red (0) to Yellow (60) to Green (120)
        if score < 0.5:
            # Red (0°) to Yellow (60°)
            hue = score * 120.0  # This creates orange and yellows in between
        else:
            # Yellow (60°) to Green (120°)
            hue = score * 120.0
            
        # Adjust saturation and brightness for best visibility
        saturation = 85.0  # Slightly more saturated
        lightness = 65.0
        
        # Create slightly darker border
        border_lightness = max(0, lightness - 15.0)
        
        return {
            'background': f"hsl({hue:.1f}, {saturation:.1f}%, {lightness:.1f}%)",
            'border': f"hsl({hue:.1f}, {saturation:.1f}%, {border_lightness:.1f}%)"
        }
    else:
        # Continuous RGB gradient - using floating point for precise calculation
        # This creates a smooth transition with unique colors for every 0.001 change
        if score < 0.5:
            # Red (255,0,0) to Yellow (255,255,0)
            red = 255.0
            green = 255.0 * (score * 2.0)  # 0 → 255 as score goes from 0 → 0.5
            blue = 0.0
        else:
            # Yellow (255,255,0) to Green (0,255,0)
            red = 255.0 * (1.0 - (score - 0.5) * 2.0)  # 255 → 0 as score goes from 0.5 → 1.0
            green = 255.0
            blue = 0.0
            
        # Convert to integer for hex formatting, but only at the very end
        # to maintain maximum precision during calculation
        red_int = max(0, min(255, int(round(red))))
        green_int = max(0, min(255, int(round(green))))
        blue_int = max(0, min(255, int(round(blue))))
        
        # Create slightly darker border
        border_factor = 0.8  # 20% darker
        border_red = max(0, min(255, int(round(red * border_factor))))
        border_green = max(0, min(255, int(round(green * border_factor))))
        border_blue = max(0, min(255, int(round(blue * border_factor))))
        
        return {
            'background': f"#{red_int:02x}{green_int:02x}{blue_int:02x}",
            'border': f"#{border_red:02x}{border_green:02x}{border_blue:02x}"
        }


def _decode_token_for_display(token_id: Any, 
                              tokenizer: Any, 
                              max_length: int = 15) -> str:
    """Decode a token ID to a displayable string.
    
    Args:
        token_id: The token ID to decode.
        tokenizer: The tokenizer to use for decoding.
        max_length: Maximum length of the displayed string.
        
    Returns:
        A string representation of the token, truncated if necessary.
    """
    if token_id == "ROOT":
        return "ROOT"
    
    if token_id is None:
        return "???"
    
    try:
        # Attempt to decode the token
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        
        # Replace whitespace with visible markers
        safe_decoded = decoded.replace('\n', '<NL>')
        safe_decoded = safe_decoded.replace(' ', '<SP>')
        safe_decoded = safe_decoded.replace('\t', '<TAB>')
        
        # Truncate if needed
        if len(safe_decoded) > max_length:
            safe_decoded = safe_decoded[:max_length-3] + '...'
            
        return safe_decoded
    except Exception:
        # Return token ID if decoding fails
        return f"ID:{token_id}"


def plot_tree_pdf(tree: CompletionTree, 
                  tokenizer: Any, 
                  output_filename: str,
                  view: bool = False,
                  fail_silently: bool = False) -> bool:
    """Plot a completion tree as a PDF using Graphviz.
    
    The visualization uses a retro-futuristic art deco style.
    
    Args:
        tree: The CompletionTree to visualize.
        tokenizer: Tokenizer to decode token IDs to text.
        output_filename: Path to save the PDF (with or without .pdf extension).
        view: Whether to open the resulting PDF.
        fail_silently: If True, return False on error instead of raising exception.
        
    Returns:
        Boolean indicating success (True) or failure (False). Only returns False
        if fail_silently is True and there was an error.
        
    Raises:
        ImportError: If Graphviz is not installed and fail_silently is False.
        RuntimeError: If PDF generation fails and fail_silently is False.
    """
    graphviz_error_message = """
ERROR: Graphviz is required for PDF plotting but is not properly installed.

To install Graphviz:
- Ubuntu/Debian: `sudo apt-get install graphviz`
- macOS: `brew install graphviz`
- Windows: Download installer from https://graphviz.org/download/

You also need the Python package: `pip install graphviz`

After installing, ensure the 'dot' command is in your PATH by running:
`dot -V`
"""
    
    # Check if graphviz Python package is installed
    if not HAS_GRAPHVIZ:
        if fail_silently:
            print("Warning: Graphviz Python package not installed. Skipping PDF generation.")
            return False
        else:
            raise ImportError(graphviz_error_message)
    
    # Check if graphviz command-line tool is installed
    import shutil
    if shutil.which('dot') is None:
        if fail_silently:
            print("Warning: Graphviz 'dot' command not found in PATH. Skipping PDF generation.")
            return False
        else:
            raise RuntimeError(graphviz_error_message)
    
    # Ensure output filename has .pdf extension
    if not output_filename.lower().endswith('.pdf'):
        pdf_filename = f"{output_filename}.pdf"
    else:
        pdf_filename = output_filename
    
    # Create Graphviz graph with art deco style settings
    graph = graphviz.Digraph(comment='Completion Tree', format='pdf')
    graph.attr(
        size="7.5,10",           # Standard letter size
        ratio="compress",        # Compress to fit
        bgcolor="#FFFFF0",       # Ivory background (art deco)
        fontname="Helvetica",    # Clean, modern font
        rankdir="TB",            # Top to bottom layout
        ranksep="0.5",           # Compact vertical spacing
        nodesep="0.3",           # Compact horizontal spacing
        overlap="false",         # Prevent overlap
        splines="ortho",         # Orthogonal edges for cleaner look
        page="8.5,11",           # US letter size
        dpi="300"                # Good resolution without bloating size
    )
    
    # Art deco framing and title
    graph.attr("graph", margin="0.2")
    
    # Track nodes and edges we've already processed
    canonical_nodes: Dict[int, TreeNode] = {}  # hash -> canonical node
    processed_nodes: Set[int] = set()  # unique_ids of processed nodes
    drawn_nodes: Set[int] = set()  # unique_ids of nodes already drawn
    
    # Start BFS traversal from root
    nodes_to_process = [tree.root]
    
    while nodes_to_process:
        current_node = nodes_to_process.pop(0)
        if current_node.unique_id in processed_nodes:
            continue
        processed_nodes.add(current_node.unique_id)
        
        # Find or assign canonical node for this hash
        current_hash = current_node.structural_hash
        if current_hash not in canonical_nodes:
            canonical_nodes[current_hash] = current_node
        
        canonical_node = canonical_nodes[current_hash]
        canonical_gv_id = f"node_{canonical_node.unique_id}"
        
        # Draw the canonical node only once
        if canonical_node.unique_id not in drawn_nodes:
            drawn_nodes.add(canonical_node.unique_id)
            
            # Get decoded token text for label
            token_text = _decode_token_for_display(
                canonical_node.token_id, tokenizer, max_length=20
            )
            
            # Prepare statistics for label
            node_count = canonical_node.count
            stats_text = f"\\nN:{node_count}"
            
            # Add leaf statistics if available
            leaf_count = canonical_node.descendant_leaf_count
            if leaf_count > 0:
                stats_text += f"\\nL:{leaf_count}"
                
                # Add score stats if scores are available
                if tree.has_scores:
                    score_pct = tree.get_node_score_percentage(canonical_node)
                    if score_pct is not None:
                        # Calculate correct and incorrect counts
                        correct_count = int(round(score_pct * leaf_count))
                        incorrect_count = leaf_count - correct_count
                        # Add stats with checkmark (✓) and X mark (✗)
                        stats_text += f" ({correct_count}✓/{incorrect_count}✗ {score_pct:.0%})"
            
            # Combine into final label
            label = f'"{token_text}"\\nT:{canonical_node.token_id}{stats_text}'
            
            # Determine node attributes with art deco style
            color = _get_color_for_score(
                tree.get_node_score_percentage(canonical_node) if tree.has_scores else None,
                html_format=False
            )
            
            node_attrs = {
                'style': 'filled',
                'fillcolor': color['background'],
                'color': color['border'],
                'fontname': 'Helvetica',
                'fontsize': '10',
                'penwidth': '1.5',
                'margin': '0.1,0.05'  # More compact nodes
            }
            
            # Special styling for different node types
            is_leaf = canonical_node.is_end_of_completion
            is_root = canonical_node.token_id == "ROOT"
            
            if is_root:
                # Root node: octagon in darker color
                node_attrs.update({
                    'shape': 'octagon',
                    'fillcolor': '#2F4F4F',  # Dark slate
                    'fontcolor': 'white',
                    'color': 'black',
                    'penwidth': '2.0',
                    'fontsize': '12'
                })
            elif is_leaf: 
                # Leaf node: diamond shape
                node_attrs.update({
                    'shape': 'diamond',
                    'penwidth': '1.8'
                })
            else:
                # Inner nodes: hexagon, pentagon, etc. based on depth
                depth = len(canonical_node.token_id) % 3 if isinstance(canonical_node.token_id, str) else canonical_node.unique_id % 3
                if depth == 0:
                    node_attrs['shape'] = 'hexagon'
                elif depth == 1:
                    node_attrs['shape'] = 'pentagon'
                else:
                    node_attrs['shape'] = 'septagon'
            
            # Add node to graph
            graph.node(canonical_gv_id, label=label, **node_attrs)
            
        # Add edges to children
        parent_gv_id = canonical_gv_id
        
        for token_id, child_node in sorted(current_node.children.items()):
            child_hash = child_node.structural_hash
            if child_hash not in canonical_nodes:
                canonical_nodes[child_hash] = child_node
                
            child_canonical_node = canonical_nodes[child_hash]
            child_gv_id = f"node_{child_canonical_node.unique_id}"
            
            # Add edge with art deco styling
            edge_style = 'solid'
            if child_canonical_node.is_end_of_completion:
                # Special styling for edges to leaf nodes
                edge_style = 'bold'
            
            # Determine edge color based on child's score
            edge_color = '#888888'  # Default gray
            if tree.has_scores:
                score_pct = tree.get_node_score_percentage(child_canonical_node)
                if score_pct is not None:
                    # Use the same continuous color gradient for edges
                    if score_pct < 0.5:
                        # Red to Yellow
                        red = 255.0
                        green = 255.0 * (score_pct * 2.0)
                        blue = 0.0
                    else:
                        # Yellow to Green
                        red = 255.0 * (1.0 - (score_pct - 0.5) * 2.0)
                        green = 255.0
                        blue = 0.0
                    
                    # Darker version for edges - 60% brightness
                    edge_factor = 0.6
                    red_int = max(0, min(255, int(round(red * edge_factor))))
                    green_int = max(0, min(255, int(round(green * edge_factor))))
                    blue_int = max(0, min(255, int(round(blue))))
                    
                    edge_color = f"#{red_int:02x}{green_int:02x}{blue_int:02x}"
            
            graph.edge(
                parent_gv_id, 
                child_gv_id,
                style=edge_style,
                color=edge_color,
                penwidth='1.2',
                arrowsize='0.8'
            )
            
            # Add child to processing queue
            if child_node.unique_id not in processed_nodes:
                nodes_to_process.append(child_node)
    
    # Render the graph
    try:
        # Specify the output file directly to avoid any extension issues
        rendered_file = graph.render(outfile=pdf_filename, view=view, cleanup=True)
        print(f"PDF visualization saved to {rendered_file}")
        return True
                
    except Exception as e:
        error_message = f"Error generating PDF: {e}\n{graphviz_error_message}"
        if fail_silently:
            print(error_message)
            return False
        else:
            raise RuntimeError(error_message)


def plot_tree_html(tree: CompletionTree, 
                   tokenizer: Any, 
                   output_filename: str) -> None:
    """Plot a completion tree as an interactive HTML file using vis.js.
    
    Args:
        tree: The CompletionTree to visualize.
        tokenizer: Tokenizer to decode token IDs to text.
        output_filename: Path to save the HTML file.
    """
    # Collect nodes and edges for vis.js
    vis_nodes = []
    vis_edges = []
    
    # Track nodes we've processed
    canonical_nodes: Dict[int, TreeNode] = {}  # hash -> canonical node
    hash_to_total_count: Dict[int, int] = {}  # hash -> aggregated path count
    processed_nodes: Set[int] = set()  # unique_ids of processed nodes
    drawn_nodes: Set[int] = set()  # canonical unique_ids already in vis_nodes
    
    # Start BFS traversal from root
    nodes_to_process = [tree.root]
    
    # First pass: aggregate counts for identical structures
    while nodes_to_process:
        current_node = nodes_to_process.pop(0)
        if current_node.unique_id in processed_nodes:
            continue
        processed_nodes.add(current_node.unique_id)
        
        current_hash = current_node.structural_hash
        
        # Aggregate path count for this hash
        hash_to_total_count[current_hash] = hash_to_total_count.get(current_hash, 0) + current_node.count
        
        # Assign canonical node if first time seeing this hash
        if current_hash not in canonical_nodes:
            canonical_nodes[current_hash] = current_node
        
        # Process children
        for child_node in current_node.children.values():
            if child_node.unique_id not in processed_nodes:
                nodes_to_process.append(child_node)
    
    # Reset for second pass to build the vis.js graph
    processed_nodes.clear()
    nodes_to_process = [tree.root]
    
    # Second pass: build vis.js nodes and edges
    while nodes_to_process:
        current_node = nodes_to_process.pop(0)
        if current_node.unique_id in processed_nodes:
            continue
        processed_nodes.add(current_node.unique_id)
        
        current_hash = current_node.structural_hash
        canonical_node = canonical_nodes[current_hash]
        canonical_unique_id = canonical_node.unique_id
        
        # Draw the canonical node only once
        if canonical_unique_id not in drawn_nodes:
            drawn_nodes.add(canonical_unique_id)
            
            # Get token text for label
            token_text = _decode_token_for_display(
                canonical_node.token_id, tokenizer, max_length=15
            )
            
            # Escape for HTML
            token_text = html.escape(token_text)
            
            # Path count for this node
            total_path_count = hash_to_total_count[current_hash]
            
            # Basic node label
            label = f'{token_text}\nN:{total_path_count}'
            
            # Prepare tooltip text
            tooltip_text = f"Token: '{token_text}'\nID: {canonical_node.token_id}"
            tooltip_text += f"\nTotal Paths: {total_path_count}"
            
            # Add leaf statistics
            total_leaves = canonical_node.descendant_leaf_count
            
            if tree.has_scores and total_leaves > 0:
                score_pct = tree.get_node_score_percentage(canonical_node)
                if score_pct is not None:
                    # Calculate correct and incorrect counts
                    correct_count = int(round(score_pct * total_leaves))
                    incorrect_count = total_leaves - correct_count
                    tooltip_text += f"\nDescendant Leaves: {total_leaves} ({correct_count}✓/{incorrect_count}✗ {score_pct:.1%})"
            else:
                tooltip_text += f"\nDescendant Leaves: {total_leaves}"
            
            # Determine node color
            node_color = _get_color_for_score(
                tree.get_node_score_percentage(canonical_node) if tree.has_scores else None,
                html_format=True
            )
            
            # Create node configuration
            node_config = {
                'id': canonical_unique_id,
                'label': label,
                'title': tooltip_text,  # Tooltip on hover
                'shape': 'ellipse',
                'color': node_color,
                'value': total_path_count,  # Size based on path count
                'font': {'size': 10},
                # Store leaf stats for JS click handler
                'leafStats': {
                    'total': total_leaves,
                    'correct': correct_count if tree.has_scores and score_pct is not None else None,
                    'incorrect': incorrect_count if tree.has_scores and score_pct is not None else None,
                    'score': tree.get_node_score_percentage(canonical_node) if tree.has_scores else None
                }
            }
            
            # Special styling for root and leaf nodes
            if canonical_node.token_id == "ROOT":
                node_config['shape'] = 'circle'
                node_config['color'] = {'background': 'lightgray', 'border': 'black'}
            elif canonical_node.is_end_of_completion:
                node_config['shape'] = 'box'
            
            vis_nodes.append(node_config)
        
        # Add edges to children
        parent_canonical_id = canonical_unique_id
        
        for token_id, child_node in sorted(current_node.children.items()):
            child_hash = child_node.structural_hash
            child_canonical_node = canonical_nodes[child_hash]
            child_canonical_id = child_canonical_node.unique_id
            
            # Add edge
            vis_edges.append({
                'from': parent_canonical_id,
                'to': child_canonical_id,
                'arrows': 'to',
                'color': {'color': '#cccccc', 'highlight': '#888888'},
                'smooth': {'type': 'cubicBezier', 'forceDirection': 'vertical', 'roundness': 0.4}
            })
            
            # Add child to processing queue
            if child_node.unique_id not in processed_nodes:
                nodes_to_process.append(child_node)
    
    # Create the HTML content
    nodes_json = json.dumps(vis_nodes, indent=None)
    edges_json = json.dumps(vis_edges, indent=None)
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Completion Tree Visualization</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        #mynetwork {{
            width: 100%;
            height: 90vh;
            border: 1px solid lightgray;
            background-color: #f8f8f8;
        }}
        body, html {{
            margin: 0;
            padding: 0;
            overflow: hidden;
            font-family: sans-serif;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }}
        #node-info {{
            padding: 10px;
            border-top: 1px solid lightgray;
            background-color: #eee;
            font-size: 0.9em;
            height: 10vh;
            overflow-y: auto;
        }}
        #node-info strong {{
            margin-right: 5px;
        }}
        #node-info span {{
            margin-right: 15px;
        }}
    </style>
</head>
<body>

<div id="mynetwork"></div>
<div id="node-info">Click a node to see statistics.</div>

<script type="text/javascript">
    // Create data structures for nodes and edges
    var nodes = new vis.DataSet({nodes_json});
    var edges = new vis.DataSet({edges_json});

    // Create the network
    var container = document.getElementById('mynetwork');
    var data = {{
        nodes: nodes,
        edges: edges
    }};
    var options = {{
        layout: {{
            hierarchical: {{
                enabled: true,
                levelSeparation: 150,
                nodeSpacing: 100,
                treeSpacing: 200,
                blockShifting: true,
                edgeMinimization: true,
                parentCentralization: true,
                direction: 'UD',
                sortMethod: 'directed'
            }}
        }},
        interaction: {{
            dragNodes: true,
            dragView: true,
            hover: true,
            zoomView: true,
            tooltipDelay: 200
        }},
        physics: {{
            enabled: false
        }},
        nodes: {{
            shape: 'dot',
            size: 16,
            font: {{
                size: 10,
                color: '#333'
            }},
            borderWidth: 1.5
        }},
        edges: {{
            width: 1,
            color: {{ inherit: 'both' }},
            smooth: {{
                enabled: true,
                type: "cubicBezier",
                forceDirection: 'vertical',
                roundness: 0.4
            }}
        }},
        scaling: {{
            min: 10,
            max: 50,
            label: {{
                enabled: true,
                min: 8,
                max: 20
            }}
        }}
    }};
    var network = new vis.Network(container, data, options);

    var nodeInfoDiv = document.getElementById('node-info');

    // Add click handler
    network.on("click", function (params) {{
        if (params.nodes.length > 0) {{
            var nodeId = params.nodes[0];
            var nodeData = nodes.get(nodeId);
            var stats = nodeData.leafStats;
            var output = "<strong>Selected Node:</strong> " + (nodeData.label.split('\\n')[0] || nodeData.id) + " (ID: " + nodeData.id + ")";

            if (stats) {{
                output += "<br><strong>Descendant Leaves:</strong> ";
                output += "<span>Total: " + stats.total + "</span> ";
                
                if (stats.correct !== null && stats.incorrect !== null && stats.score !== null) {{
                    var scoreText = (stats.score * 100).toFixed(1) + '%';
                    output += "<span>" + stats.correct + " ✓ / " + stats.incorrect + " ✗ (" + scoreText + ")</span>";
                }}
            }} else {{
                output += "<br>No statistics available for this node.";
            }}
            nodeInfoDiv.innerHTML = output;
        }} else {{
            nodeInfoDiv.innerHTML = "Click a node to see statistics.";
        }}
    }});

    // Disable physics after initial layout
    network.on("stabilizationIterationsDone", function () {{
        network.setOptions({{ physics: false }});
    }});
</script>

</body>
</html>"""
    
    # Write the HTML file
    try:
        with open(output_filename, 'w') as f:
            f.write(html_content)
        print(f"HTML visualization saved to {output_filename}")
    except Exception as e:
        print(f"Error writing HTML file: {e}") 