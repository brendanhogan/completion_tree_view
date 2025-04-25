"""
Example script demonstrating the use of CompletionTreeView with a math problem.

This script shows how to:
1. Generate token completions for a simple math problem
2. Create a CompletionTree from these completions
3. Visualize the tree as both PDF and HTML
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

# Add parent directory to path to import from completion_tree_view
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from completion_tree_view import CompletionTree, plot_tree_pdf, plot_tree_html

def main():
    # Define a grade school math problem
    problem = "Ravi can jump higher than anyone in the class. In fact, he can jump 1.5 times higher than the average jump of the three next highest jumpers. If the three next highest jumpers can jump 23 inches, 27 inches, and 28 inches, how high can Ravi jump?"
    
    # Load Qwen model for demonstration
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # Use Qwen 7B model
    print(f"Loading model: {model_name}...")
    
    # Option 1: Load from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    
    # Option 2 (commented out): Load from local checkpoint
    # local_path = "/path/to/your/local/qwen2.5-7b-instruct"
    # tokenizer = AutoTokenizer.from_pretrained(local_path)
    # model = AutoModelForCausalLM.from_pretrained(local_path, torch_dtype=torch.float16, device_map="auto")
    
    # Define how many completions to generate
    num_completions = 20
    
    # Settings for generation
    temperature = 0.8
    max_tokens = 1200
    
    # Format prompt with system message and required output format
    system_prompt = """You will be given a question that involves reasoning. You should reason carefully about the question, then provide your answer.
            It is very important that you put your reasoning process inside <reasoning> tags and your final answer inside <answer> tags, like this:

            
            <reasoning>
            Your step-by-step reasoning process here
            </reasoning>
            <answer>
            Your final answer here
            </answer>

            All of your returned text should either be in the <reasoning> or <answer> tags - no text outside! Start each answer by immediately starting with <reasoning>. 
            It is is extremely important you answer in this way - do not put any information or text outside of these tags!

            In the answer tags - you should only put the final numerical answer, no other text or information.
            
            Question: """
    
    chat_format = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem}
    ]
    
    # Format prompt with chat template
    prompt_text = tokenizer.apply_chat_template(chat_format, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(model.device)
    
    # Generate completions all at once using batch generation
    print(f"Generating {num_completions} completions in a single batch...")
    completions_tokens = []
    completions_text = []
    
    # Create a batch of identical inputs
    batch_input_ids = input_ids.repeat(num_completions, 1)
    
    # Set generation parameters
    generation_config = {
        "max_new_tokens": max_tokens,
        "do_sample": True,
        "temperature": temperature,
        "pad_token_id": tokenizer.eos_token_id,
        "num_return_sequences": 1  # One sequence per input
    }
    
    # Set different seeds for each example in the batch
    torch.manual_seed(42)  # Set a base seed
    # Each sequence in the batch will get different sampling due to the nature of batched generation
    
    # Generate all completions at once
    with torch.no_grad():
        outputs = model.generate(
            batch_input_ids,
            **generation_config
        )
    
    # Process the batch outputs
    prompt_length = input_ids.shape[1]
    
    for i in range(num_completions):
        # Extract completion tokens (skip the prompt tokens)
        new_tokens = outputs[i][prompt_length:]
        
        # Store tokens and decoded text
        completions_tokens.append(new_tokens.tolist())
        completion_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        completions_text.append(completion_text)
        
        # Print a preview of the completion
        preview = completion_text[:100] + "..." if len(completion_text) > 100 else completion_text
        print(f"[{i+1}/{num_completions}] Generated: {preview}")
    
    # Evaluate the correctness of each completion
    # Simple check: if "50" appears in the answer section
    scores = []
    for text in completions_text:
        # Try to extract answer section
        try:
            answer_part = text.split("<answer>")[1].split("</answer>")[0].strip()
            score = 1.0 if "39" in answer_part else 0.0
        except (IndexError, ValueError):
            score = 0.0
        scores.append(score)
    
    # Calculate correctness percentage
    correct_count = sum(1 for score in scores if score > 0.5)
    print(f"\nCorrect answers: {correct_count}/{num_completions} ({correct_count/num_completions:.1%})")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save completions to JSON file
    json_output = os.path.join(output_dir, "math_completions.json")
    completions_data = []
    
    for i in range(len(completions_text)):
        completions_data.append({
            "completion": completions_text[i],
            "tokens": completions_tokens[i],
            "is_correct": scores[i] > 0.5,
            "score": float(scores[i])  # Convert to float for JSON serialization
        })
    
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(completions_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {num_completions} completions to: {json_output}")
    
    # Generate visualizations
    print("\nBuilding CompletionTree...")
    tree = CompletionTree(completions_tokens, scores)
    
    # Try to create PDF visualization
    print("\nGenerating PDF visualization...")
    try:
        pdf_output = os.path.join(output_dir, "math_example.pdf")
        pdf_success = plot_tree_pdf(tree, tokenizer, pdf_output)
        if pdf_success:
            print(f"✅ PDF visualization created: {pdf_output}")
        else:
            print("❌ PDF generation failed. See error message above.")
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        print("   Continuing with HTML generation...")
    
    # Create HTML visualization
    print("\nGenerating HTML visualization...")
    try:
        html_output = os.path.join(output_dir, "math_example.html")
        plot_tree_html(tree, tokenizer, html_output)
        print(f"✅ HTML visualization created: {html_output}")
    except Exception as e:
        print(f"❌ Error generating HTML: {e}")
    
    print("\nVisualization process complete!")
    print("Note: For PDF generation to work, you must have Graphviz installed.")
    print("      See requirements in the README.md file.")

if __name__ == "__main__":
    main() 