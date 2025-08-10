#!/usr/bin/env python3
"""
Create stacked bar chart of StrongREJECT score vs reasoning tokens
- Group bars by reasoning token bins
- Each bar represents a model within the group
- Remove Qwen data
- Use matplotlib default colors
- Output as PDF

TWO-MODE SYSTEM:
- Mode 1 (CSV - default): Load from existing CSV file for figure generation
- Mode 2 (Raw): Process JSONL files, save to CSV, then load CSV for figures
  ⚠️  Raw mode processes sensitive conversation data - use with caution!
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
import argparse
import os

# Enable LaTeX rendering
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 24

def clean_model_name(raw_model_name):
    """Clean up model names for display"""
    if not raw_model_name or raw_model_name == 'unknown':
        return 'unknown'
    
    # Remove provider prefixes and extract main model name
    model_name = raw_model_name.lower()
    
    if 'claude' in model_name:
        return 'Claude'
    elif 'gemini' in model_name:
        return 'Gemini'
    elif 'gpt' in model_name or 'o1-mini' in model_name or 'o3-mini' in model_name or 'o4-mini' in model_name:
        return 'OpenAI'
    elif 'qwen' in model_name:
        return 'Qwen'
    else:
        # Return the part after the slash if it exists, otherwise the whole name
        if '/' in raw_model_name:
            return raw_model_name.split('/')[-1].title()
        return raw_model_name.title()

def load_and_process_data(data_dir="clean_results/final_runs/batch_thinking"):
    """Load data and compute scores correctly"""
    data_dir = Path(data_dir)
    single_turn_data = []
    multi_turn_data = []
    
    print("Loading data from", data_dir)
    
    # Find all JSONL files recursively
    jsonl_files = list(data_dir.glob("**/*.jsonl"))
    print("Found", len(jsonl_files), "JSONL files")
    
    for file_path in jsonl_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if len(lines) < 2:
                continue
                
            # Extract metadata from first line
            try:
                metadata_line = json.loads(lines[0].strip())
                file_metadata = {
                    'model': clean_model_name(metadata_line.get('target_model', 'unknown')),
                    'test_case': metadata_line.get('test_case', 'unknown'),
                    'turn_type': metadata_line.get('turn_type', 'unknown'),
                    'jailbreak_tactic': metadata_line.get('jailbreak_tactic', 'unknown'),
                }
            except json.JSONDecodeError as e:
                print(f"Error parsing metadata in {file_path}: {e}")
                continue
                
            # Process data lines (skip metadata line)
            data_lines = lines[1:]
            
            # Collect data for this JSONL file
            conversation_scores = []
            conversation_reasoning_tokens = []
            valid_data_found = False
            
            for line_num, line in enumerate(data_lines, 2):
                try:
                    data = json.loads(line.strip())
                    
                    # Only process if this matches our target tactics
                    jailbreak_tactic = data.get('jailbreak_tactic', file_metadata['jailbreak_tactic']).lower()
                    if jailbreak_tactic in ['direct_request', 'command']:
                        valid_data_found = True
                    
                    # Extract score - treat scores not between 0 and 1 as 0
                    if data.get('score') is not None:
                        try:
                            score = float(data['score'])
                            if not (0 <= score <= 1):
                                score = 0
                            conversation_scores.append(score)
                        except (ValueError, TypeError):
                            conversation_scores.append(0)
                    
                    # Extract reasoning tokens
                    if 'token_usage' in data and data['token_usage'].get('reasoning_tokens'):
                        reasoning_tokens = data['token_usage']['reasoning_tokens']
                        if reasoning_tokens > 0:
                            conversation_reasoning_tokens.append(reasoning_tokens)
                        
                except json.JSONDecodeError as e:
                    print(f"JSON error in {file_path}:{line_num} - {e}")
                    continue
                except Exception as e:
                    print(f"Error processing {file_path}:{line_num} - {e}")
                    continue
                    
            # Only include conversations with valid tactic and scores
            if valid_data_found and conversation_scores:
                # Maximum score across all rounds
                max_score = max(conversation_scores)
                
                # Average reasoning tokens (only counting non-zero tokens)
                avg_reasoning_tokens = np.mean(conversation_reasoning_tokens) if conversation_reasoning_tokens else 0
                
                conversation_record = {
                    'file': file_path.name,
                    'max_score': max_score,
                    'avg_reasoning_tokens': avg_reasoning_tokens,
                    'num_rounds': len(conversation_scores),
                    **file_metadata
                }
                
                # Separate by turn type
                turn_type = file_metadata.get('turn_type', 'unknown')
                if turn_type == 'single' or 'single_turn' in file_path.name:
                    single_turn_data.append(conversation_record)
                elif turn_type == 'multi' or 'multi_turn' in file_path.name:
                    multi_turn_data.append(conversation_record)
                    
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    print(f"Loaded {len(single_turn_data)} single-turn conversations")
    print(f"Loaded {len(multi_turn_data)} multi-turn conversations")
    
    return pd.DataFrame(single_turn_data), pd.DataFrame(multi_turn_data)

def create_reasoning_token_bins(df):
    """Create reasoning token bins for all models"""
    
    if len(df) == 0:
        return pd.DataFrame()
    
    # Use the same bins as the modified line plot
    bin_edges = [0, 200, 500, 1000, 1500, float('inf')]
    bin_labels = ['0-200', '200-500', '500-1000', '1000-1500', '1500+']
    
    # Create bins
    df = df.copy()
    df['reasoning_bin'] = pd.cut(df['avg_reasoning_tokens'], bins=bin_edges, labels=bin_labels, right=False)
    
    return df

def create_stacked_bar_plot(single_df, multi_df, output_dir="result_figures"):
    """Create the stacked bar chart with single/multi-turn segments"""
    
    # Remove Qwen and Gemini data, and exclude fake_online_profile test case
    single_df = single_df[(single_df['model'] != 'Qwen') & (single_df['model'] != 'Gemini') & (single_df['test_case'] != 'fake_online_profile')]
    multi_df = multi_df[(multi_df['model'] != 'Qwen') & (multi_df['model'] != 'Gemini') & (multi_df['test_case'] != 'fake_online_profile')]
    
    # Create reasoning token bins for both datasets
    single_df = create_reasoning_token_bins(single_df)
    multi_df = create_reasoning_token_bins(multi_df)
    
    # Remove rows with NaN bins
    single_df = single_df.dropna(subset=['reasoning_bin'])
    multi_df = multi_df.dropna(subset=['reasoning_bin'])
    
    # Get models with sufficient data (checking combined data)
    combined_df = pd.concat([single_df, multi_df], ignore_index=True)
    model_counts = combined_df['model'].value_counts()
    valid_models = model_counts[model_counts >= 10].index.tolist()
    
    single_df = single_df[single_df['model'].isin(valid_models)]
    multi_df = multi_df[multi_df['model'].isin(valid_models)]
    
    # Calculate average scores by model, reasoning bin, and turn type
    single_grouped = single_df.groupby(['reasoning_bin', 'model'])['max_score'].agg(['mean', 'std', 'count']).reset_index()
    single_grouped.columns = ['reasoning_bin', 'model', 'avg_score', 'std_score', 'count']
    single_grouped['turn_type'] = 'single'
    
    multi_grouped = multi_df.groupby(['reasoning_bin', 'model'])['max_score'].agg(['mean', 'std', 'count']).reset_index()
    multi_grouped.columns = ['reasoning_bin', 'model', 'avg_score', 'std_score', 'count']
    multi_grouped['turn_type'] = 'multi'
    
    # Pivot to get models as columns for each turn type
    single_pivot = single_grouped.pivot(index='reasoning_bin', columns='model', values='avg_score')
    multi_pivot = multi_grouped.pivot(index='reasoning_bin', columns='model', values='avg_score')
    
    # Fill NaN values with 0
    single_pivot = single_pivot.fillna(0)
    multi_pivot = multi_pivot.fillna(0)
    
    # Define the order of reasoning bins
    bin_order = ['0-200', '200-500', '500-1000', '1000-1500', '1500+']
    single_pivot = single_pivot.reindex(bin_order, fill_value=0)
    multi_pivot = multi_pivot.reindex(bin_order, fill_value=0)
    
    # Get models and assign colors
    models = list(set(single_pivot.columns.tolist() + multi_pivot.columns.tolist()))
    models.sort()  # For consistent ordering
    colors = [f'C{i}' for i in range(len(models))]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set up bar positions
    x = np.arange(len(bin_order))
    bar_width = 0.8 / len(models)
    
    # Create stacked bars for each model
    for i, (model, color) in enumerate(zip(models, colors)):
        x_pos = x + (i - len(models)/2 + 0.5) * bar_width
        
        # Get single and multi values for this model
        single_values = single_pivot[model].values if model in single_pivot.columns else np.zeros(len(bin_order))
        multi_values = multi_pivot[model].values if model in multi_pivot.columns else np.zeros(len(bin_order))
        
        # Create overlapping bars
        ax.bar(x_pos, single_values, bar_width, 
               label=f'{model} (single)' if i == 0 else None, 
               color=color, alpha=0.8)
        ax.bar(x_pos, multi_values, bar_width, 
               label=f'{model} (multi)' if i == 0 else None,
               color=color, alpha=0.3)
    
    # Customize the plot
    ax.set_xlabel('Reasoning Tokens')
    ax.set_ylabel('StrongREJECT score')
    ax.set_ylim(0, 1)  # Set y-axis range from 0 to 1
    ax.set_xticks(x)
    ax.set_xticklabels(bin_order)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Create custom legend (models only)
    legend_handles = []
    
    # Add model color entries only
    for model, color in zip(models, colors):
        legend_handles.append(plt.Rectangle((0,0),1,1, color=color, alpha=0.8, label=model))
    
    ax.legend(handles=legend_handles, loc='upper left', fontsize=20)
    
    plt.tight_layout()
    output_path = f'{output_dir}/strongreject_vs_reasoning_tokens_stacked_bars.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print some statistics
    print("\nData summary:")
    print(f"Models included: {models}")
    print(f"Reasoning token bins: {bin_order}")
    print("\nSingle-turn scores by model and bin:")
    print(single_pivot.round(3))
    print("\nMulti-turn scores by model and bin:")
    print(multi_pivot.round(3))

def save_to_csv(single_df, multi_df, csv_path="csv_results/stacked_bar_plot_data.csv"):
    """Save processed data to CSV file"""
    # Combine single and multi dataframes
    combined_df = pd.concat([single_df, multi_df], ignore_index=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Save to CSV
    combined_df.to_csv(csv_path, index=False)
    print(f"✓ Data saved to {csv_path}")
    return csv_path

def load_from_csv(csv_path="csv_results/stacked_bar_plot_data.csv"):
    """Load data from CSV file and separate into single/multi turn"""
    if not os.path.exists(csv_path):
        print(f"✗ CSV file not found: {csv_path}")
        print("Please run with --mode raw to generate the CSV file first.")
        return pd.DataFrame(), pd.DataFrame()
    
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded data from {csv_path}")
    print(f"Total conversations: {len(df)}")
    
    # Separate by turn type
    single_df = df[df['turn_type'] == 'single'].copy()
    multi_df = df[df['turn_type'] == 'multi'].copy()
    
    print(f"Single-turn conversations: {len(single_df)}")
    print(f"Multi-turn conversations: {len(multi_df)}")
    
    return single_df, multi_df

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Create stacked bar chart of StrongREJECT score vs reasoning tokens')
    parser.add_argument('--mode', choices=['raw', 'csv'], default='csv',
                        help='Mode: "raw" to process JSONL to CSV, "csv" to generate figures from existing CSV (default: csv)')
    parser.add_argument('--output-dir', default='result_figures',
                        help='Output directory for figures (default: result_figures)')
    
    args = parser.parse_args()
    
    print("Creating stacked bar chart...")
    print(f"Mode: {args.mode}")
    
    if args.mode == 'raw':
        print("\n⚠️  WARNING: Raw mode processes sensitive conversation data!")
        print("⚠️  This mode should only be used in secure environments.")
        print("⚠️  The generated CSV will be safe for sharing.\n")
        
        # Process JSONL files
        single_df, multi_df = load_and_process_data()
        
        # Save to CSV
        csv_path = save_to_csv(single_df, multi_df)
        
        # Load back from CSV (ensures both modes use same data format)
        print("\nLoading processed data from CSV...")
        single_df, multi_df = load_from_csv(csv_path)
        
    else:  # csv mode
        # Load from existing CSV
        single_df, multi_df = load_from_csv()
        
        if len(single_df) == 0 and len(multi_df) == 0:
            return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create the plot
    create_stacked_bar_plot(single_df, multi_df, args.output_dir)
    
    print(f"\n✓ Plot saved to {args.output_dir}/strongreject_vs_reasoning_tokens_stacked_bars.pdf")

if __name__ == "__main__":
    main()