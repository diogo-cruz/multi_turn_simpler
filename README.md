# Multi-Turn Jailbreaks Are Simpler Than They Seem

This repository contains the code and data for our paper "Multi-Turn Jailbreaks Are Simpler Than They Seem", accepted at the SoLaR (Society, Language, and Reasoning) workshop at COLM 2025.

## Abstract

While defenses against single-turn jailbreak attacks on Large Language Models (LLMs) have improved significantly, multi-turn jailbreaks remain a persistent vulnerability, often achieving success rates exceeding 70% against models optimized for single-turn protection. This work presents an empirical analysis of automated multi-turn jailbreak attacks across state-of-the-art models including GPT-4, Claude, and Gemini variants, using the StrongREJECT benchmark. Our findings challenge the perceived sophistication of multi-turn attacks: when accounting for the attacker's ability to learn from how models refuse harmful requests, multi-turn jailbreaking approaches are approximately equivalent to simply resampling single-turn attacks multiple times. Moreover, attack success is correlated among similar models, making it easier to jailbreak newly released ones. Additionally, for reasoning models, we find surprisingly that higher reasoning effort often leads to higher attack success rates.

## Repository Structure

```
├── main.py                 # Main entry point for running attacks
├── utils/                  # Utility functions for generation and evaluation
├── jailbreaks/            # Jailbreak tactics implementation
│   └── direct_request/    # Direct Request tactic (main focus)
├── test_cases/            # 30 harmful behaviors from StrongREJECT
├── csv_results/           # Experimental results
│   └── results_strongreject.csv  # Main results file
├── figure_generation/     # Python scripts that generate paper figures
│   ├── create_custom_asr_figure_batch7.py    # Figure 2
│   ├── batch7_correlation_analysis.py        # Correlation matrices
│   ├── create_stacked_bar_plot.py           # Figure 4
│   ├── create_combined_refusal_plot.py      # Refusal analysis
│   ├── publication_quality_plots_by_turns.py # Turn comparisons
│   └── reasoning_token_analysis.py           # Reasoning analysis
├── notebooks/             # Jupyter notebooks for figure generation
│   ├── asr_rounds_samples_comparison.ipynb   # Figures 3 & 10
│   ├── refusal_vs_scratch_sampling_analysis.ipynb
│   └── asr_samples_direct_request_analysis.ipynb
└── asr_analysis_utils.py  # Utility functions for ASR analysis
```

## Quick Setup

```bash
# Create virtual environment and install dependencies
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set up API key (only needed for new experiments)
cp .env.example .env  # Edit with your OpenRouter API key
```

## Reproducing Paper Figures

Below are instructions for reproducing each figure from the paper. Most scripts can work in two modes: using our provided CSV files (recommended for reproducing paper figures) or processing your own raw JSONL data from experiments (keep local for safety).

### Figure 2: Single-Turn vs Multi-Turn Attack Success Rates

This figure compares attack success rates between single-turn and multi-turn approaches.

**The script supports two modes:**

This repository provides systematic testing code for multi-turn jailbreaks, allowing researchers to run their own experiments and generate JSONL files with conversation logs. However, for safety reasons, raw experiment data (JSONL files) should be kept local only. We provide our processed statistics (CSV files) so others can reproduce our figures without needing raw data. The `clean_results/` folder is already in `.gitignore` to prevent accidental sharing of sensitive data.

**Mode 1: CSV → Figures**
```bash
python3 figure_generation/create_custom_asr_figure_batch7.py --mode csv

# Outputs: 
# - result_figures/custom_asr_figure_batch7.pdf
# - result_figures/custom_asr_figure_batch7.png
```
This mode uses the pre-processed CSV data file (`csv_results/asr_three_scenarios_batch7_data.csv`) and is the recommended way to generate figures. Use this mode to reproduce our paper's figures.

**Mode 2: Raw JSONL → CSV + Figures (For Your Own Experiments)**
```bash
python3 figure_generation/create_custom_asr_figure_batch7.py --mode raw

# Outputs: 
# - result_figures/custom_asr_figure_batch7.pdf
# - result_figures/custom_asr_figure_batch7.png
# - csv_results/asr_three_scenarios_batch7_data.csv
```
This mode processes raw JSONL data files from `clean_results/final_runs/batch7/` that you generate by running your own experiments. **Important**: Do not share the generated JSONL files - keep them local for safety reasons.

### Figures 3 & 10: Attack Success vs Number of Samples/Turns

These figures show how attack success increases with additional turns or resampling attempts.

**The notebook supports two modes:**

**Mode 1: CSV → Figures (Public/Recommended)**
```bash
cd notebooks
jupyter notebook asr_rounds_samples_comparison.ipynb
# Set load_from_csv = True in the CSV cell, then run visualization cells
```
Uses pre-processed CSV data files from `csv_results/` directory.

**Mode 2: Raw JSONL → CSV + Figures (For Your Own Experiments)**  
```bash
cd notebooks
jupyter notebook asr_rounds_samples_comparison.ipynb
# Run all cells to process raw data and generate figures
```
Processes raw JSONL data from `clean_results/final_runs/batch6A` and `batch6B`. **Important**: Keep raw data local for safety.

**Outputs:**
- **Figures**: `result_figures/batch6a_6b_asr_*_direct_only.pdf` and `.png` files
- **Data**: `csv_results/asr_*_results_batch6a_6b_direct_only.csv` files

### Figure 4: Reasoning Model Analysis (Stacked Bar Plot)

This figure shows the relationship between reasoning tokens and attack success.

**Mode 1 (CSV - recommended):**
```bash
python3 figure_generation/create_stacked_bar_plot.py --mode csv

# Output: result_figures/strongreject_vs_reasoning_tokens_stacked_bars.pdf
```

**Mode 2 (Raw JSONL - for processing new data):**
```bash
python3 figure_generation/create_stacked_bar_plot.py --mode raw

# ⚠️  WARNING: Processes sensitive conversation data!
# Generates: csv_results/stacked_bar_plot_data.csv
# Output: result_figures/strongreject_vs_reasoning_tokens_stacked_bars.pdf
```

### Correlation Matrices (Appendix)

These figures show attack success correlation across different models.

```bash
python3 figure_generation/batch7_correlation_analysis.py

# Outputs:
# - batch7_correlation_llama31_8b.pdf
# - batch7_correlation_llama32_3b.pdf
# - batch7_correlation_meta_gpt4omini_llama313_70b.pdf
# - batch7_single_turn_correlation_matrix.csv
# - batch7_multi_turn_correlation_matrix.csv
```

### Refusal vs Scratch Sampling Analysis

This figure compares sampling behavior after refusals vs from scratch.

**Use the notebook for detailed histogram analysis:**
cd notebooks
jupyter notebook refusal_vs_scratch_sampling_analysis.ipynb
```

### Turn Type Comparisons

Publication-quality comparison plots between different turn types.

```bash
python3 figure_generation/publication_quality_plots_by_turns.py

# Output: publication_quality_turn_type_comparisons.pdf
```

### Additional Reasoning Analysis

For deeper reasoning token analysis and correlation studies:

**Mode 1 (CSV - recommended):**
```bash
python3 figure_generation/reasoning_token_analysis.py --mode csv

# Outputs:
# - result_figures/reasoning_tokens_vs_success_scatter.png
# - result_figures/reasoning_level_analysis.png
# - result_figures/model_reasoning_heatmap.png
# - result_figures/reasoning_tokens_distribution.png
# - result_figures/reasoning_analysis_report.md
```

**Mode 2 (Raw JSONL - for processing new data):**
```bash
python3 figure_generation/reasoning_token_analysis.py --mode raw

# ⚠️  WARNING: Processes sensitive conversation data!
# Generates: csv_results/reasoning_token_analysis_data.csv
# Then creates all analysis outputs above
```

## Additional Analysis Notebooks

### ASR vs Samples Analysis

This notebook provides theoretical analysis of expected maximum scores:

```bash
cd notebooks
jupyter notebook asr_samples_direct_request_analysis.ipynb
```

## Data Requirements

Most figure generation scripts expect data in specific locations:
- JSONL files in `clean_results/final_runs/batch*/`
- CSV results in `csv_results/`

If you don't have the raw JSONL data, some scripts can work with the provided CSV file in `csv_results/results_strongreject.csv`.

## Running New Experiments

### Single Attack Example
```bash
python main.py \
  --jailbreak-tactic "direct_request" \
  --test-case "counterfeit_money" \
  --target-model "gpt-4" \
  --attacker-model "gpt-4o-mini"
```

### Parameters
- `--jailbreak-tactic`: Attack strategy (use "direct_request")
- `--test-case`: Harmful behavior from test_cases/
- `--target-model`: Model to attack
- `--attacker-model`: Model generating attacks
- `--turn-type`: "single_turn" or "multi_turn" (default: "multi_turn")
- `--samples`: Number of attack attempts

### Output Location

Attack results are automatically saved to:
```
./clean_results/final_runs/batch{ID}/{jailbreak_tactic}/
```

**File naming pattern:**
```
{jailbreak_tactic}_{test_case}_{target_model}_{turn_type}_sample{N}_{timestamp}.jsonl
```

Each JSONL file contains:
- Full conversation logs (user prompts and assistant responses)
- Attack success scores and evaluation rationale
- Token usage statistics for cost analysis
- Metadata including timestamps and model configurations

### API Keys
Set up your API keys in a `.env` file:
```bash
cp .env.example .env
# Edit .env with your OpenRouter API key
```

## Key Findings

1. **Multi-turn attacks are more effective than single-turn attacks** across all tested models
2. **The advantage of multi-turn attacks can be largely explained by additional sampling opportunities** rather than sophisticated conversational strategies
3. **Attack success correlates among similar models**, suggesting transferability of vulnerabilities
4. **Reasoning models show higher vulnerability** when using more reasoning tokens

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{yang2025multiturn,
  title={Multi-Turn Jailbreaks Are Simpler Than They Seem},
  author={[Authors]},
  booktitle={SoLaR Workshop at COLM 2025},
  year={2025},
  url={https://solar-colm.github.io/}
}
```

## Ethical Considerations

This tool is intended for research and educational purposes only. It should be used responsibly to improve AI safety, not to cause harm. Please use in accordance with applicable laws and regulations.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This work was conducted as part of the AI Safety Camp. We thank the SoLaR workshop organizers and the COLM community.
