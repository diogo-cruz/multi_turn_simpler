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

## Installation

```bash
# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For figure generation and notebooks, also install:
pip install matplotlib seaborn pandas numpy scipy jupyter
```

## Reproducing Paper Figures

Below are instructions for reproducing each figure from the paper. All scripts assume you have the required data files (either raw JSONL files or the preprocessed CSV).

### Figure 2: Single-Turn vs Multi-Turn Attack Success Rates

This figure compares attack success rates between single-turn and multi-turn approaches.

```bash
cd figure_generation
python3 create_custom_asr_figure_batch7.py

# Output: custom_asr_figure_batch7.pdf
```

**Note**: This script requires batch7 JSONL data files in `clean_results/final_runs/batch7/`.

### Figures 3 & 10: Attack Success vs Number of Samples/Turns

These figures show how attack success increases with additional turns or resampling attempts.

```bash
# Start Jupyter notebook
jupyter notebook notebooks/asr_rounds_samples_comparison.ipynb

# In the notebook's control variables cell, set:
BATCH_NAME = "both"  # Uses both batch6A and batch6B data
INCLUDE_COMMAND = False  # Only includes direct_request tactic

# Run all cells in the notebook

# The notebook generates PNG files:
# - batch6a_6b_asr_rounds_samples_original_direct_only.png
# - batch6a_6b_asr_averaged_original_direct_only.png
# - And variants with different refusal handling methods

# Convert PNG to PDF if needed:
# convert batch6a_6b_asr_rounds_samples_original_direct_only.png figure3.pdf
```

### Figure 4: Reasoning Model Analysis (Stacked Bar Plot)

This figure shows the relationship between reasoning tokens and attack success.

```bash
cd figure_generation
python3 create_stacked_bar_plot.py

# Output: strongreject_vs_reasoning_tokens_stacked_bars.pdf
```

### Correlation Matrices (Appendix)

These figures show attack success correlation across different models.

```bash
cd figure_generation
python3 batch7_correlation_analysis.py

# Outputs:
# - batch7_correlation_llama31_8b.pdf
# - batch7_correlation_llama32_3b.pdf
# - batch7_correlation_meta_gpt4omini_llama313_70b.pdf
# - batch7_single_turn_correlation_matrix.csv
# - batch7_multi_turn_correlation_matrix.csv
```

### Refusal vs Scratch Sampling Analysis

This figure compares sampling behavior after refusals vs from scratch.

```bash
cd figure_generation
python3 create_combined_refusal_plot.py

# Output: refusal_vs_scratch_combined_plot.pdf

# Alternative: Use the notebook for detailed histogram analysis
jupyter notebook notebooks/refusal_vs_scratch_sampling_analysis.ipynb
```

### Turn Type Comparisons

Publication-quality comparison plots between different turn types.

```bash
cd figure_generation
python3 publication_quality_plots_by_turns.py

# Output: publication_quality_turn_type_comparisons.pdf
```

### Additional Reasoning Analysis

For deeper reasoning token analysis:

```bash
cd figure_generation
python3 reasoning_token_analysis.py

# This generates various analysis files about reasoning tokens
```

## Additional Analysis Notebooks

### ASR vs Samples Analysis

This notebook provides theoretical analysis of expected maximum scores:

```bash
jupyter notebook notebooks/asr_samples_direct_request_analysis.ipynb
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
- `--turn-type`: "single" or "multi" (default: "multi")
- `--n-samples`: Number of attack attempts

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
