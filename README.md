# Multi-Turn Jailbreaks Are Simpler Than They Seem

This repository contains the code and data for our paper "Multi-Turn Jailbreaks Are Simpler Than They Seem", accepted at NeurIPS 2025 Workshop.

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
└── plot_scripts/          # Scripts for generating paper figures
```

## Installation

```bash
# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Reproducing Paper Figures

The main results used in our paper are stored in `csv_results/results_strongreject.csv`. To reproduce the figures from the paper, you'll need to install the plotting dependencies:

```bash
pip install pandas matplotlib seaborn numpy
```

### Figure 1: Pipeline Overview (figures/pipeline.png)
This is a conceptual diagram created outside of the codebase.

### Figure 2: Single-Turn vs Multi-Turn Attack Success Rates (figures/placeholder_single_vs_multi_7_2.pdf)
This figure shows the comparison between single-turn and multi-turn attack success rates across different models.

```bash
cd plot_scripts
python3 plot_main_graphs_from_csv.py --input ../csv_results/results_strongreject.csv --output_dir ../figures
```

### Figure 3: Attack Success vs Number of Samples/Turns (figures/placeholder_samples_turns.pdf)
This figure demonstrates how attack success increases with additional turns or resampling attempts.

The plotting script will generate this figure alongside Figure 2.

### Figure 4: Attack Success Rates for Reasoning Models (figures/placeholder_reasoning_bars.pdf)
This figure shows the relationship between reasoning token usage and attack success rates.

```bash
# This requires additional analysis of reasoning tokens from the experimental logs
# The script is included in the plot_scripts directory
python3 analyze_reasoning_tokens.py --input ../csv_results/results_strongreject.csv
```

### Additional Figures in Appendix

Several additional figures appear in the appendix:
- **figures/placeholder_refusal_vs_sampling.pdf**: Comparison of refusal-based resampling strategies
- **figures/o4_mini.pdf**: Detailed results for o4-mini model
- **figures/command.pdf**: Results for command-based jailbreak tactics
- **figures/command_test_cases.pdf**: Breakdown by test case for command tactic

These can be generated using the same plotting scripts with different parameters.

## Running New Experiments

### Single Attack Example
```bash
python main.py --jailbreak-tactic "direct_request" --test-case "counterfeit_money" --target-model "gpt-4" --attacker-model "gpt-4o-mini"
```

### Parameters
- `--jailbreak-tactic`: Attack strategy to use (e.g., "direct_request")
- `--test-case`: Harmful behavior to test (see test_cases/ directory)
- `--target-model`: Model to attack
- `--attacker-model`: Model generating attack prompts
- `--turn-type`: "single" or "multi" (default: "multi")
- `--n-samples`: Number of times to run the attack (default: 1)

### API Keys
You'll need to set up your API keys in a `.env` file:
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
@inproceedings{cruz2025multiturn,
  title={Multi-Turn Jailbreaks Are Simpler Than They Seem},
  author={Anonymous Authors},
  booktitle={NeurIPS 2025 Workshop},
  year={2025}
}
```

## Ethical Considerations

This tool is intended for research and educational purposes only. It should be used responsibly to improve AI safety, not to cause harm. Please use in accordance with applicable laws and regulations.

## License

MIT License - see LICENSE file for details.