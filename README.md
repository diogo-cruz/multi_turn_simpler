# Multi-Turn Jailbreaks: Exploring Conversational Attack Strategies

This repository contains the code and data for our paper on multi-turn jailbreak attacks against large language models, accepted at [Workshop Name].

## Overview

This work demonstrates that multi-turn conversational attacks are significantly more effective than single-turn attacks at bypassing AI safety guardrails. We focus on the "Direct Request" strategy, which uses natural-sounding dialogue to progressively lead models toward prohibited responses.

## Key Findings

- Multi-turn attacks achieve higher success rates than single-turn attacks across all tested models
- The Direct Request tactic, despite its simplicity, proves highly effective
- Models with reasoning capabilities show different vulnerability patterns

## Repository Structure

```
├── main.py                 # Main entry point for running attacks
├── utils/                  # Utility functions for generation and evaluation
├── jailbreaks/            # Jailbreak tactics implementation
│   └── direct_request/    # Direct Request tactic (main focus)
├── test_cases/            # 30 harmful behaviors from StrongREJECT
├── csv_results/           # Experimental results
└── plot_scripts/          # Scripts for generating paper figures
```

## Installation

```bash
pip install -r requirements.txt
```

## Running Experiments

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

## Reproducing Paper Results

The main results from our paper can be found in `csv_results/results_strongreject.csv`. To generate the figures:

```bash
cd plot_scripts
python plot_main_graphs_from_csv.py
```

## Test Cases

We evaluate on 30 harmful behaviors from the StrongREJECT dataset, with 5 prompts from each category:
- Illegal activities
- Harmful content generation
- Deceptive practices
- Privacy violations
- Dangerous instructions
- Discriminatory content

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{cruz2025multiturn,
  title={Multi-Turn Jailbreaks: Exploring Conversational Attack Strategies},
  author={[Authors]},
  booktitle={[Workshop Name]},
  year={2025}
}
```

## Ethical Considerations

This tool is intended for research and educational purposes only. It should be used responsibly to improve AI safety, not to cause harm. Please use in accordance with applicable laws and regulations.

## License

MIT License - see LICENSE file for details.