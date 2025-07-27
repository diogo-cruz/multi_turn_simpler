# Figure Mapping for Reproducible Figures

This document maps the figures from the paper to their generation scripts. Only figures that can be traced to specific scripts are included.

## Python Script Generated Figures

### Figure 2: Single-Turn vs Multi-Turn Attack Success Rates
- **Paper filename**: `placeholder_single_vs_multi_7_2.pdf`
- **Script**: `figure_generation/create_custom_asr_figure_batch7.py`
- **Output**: `custom_asr_figure_batch7.pdf`
- **Requirements**: batch7 JSONL data in `clean_results/final_runs/batch7/`

### Figure 4: Reasoning Model Analysis
- **Paper filename**: `placeholder_reasoning_bars.pdf`
- **Script**: `figure_generation/create_stacked_bar_plot.py`
- **Output**: `strongreject_vs_reasoning_tokens_stacked_bars.pdf`
- **Description**: Stacked bar plot showing relationship between reasoning tokens and attack success

### Correlation Matrices (Appendix)
- **Paper filenames**: Various correlation matrix PDFs
- **Script**: `figure_generation/batch7_correlation_analysis.py`
- **Outputs**:
  - `batch7_correlation_llama31_8b.pdf`
  - `batch7_correlation_llama32_3b.pdf`
  - `batch7_correlation_meta_gpt4omini_llama313_70b.pdf`
  - CSV files with correlation data

### Refusal vs Scratch Sampling
- **Paper filename**: `placeholder_refusal_scratch.pdf`
- **Script**: `figure_generation/create_combined_refusal_plot.py`
- **Output**: `refusal_vs_scratch_combined_plot.pdf`
- **Alternative**: `notebooks/refusal_vs_scratch_sampling_analysis.ipynb` for histogram analysis

### Turn Type Comparisons
- **Paper filename**: `publication_quality_turn_type_comparisons.pdf`
- **Script**: `figure_generation/publication_quality_plots_by_turns.py`
- **Output**: `publication_quality_turn_type_comparisons.pdf`

### Additional Reasoning Analysis
- **Paper filename**: `placeholder_reasoning_analysis.pdf`
- **Script**: `figure_generation/reasoning_token_analysis.py`
- **Description**: Provides detailed reasoning token analysis

## Notebook Generated Figures

### Figures 3 & 10: ASR vs Number of Samples/Turns
- **Paper filename**: `placeholder_samples_turns.pdf`
- **Notebook**: `notebooks/asr_rounds_samples_comparison.ipynb`
- **Control Variables**:
  ```python
  BATCH_NAME = "both"  # Use both batch6A and batch6B
  INCLUDE_COMMAND = False  # Direct request tactic only
  ```
- **Outputs** (PNG format, need PDF conversion):
  - `batch6a_6b_asr_rounds_samples_original_direct_only.png`
  - `batch6a_6b_asr_averaged_original_direct_only.png`
  - Various refusal handling variants

### Additional Analysis Notebooks

1. **ASR vs Samples Theoretical Analysis**
   - Notebook: `notebooks/asr_samples_direct_request_analysis.ipynb`
   - Description: Theoretical analysis using expected maximum formulas

2. **Refusal vs Scratch Detailed Analysis**
   - Notebook: `notebooks/refusal_vs_scratch_sampling_analysis.ipynb`
   - Description: Detailed histogram analysis of sampling behavior

## Data Requirements

Most scripts require:
- JSONL files in `clean_results/final_runs/batch*/` for raw data analysis
- CSV file in `csv_results/results_strongreject.csv` for some analyses

## Notes

- PNG outputs from notebooks can be converted to PDF using ImageMagick: `convert input.png output.pdf`
- Some scripts may need path adjustments depending on your data location
- The `asr_analysis_utils.py` file contains shared functions used by notebooks