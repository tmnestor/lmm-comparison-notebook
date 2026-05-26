# LMM Comparison Notebook

Jupyter notebook for comparing field extraction accuracy (F1) and throughput across large multimodal models.

## Structure

```
comparison_config.yaml   # Models, fields, and display settings
comparison_utils.py      # YAML loaders, scoring logic, seaborn visualizations
model_comparison.ipynb   # Interactive comparison notebook
```

## Setup

Install dependencies:

```bash
pip install pyyaml pandas numpy matplotlib seaborn
```

## Data Layout

The notebook expects data in the following structure (not included in the repo):

```
evaluation_data/
  ground_truth.yaml        # Ground truth field values per image

results/
  internvl3_5_8b_results.yaml
  sonnet_3_5_results.yaml
  ocr_extraction_results.yaml
```

### Ground truth format

```yaml
images:
  invoice_001:
    DOCUMENT_TYPE: INVOICE
    BUSINESS_ABN: "12345678901"
```

### Results format

```yaml
results:
  - image: invoice_001
    processing_time: 2.34
    fields:
      DOCUMENT_TYPE: INVOICE
      BUSINESS_ABN: "12345678901"
```

## Configuration

Edit `comparison_config.yaml` to add models, change fields, or adjust display settings. All keys are required.

## Output

The notebook produces:

- A three-panel dashboard (overall F1, per-field heatmap, throughput)
- CSV exports for per-field F1, model summary, and throughput

Output is saved to `output/comparison/` by default.
