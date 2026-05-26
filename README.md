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

The notebook expects data in the following structure:

```
evaluation_data/
  sroie_ground_truth.yaml        # Ground truth field values per image

results/
  sroie_internvl3_5_8b_results.yaml
  sroie_sonnet_3_5_results.yaml
```

### Ground truth format

Each image key is the receipt filename stem. The four SROIE fields are `company`,
`date`, `address`, and `total`.

```yaml
images:
  X00016469612:
    company: "SYARIKAT PERNIAGAAN GIN KEE"
    date: "25/12/2018"
    address: "NO.1, GROUND FLOOR, JALAN LAGENDA 2"
    total: "4.95"
```

### Results format

Each entry has the image key, a `processing_time` in seconds, and extracted `fields`.

```yaml
results:
  - image: X00016469612
    processing_time: 1.05
    fields:
      company: "SYARIKAT PERNIAGAAN GIN KEE"
      date: "25/12/2018"
      address: "NO.1, GROUND FLOOR, JALAN LAGENDA 2"
      total: "4.95"
```

### Converting SROIE benchmark outputs

The [SROIE benchmark notebook](https://github.com/tmnestor/SROIE) outputs per-image
CSV and summary JSON. To use those results here, convert them to the YAML formats
above:

1. **Ground truth** — the SROIE dataset stores ground truth as one JSON file per image
   in `data/sroie/test/entities/`. Merge these into a single `sroie_ground_truth.yaml`
   mapping each image stem to its four fields.

2. **Model results** — the benchmark notebook writes
   `data/sroie/output/sroie_internvl3_per_image.csv` with columns
   `image_id, company_gt, company_pred, ..., total_match`. Reshape this into the YAML
   results format, adding `processing_time` from the summary JSON.

## Configuration

Edit `comparison_config.yaml` to add models, change fields, or adjust display settings. All keys are required.

## Output

The notebook produces:

- A three-panel dashboard (overall F1, per-field heatmap, throughput)
- CSV exports for per-field F1, model summary, and throughput

Output is saved to `output/comparison/` by default.
