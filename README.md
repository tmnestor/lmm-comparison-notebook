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

## Scoring

### Field normalization

Before comparison, both the predicted and ground truth values are normalized according
to the field type. This matches the [ICDAR 2019 SROIE](https://rrc.cvc.uab.es/?ch=13)
evaluation protocol.

| Field | Normalization |
|-------|---------------|
| `date` | Parse multiple formats (ISO `YYYY-MM-DD`, `DD-MM-YYYY`, `DD/MM/YYYY`, `DD.MM.YYYY`, `DD Month YYYY`) and convert to canonical `DD/MM/YYYY`. Falls back to text normalization if no pattern matches. |
| `total` | Strip currency symbols (`$`, `RM`, etc.), remove commas from numbers, format as a 2-decimal float (e.g. `"RM 1,200.00"` becomes `"1200.00"`). Falls back to text normalization if not a valid number. |
| All other fields | Lowercase, collapse all internal whitespace to a single space, strip leading/trailing whitespace. |

After normalization, scoring uses exact string equality (match = 1.0, mismatch = 0.0).

### Scalar fields

Scalar fields (configured under `fields.scalar`) are scored per-image as a binary
match after normalization:

```
score = 1.0 if normalize(predicted) == normalize(ground_truth) else 0.0
```

The per-field F1 reported in the dashboard is the mean of these binary scores across
all images (equivalent to accuracy when every image has a ground truth value).

### List fields (position-aware F1)

List fields (configured under `fields.list`) contain multiple values separated by `|`.
These are scored using position-aware token-level F1:

1. Split both predicted and ground truth on `|`, strip and lowercase each item.
2. Align items by position index (not set intersection).
3. Count:
   - **TP** — items at the same position that match exactly.
   - **FP** — predicted items with no matching GT at that position (extra or wrong).
   - **FN** — GT items with no matching prediction at that position (missing or wrong).
4. Compute F1:

```
F1 = 2 * TP / (2 * TP + FP + FN)
```

This penalizes both missing items and ordering errors. For example:

| Ground truth | Predicted | TP | FP | FN | F1 |
|---|---|---|---|---|---|
| `A\|B\|C` | `A\|B\|C` | 3 | 0 | 0 | 1.00 |
| `A\|B\|C` | `A\|C\|B` | 1 | 2 | 2 | 0.33 |
| `A\|B\|C` | `A\|B` | 2 | 0 | 1 | 0.80 |
| `A\|B` | `A\|B\|C` | 2 | 1 | 0 | 0.80 |

The current SROIE configuration uses only scalar fields (`list: []`), but the scoring
engine supports list fields for datasets that require them.

## Configuration

Edit `comparison_config.yaml` to add models, change fields, or adjust display settings. All keys are required.

## Output

The notebook produces:

- A three-panel dashboard (overall F1, per-field heatmap, throughput)
- CSV exports for per-field F1, model summary, and throughput

Output is saved to `output/comparison/` by default.
