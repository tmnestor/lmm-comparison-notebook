"""Comparison utilities for model evaluation notebook.

Provides YAML loaders, field comparison logic, and seaborn visualizations.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


def _normalize(value: str | None) -> str | None:
    """Normalize a field value: strip whitespace, treat empty as None."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


# --- SROIE-specific normalization (matches ICDAR 2019 SROIE evaluation) ---

_WHITESPACE_RE = re.compile(r"\s+")
_CURRENCY_RE = re.compile(r"[$\u00a3\u20ac\u00a5]|RM")
_COMMA_IN_NUMBER_RE = re.compile(r"(\d),(\d)")

_MONTHS = {
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}


def _normalize_text(text: str) -> str:
    """Normalize text: lowercase, collapse whitespace, strip."""
    return _WHITESPACE_RE.sub(" ", text.lower().strip())


def _normalize_total(text: str) -> str:
    """Normalize monetary total: strip currency, commas; format as 2-decimal float."""
    text = text.strip()
    text = _CURRENCY_RE.sub("", text)
    text = _COMMA_IN_NUMBER_RE.sub(r"\1\2", text)
    text = text.strip()
    try:
        return f"{float(text):.2f}"
    except ValueError:
        return _normalize_text(text)


def _normalize_date(text: str) -> str:
    """Normalize date to DD/MM/YYYY format."""
    text = text.strip()

    # ISO: YYYY-MM-DD or YYYY/MM/DD
    m = re.match(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", text)
    if m:
        return f"{int(m.group(3)):02d}/{int(m.group(2)):02d}/{m.group(1)}"

    # DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
    m = re.match(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", text)
    if m:
        return f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{m.group(3)}"

    # DD Mon YYYY (e.g. "25 December 2018")
    m = re.match(r"(\d{1,2})\s+(\w{3,})\s+(\d{4})", text, re.IGNORECASE)
    if m:
        month_str = m.group(2)[:3].lower()
        if month_str in _MONTHS:
            return f"{int(m.group(1)):02d}/{_MONTHS[month_str]}/{m.group(3)}"

    return _normalize_text(text)


def _normalize_for_field(field_name: str, value: str) -> str:
    """Apply field-specific normalization.

    Date and total fields get format-aware parsing. All other fields get
    whitespace-collapsed case-insensitive comparison (SROIE standard).
    Handles both SROIE (lowercase) and WildReceipt (capitalized) field names.
    """
    name = field_name.lower()
    if name in ("total", "subtotal", "tax", "tips", "prod_price"):
        return _normalize_total(value)
    if name == "date":
        return _normalize_date(value)
    return _normalize_text(value)


def compare_field(
    extracted: str | None,
    expected: str | None,
    *,
    is_list: bool,
    field_name: str = "",
) -> float:
    """Compare an extracted field value against ground truth.

    Args:
        extracted: The model's extracted value (or None if missing).
        expected: The ground truth value (or None if not applicable).
        is_list: If True, split on '|' and use position-aware F1.
        field_name: Field name used for SROIE-specific normalization.

    Returns:
        Score from 0.0 to 1.0.
    """
    extracted = _normalize(extracted)
    expected = _normalize(expected)

    if extracted is None and expected is None:
        return 1.0
    if extracted is None or expected is None:
        return 0.0

    if not is_list:
        norm_ext = _normalize_for_field(field_name, extracted)
        norm_exp = _normalize_for_field(field_name, expected)
        return 1.0 if norm_ext == norm_exp else 0.0

    ext_items = [item.strip().lower() for item in extracted.split("|")]
    exp_items = [item.strip().lower() for item in expected.split("|")]

    if not ext_items and not exp_items:
        return 1.0

    tp = 0
    fp = 0
    fn = 0
    for i in range(max(len(ext_items), len(exp_items))):
        has_ext = i < len(ext_items)
        has_exp = i < len(exp_items)
        if has_ext and has_exp:
            if ext_items[i] == exp_items[i]:
                tp += 1
            else:
                fp += 1
                fn += 1
        elif has_ext:
            fp += 1
        else:
            fn += 1

    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 1.0


class ConfigError(ValueError):
    """Raised when comparison config is invalid."""


def load_config(path: Path) -> dict:
    """Load and validate comparison_config.yaml.

    Args:
        path: Path to the YAML config file.

    Returns:
        Validated config dict.

    Raises:
        FileNotFoundError: If the file does not exist.
        ConfigError: If required keys are missing.
    """
    if not path.exists():
        msg = f"Config file not found: {path.resolve()}"
        raise FileNotFoundError(msg)

    raw = yaml.safe_load(path.read_text())

    for key in ("ground_truth", "fields", "models", "display"):
        if key not in raw:
            msg = (
                f"Missing required key '{key}' in comparison config.\n"
                f"  File: {path.resolve()}\n"
                f"  Expected: '{key}' section in the YAML file.\n"
                f"  Fix: Add the '{key}' section to {path.name}"
            )
            raise ConfigError(msg)

    for sub_key in ("scalar", "list"):
        if sub_key not in raw["fields"]:
            msg = (
                f"Missing required key 'fields.{sub_key}' in comparison config.\n"
                f"  File: {path.resolve()}\n"
                f"  Expected: 'fields.{sub_key}' as a list of field names, e.g.:\n"
                f"    fields:\n"
                f"      {sub_key}: [DOCUMENT_TYPE, BUSINESS_ABN]\n"
                f"  Fix: Add 'fields.{sub_key}' to {path.name}"
            )
            raise ConfigError(msg)

    return raw


def load_results(path: Path) -> dict[str, dict]:
    """Load a model results YAML file.

    Args:
        path: Path to the model results YAML.

    Returns:
        Dict mapping image_stem to {"fields": {...}, "processing_time": float}.

    Raises:
        FileNotFoundError: If the file does not exist.
        ConfigError: If 'results' key is missing.
    """
    if not path.exists():
        msg = f"Results file not found: {path.resolve()}"
        raise FileNotFoundError(msg)

    raw = yaml.safe_load(path.read_text())

    if "results" not in raw:
        msg = (
            f"Missing required key 'results' in results file.\n"
            f"  File: {path.resolve()}\n"
            f"  Expected: 'results' key containing a list of image results, e.g.:\n"
            f"    results:\n"
            f"      - image: invoice_001\n"
            f"        processing_time: 2.34\n"
            f"        fields:\n"
            f"          DOCUMENT_TYPE: INVOICE\n"
            f"  Fix: Add the 'results' list to {path.name}"
        )
        raise ConfigError(msg)

    return {
        entry["image"]: {
            "fields": entry.get("fields", {}),
            "processing_time": entry.get("processing_time", 0.0),
        }
        for entry in raw["results"]
    }


def load_ground_truth(path: Path) -> dict[str, dict[str, str]]:
    """Load a ground truth YAML file.

    Args:
        path: Path to the ground truth YAML.

    Returns:
        Dict mapping image_stem to {field_name: value}.

    Raises:
        FileNotFoundError: If the file does not exist.
        ConfigError: If 'images' key is missing.
    """
    if not path.exists():
        msg = f"Ground truth file not found: {path.resolve()}"
        raise FileNotFoundError(msg)

    raw = yaml.safe_load(path.read_text())

    if "images" not in raw:
        msg = (
            f"Missing required key 'images' in ground truth file.\n"
            f"  File: {path.resolve()}\n"
            f"  Expected: 'images' key mapping image stems to field values, e.g.:\n"
            f"    images:\n"
            f"      invoice_001:\n"
            f"        DOCUMENT_TYPE: INVOICE\n"
            f"  Fix: Add the 'images' mapping to {path.name}"
        )
        raise ConfigError(msg)

    return raw["images"]


def evaluate_model(
    results: dict[str, dict],
    ground_truth: dict[str, dict[str, str]],
    scalar_fields: list[str],
    list_fields: list[str],
) -> pd.DataFrame:
    """Evaluate a model's extractions against ground truth.

    Only images present in both results and ground_truth are evaluated.
    Fields absent from both result and GT for an image are skipped.

    Args:
        results: From load_results(). {image: {fields: {...}, processing_time: float}}.
        ground_truth: From load_ground_truth(). {image: {field: value}}.
        scalar_fields: Field names to evaluate with exact match.
        list_fields: Field names to evaluate with position-aware F1.

    Returns:
        DataFrame with columns: image, field, score.
    """
    all_fields = [(f, False) for f in scalar_fields] + [(f, True) for f in list_fields]
    rows: list[dict] = []

    common_images = set(results) & set(ground_truth)

    for image in sorted(common_images):
        extracted_fields = results[image].get("fields", {})
        gt_fields = ground_truth[image]

        for field_name, is_list in all_fields:
            ext_val = extracted_fields.get(field_name)
            gt_val = gt_fields.get(field_name)

            # Skip fields absent from both
            if ext_val is None and gt_val is None:
                continue

            score = compare_field(
                ext_val, gt_val, is_list=is_list, field_name=field_name
            )
            rows.append({"image": image, "field": field_name, "score": score})

    return pd.DataFrame(rows, columns=["image", "field", "score"])


def plot_overall_f1(
    summary_df: pd.DataFrame,
    colors: dict[str, str],
    ax: plt.Axes,
) -> None:
    """Bar chart of mean F1 per model with error bars.

    Args:
        summary_df: DataFrame with columns: model, mean_f1, std_f1.
        colors: Mapping of model display name to hex color.
        ax: Matplotlib axes to draw on.
    """
    palette = [colors.get(m, "#999999") for m in summary_df["model"]]
    sns.barplot(
        data=summary_df,
        x="model",
        y="mean_f1",
        hue="model",
        palette=palette,
        ax=ax,
        edgecolor="black",
        linewidth=0.8,
        legend=False,
    )
    # Add error bars manually
    for i, row in summary_df.iterrows():
        ax.errorbar(
            i,
            row["mean_f1"],
            yerr=row["std_f1"],
            fmt="none",
            color="black",
            capsize=4,
            linewidth=1.2,
        )
        ax.text(
            i,
            row["mean_f1"] + row["std_f1"] + 0.02,
            f"{row['mean_f1']:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    ax.set_title("Overall Mean F1 Score", fontweight="bold", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("Mean F1")
    ax.set_ylim(0, 1.15)
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")


def plot_field_heatmap(field_df: pd.DataFrame, ax: plt.Axes) -> None:
    """Heatmap of per-field F1 scores across models.

    Args:
        field_df: Pivot table with rows=fields, columns=models, values=F1 (0-1).
        ax: Matplotlib axes to draw on.
    """
    sns.heatmap(
        field_df,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "F1 Score"},
    )
    ax.set_title("Per-Field F1 Scores", fontweight="bold", fontsize=12)
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")


def plot_throughput(
    throughput_df: pd.DataFrame,
    colors: dict[str, str],
    ax: plt.Axes,
) -> None:
    """Bar chart of throughput (docs/minute) per model.

    Args:
        throughput_df: DataFrame with columns: model, docs_per_min.
        colors: Mapping of model display name to hex color.
        ax: Matplotlib axes to draw on.
    """
    palette = [colors.get(m, "#999999") for m in throughput_df["model"]]
    sns.barplot(
        data=throughput_df,
        x="model",
        y="docs_per_min",
        hue="model",
        palette=palette,
        ax=ax,
        edgecolor="black",
        linewidth=0.8,
        legend=False,
    )
    for i, row in throughput_df.iterrows():
        ax.text(
            i,
            row["docs_per_min"] + 0.5,
            f"{row['docs_per_min']:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    ax.set_title("Throughput (docs/min)", fontweight="bold", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("Documents / Minute")
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
