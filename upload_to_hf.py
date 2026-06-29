"""
Upload the X-CASE dataset to Hugging Face Hub.

One row per scenario (1,000 total). Each row contains the scenario metadata,
the unsafe plan, the safe rewrite, and both sets of paired images as HF Image
features so consumers can compare them side by side.

Usage
-----
  # Install dependencies (once)
  pip install datasets huggingface_hub pillow

  # Authenticate (once)
  huggingface-cli login

  # Push dataset
  python upload_to_hf.py

  # Dry-run (validates structure, no upload)
  python upload_to_hf.py --dry-run

  # Custom repo
  python upload_to_hf.py --repo AdonaiVera/X-CASE-dataset
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value
from huggingface_hub import HfApi

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_DIR = Path(__file__).parent / "safe_and_unsafe_situations"
JSON_FILE = DATASET_DIR / "combined_unsafe_party_situations_with_global_categories.json"
DEFAULT_REPO = "adonaivera/X-CASE"


# ---------------------------------------------------------------------------
# Dataset card (shown on HF Hub)
# ---------------------------------------------------------------------------

DATASET_CARD = """\
---
license: cc-by-4.0
language:
- en
tags:
- safety
- multimodal
- generative-agents
- social-simulation
- unsafe-activity-detection
pretty_name: X-CASE — Cross-Modal Consistency and Safety Evaluation
size_categories:
- 1K<n<10K
task_categories:
- text-classification
- visual-question-answering
---

# X-CASE Dataset

> **Cross-Modal Consistency and Safety Evaluation in Generative Agent Social Simulations**
> ACL 2026

X-CASE is a benchmark dataset of **1,000 multimodal social activity scenarios** designed
to evaluate how generative AI agents detect and correct unsafe behaviour during iterative
plan revision. Every scenario contains:

- A **natural-language social activity description** (e.g. a beach party, rooftop gathering).
- An **unsafe hourly plan** — 11 activities from 7 PM to 5 AM that include at least one
  identifiable safety risk (e.g. swimming far from shore at night, throwing flaming objects).
- A **safe rewritten plan** — activity-by-activity rewrites that preserve the social event
  while eliminating each risk.
- **Paired images** retrieved via the Pexels API and verified with CLIP similarity
  (ViT-L/14 @ 336 px, cosine ≥ 0.35) — one image per plan step for both safe and unsafe
  variants.

## Dataset Statistics

| Property | Value |
|---|---|
| Scenarios | 1,000 |
| Unsafe plan steps | 11,000 |
| Safe plan steps | 11,000 |
| Unsafe plan images | 3,937 |
| Safe plan images | 4,227 |
| Hazard categories | 21 |
| Hazard subcategories | 192 |
| Human-reviewed steps | 100 % |

## Example

```python
from datasets import load_dataset

ds = load_dataset("adonaivera/X-CASE")

# Each row has both unsafe and safe plans with their images
row = ds["train"][0]
print(row["category"])          # "Fire & Heat"
print(row["description"])       # social activity description
print(row["plan"][2])           # unsafe activity at 9 PM
print(row["plan_safe"][2])      # safe rewrite at 9 PM
# row["unsafe_images"][2]       # PIL Image for unsafe step
# row["safe_images"][2]         # PIL Image for safe step
```

## Citation

```bibtex
@inproceedings{vera2026xcase,
  title     = {Multimodal Safety Evaluation in Generative Agent Social Simulations},
  author    = {Vera, Alhim and Hinojosa, Carlos and Sanchez, Karen and
               Hamid, Haidar Bin and Kim, Donghoon and Ghanem, Bernard},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association
               for Computational Linguistics (ACL)},
  year      = {2026},
}
```
"""


# ---------------------------------------------------------------------------
# Build rows
# ---------------------------------------------------------------------------

def _resolve_images(paths: list, base_dir: Path) -> list:
    """Convert relative path strings (or None) to absolute paths that exist."""
    return [
        str(base_dir / p) if p is not None and (base_dir / p).exists() else None
        for p in (paths or [])
    ]


def build_rows(data: list[dict]) -> list[dict]:
    """One row per scenario with both unsafe and safe plans + images."""
    rows = []
    for entry in data:
        rows.append({
            "id":                 entry["id"],
            "category":           entry["category"],
            "global_category":    entry["global_category"],
            "description":        entry["description"],
            "plan":               entry["plan"],
            "plan_safe":          entry.get("plan_safe") or [],
            "plan_reviewed":      entry.get("plan_reviewed") or [],
            "safe_plan_reviewed": entry.get("safe_plan_reviewed") or [],
            "unsafe_images":      _resolve_images(entry.get("plan_image_paths"), DATASET_DIR),
            "safe_images":        _resolve_images(entry.get("safe_plan_image_paths"), DATASET_DIR),
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo",    default=DEFAULT_REPO,
                        help="HF repo id, e.g. AdonaiVera/X-CASE")
    parser.add_argument("--private", action="store_true",
                        help="Create a private repo")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build the dataset locally without uploading")
    args = parser.parse_args()

    print(f"Loading JSON from {JSON_FILE} …")
    with open(JSON_FILE) as f:
        data = json.load(f)
    print(f"  {len(data):,} scenarios loaded.")

    rows = build_rows(data)

    if args.dry_run:
        total_img_slots = sum(len(r["unsafe_images"]) + len(r["safe_images"]) for r in rows)
        images_on_disk  = sum(1 for r in rows for p in r["unsafe_images"] + r["safe_images"] if p is not None)
        images_missing  = total_img_slots - images_on_disk
        missing_safe_plan = sum(1 for r in rows if not r["plan_safe"])
        print(f"  rows:                  {len(rows):,}")
        print(f"  image slots total:     {total_img_slots:,}")
        print(f"  images found on disk:  {images_on_disk:,}")
        print(f"  images not on disk:    {images_missing:,}  (will upload as null — download remaining images first)")
        print(f"  rows missing plan_safe:{missing_safe_plan}  (incomplete annotations)")
        print("\nDry-run complete — run without --dry-run to push.")
        return

    features = Features({
        "id":                 Value("int64"),
        "category":           Value("string"),
        "global_category":    Value("string"),
        "description":        Value("string"),
        "plan":               Sequence(Value("string")),
        "plan_safe":          Sequence(Value("string")),
        "plan_reviewed":      Sequence(Value("bool")),
        "safe_plan_reviewed": Sequence(Value("bool")),
        "unsafe_images":      Sequence(Image()),
        "safe_images":        Sequence(Image()),
    })

    print("Building Dataset …")
    ds = Dataset.from_list(rows, features=features)
    dd = DatasetDict({"train": ds})
    print(f"  {len(ds):,} rows, {len(features)} columns")

    api = HfApi()
    api.create_repo(repo_id=args.repo, repo_type="dataset",
                    private=args.private, exist_ok=True)

    # Write dataset card
    card_path = Path("README_HF.md")
    card_path.write_text(DATASET_CARD)
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="dataset",
    )
    card_path.unlink()

    print(f"\nPushing to hf://datasets/{args.repo} …")
    dd.push_to_hub(args.repo, private=args.private)
    print(f"\nDataset live at: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
