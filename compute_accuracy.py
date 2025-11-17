from pathlib import Path
import pandas as pd

# Directories
LABELED_DIR = Path("labeled")
PRED_DIR = Path("llm_gen/gemma")

# Column names (change here if yours differ)
ID_COL = "Unnamed: 0"
LABEL_COL = "label"


def eval_one(gold_path: Path, pred_path: Path):
    """Return (correct, total, accuracy) for a single pair of CSVs."""
    gold = pd.read_csv(gold_path)
    pred = pd.read_csv(pred_path)

    # Prefer joining on an ID column if present
    if ID_COL in gold.columns and ID_COL in pred.columns:
        merged = gold[[ID_COL, LABEL_COL]].merge(
            pred[[ID_COL, LABEL_COL]],
            on=ID_COL,
            how="inner",
            suffixes=("_gold", "_pred"),
        )
    else:
        # Fallback: align by row order
        n = min(len(gold), len(pred))
        merged = pd.DataFrame({
            "label_gold": gold[LABEL_COL].iloc[:n].to_numpy(),
            "label_pred": pred[LABEL_COL].iloc[:n].to_numpy(),
        })

    correct = (merged["label_gold"] == merged["label_pred"]).sum()
    total = len(merged)
    acc = correct / total if total else float("nan")
    return correct, total, acc


def main():
    total_correct = 0
    total_examples = 0
    summary_rows = []
    # Loop over all gold files in labeled/
    for gold_path in sorted(LABELED_DIR.glob("*.csv")):
        stem = gold_path.stem          # e.g. "august"
        pred_path = PRED_DIR / f"gemma_4b_{stem}.csv"

        if not pred_path.exists():
            print(f"[WARN] Missing prediction file for {gold_path.name}, "
                  f"expected {pred_path.name}")
            continue

        correct, total, acc = eval_one(gold_path, pred_path)
        total_correct += correct
        total_examples += total

        print(f"{stem:15s} accuracy = {acc:.4f}  ({correct}/{total})")
        summary_rows.append([stem, correct, total, acc])

    if total_examples:
        overall_acc = total_correct / total_examples
        print("-" * 50)
        print(f"OVERALL        accuracy = {overall_acc:.4f} "
              f"({total_correct}/{total_examples})")
    else:
        print("No matching gold/prediction CSV pairs found.")
    results_df= pd.DataFrame(summary_rows)
    results_df.to_csv("gemma_4b_results.csv",index=False)


if __name__ == "__main__":
    main()
