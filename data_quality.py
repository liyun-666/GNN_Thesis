import argparse
import json
from pathlib import Path

import pandas as pd


VALID_BEHAVIORS = {0, 1, 2, 3}


def audit_df(df: pd.DataFrame) -> dict:
    report = {
        "rows": int(len(df)),
        "missing": {k: int(v) for k, v in df[["u", "i", "b", "t"]].isna().sum().to_dict().items()},
        "duplicate_ui_bt": int(df.duplicated(subset=["u", "i", "b", "t"]).sum()),
        "invalid_behavior": int((~df["b"].isin(list(VALID_BEHAVIORS))).sum()),
        "non_positive_timestamp": int((df["t"] <= 0).sum()),
        "negative_user": int((df["u"] < 0).sum()),
        "negative_item": int((df["i"] < 0).sum()),
        "num_users": int(df["u"].nunique()),
        "num_items": int(df["i"].nunique()),
        "behavior_dist": {int(k): int(v) for k, v in df["b"].value_counts().sort_index().to_dict().items()},
    }

    uc = df.groupby("u").size()
    ic = df.groupby("i").size()
    report["user_inter_quantiles"] = {str(k): float(v) for k, v in uc.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).to_dict().items()}
    report["item_inter_quantiles"] = {str(k): float(v) for k, v in ic.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).to_dict().items()}

    one_item_ratio = float((ic == 1).mean())
    report["item_single_interaction_ratio"] = one_item_ratio

    return report


def clean_df(
    df: pd.DataFrame,
    min_user_inter: int = 5,
    min_item_inter: int = 2,
) -> pd.DataFrame:
    x = df[["u", "i", "b", "t"]].copy()
    x = x.dropna()
    x = x.astype({"u": int, "i": int, "b": int, "t": int})

    x = x[x["b"].isin(list(VALID_BEHAVIORS))]
    x = x[x["t"] > 0]
    x = x[(x["u"] >= 0) & (x["i"] >= 0)]
    x = x.drop_duplicates(subset=["u", "i", "b", "t"])

    changed = True
    while changed:
        before = len(x)
        user_cnt = x.groupby("u").size()
        item_cnt = x.groupby("i").size()

        x = x[x["u"].isin(user_cnt[user_cnt >= min_user_inter].index)]
        x = x[x["i"].isin(item_cnt[item_cnt >= min_item_inter].index)]
        x = x.sort_values(["u", "t"]).reset_index(drop=True)

        changed = len(x) != before

    return x


def main():
    parser = argparse.ArgumentParser(description="Audit and clean e-commerce behavior data")
    parser.add_argument("--input", type=str, default="final_real_data.csv")
    parser.add_argument("--clean-output", type=str, default="final_real_data_clean.csv")
    parser.add_argument("--report-json", type=str, default="artifacts/data_quality_report.json")
    parser.add_argument("--min-user-inter", type=int, default=5)
    parser.add_argument("--min-item-inter", type=int, default=2)
    args = parser.parse_args()

    in_path = Path(args.input).resolve()
    out_path = Path(args.clean_output).resolve()
    report_path = Path(args.report_json).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    before = audit_df(df)

    clean = clean_df(df, min_user_inter=args.min_user_inter, min_item_inter=args.min_item_inter)
    after = audit_df(clean)

    clean.to_csv(out_path, index=False)

    recommendation = []
    if before["duplicate_ui_bt"] > 0 or before["missing"]["u"] > 0:
        recommendation.append("Data cleaning is required: duplicates or missing values detected.")
    else:
        recommendation.append("Base data quality passed: no missing, duplicate, or invalid values.")

    if before["item_single_interaction_ratio"] > 0.4:
        recommendation.append("Long-tail filtering is recommended: many items have only one interaction; try min_item_inter=2.")
    if before["behavior_dist"].get(0, 0) / max(1, before["rows"]) > 0.8:
        recommendation.append("Behavior imbalance is severe: click dominates. Use behavior weighting or resampling in training.")

    report = {
        "before": before,
        "after": after,
        "cleaning_params": {
            "min_user_inter": args.min_user_inter,
            "min_item_inter": args.min_item_inter,
        },
        "recommendation": recommendation,
        "input": str(in_path),
        "clean_output": str(out_path),
    }

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] Data audit done")
    print(f"[OK] clean data: {out_path}")
    print(f"[OK] report: {report_path}")
    print("[Summary]")
    print(f"rows: {before['rows']} -> {after['rows']}")
    print(f"users: {before['num_users']} -> {after['num_users']}")
    print(f"items: {before['num_items']} -> {after['num_items']}")


if __name__ == "__main__":
    main()
