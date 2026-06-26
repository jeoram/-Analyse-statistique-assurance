from pathlib import Path
import pandas as pd
import numpy as np


def generate_reporting_data(path: Path) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 1500
    df = pd.DataFrame({
        "month": pd.date_range("2023-01-01", periods=n, freq="M")[:n],
        "region": rng.choice(["North", "South", "East", "West"], size=n),
        "segment": rng.choice(["standard", "premium", "family"], size=n),
        "new_policies": rng.integers(40, 180, size=n),
        "renewals": rng.integers(30, 160, size=n),
        "claims_count": rng.integers(5, 60, size=n),
        "claim_amount": np.clip(rng.normal(180000, 50000, size=n), 50000, 500000).round(2),
        "satisfaction_score": rng.integers(1, 6, size=n),
        "churn_rate": np.clip(rng.normal(0.12, 0.04, size=n), 0.02, 0.35).round(3),
    })
    df.to_csv(path, index=False)
    return df


def create_outputs(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True)
    summary = df.groupby(["region", "segment"]).agg(
        new_policies=("new_policies", "sum"),
        renewals=("renewals", "sum"),
        claims_count=("claims_count", "sum"),
        claim_amount=("claim_amount", "sum"),
        avg_satisfaction=("satisfaction_score", "mean"),
        avg_churn=("churn_rate", "mean"),
    ).reset_index()
    summary.to_csv(output_dir / "dashboard_summary.csv", index=False)

    documentation = """# Dashboard Documentation

## Purpose
This dashboard provides a business view of insurance performance indicators.

## KPI definitions
- New policies: number of policies sold during the month
- Renewals: number of renewals completed
- Claims count: number of claims opened
- Claim amount: total claim amount
- Satisfaction score: average customer satisfaction score
- Churn rate: expected churn rate by region and segment

## Data sources
- reporting dataset generated for this project

## Notes
- Report should be reviewed regularly with business stakeholders
"""
    (output_dir / "dashboard_documentation.md").write_text(documentation, encoding="utf-8")

    print("Project 2 completed successfully")


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(exist_ok=True)
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    df = generate_reporting_data(data_dir / "reporting_data.csv")
    create_outputs(df, output_dir)
