from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def generate_data(path: Path) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 2000
    df = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(1, n + 1)],
        "age": rng.integers(20, 75, size=n),
        "tenure_months": rng.integers(6, 120, size=n),
        "annual_premium": np.clip(rng.normal(900, 220, size=n), 200, 2500).round(2),
        "claims_count": rng.poisson(0.9, size=n),
        "claim_amount": np.clip(rng.gamma(2.0, 120, size=n), 0, 4000).round(2),
        "satisfaction_score": rng.integers(1, 6, size=n),
        "payment_delay_days": rng.poisson(1.7, size=n),
        "has_dispute": rng.binomial(1, 0.12, size=n),
        "churn": rng.binomial(1, 0.15, size=n),
        "segment": rng.choice(["standard", "premium", "family"], size=n, p=[0.55, 0.25, 0.20]),
    })
    df.to_csv(path, index=False)
    return df


def analyze(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True)
    summary = df.describe(include='all').T
    summary.to_csv(output_dir / "summary_stats.csv")

    plt.figure(figsize=(8, 5))
    sns.histplot(df["annual_premium"], bins=20, kde=True)
    plt.title("Distribution of annual premium")
    plt.tight_layout()
    plt.savefig(output_dir / "premium_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(x="segment", y="claims_count", data=df)
    plt.title("Claims by customer segment")
    plt.tight_layout()
    plt.savefig(output_dir / "claims_by_segment.png", dpi=180)
    plt.close()

    corr = df[["age", "tenure_months", "annual_premium", "claims_count", "claim_amount", "satisfaction_score", "payment_delay_days", "has_dispute", "churn"]].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=180)
    plt.close()

    print("Project 1 completed successfully")


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(exist_ok=True)
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    df = generate_data(data_dir / "insurance_data.csv")
    analyze(df, output_dir)
