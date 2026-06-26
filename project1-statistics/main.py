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

    kpi_df = pd.DataFrame({
        "kpi": [
            "total_customers",
            "average_premium",
            "average_claim_amount",
            "claim_frequency",
            "average_satisfaction",
            "churn_rate",
            "dispute_rate",
            "late_payment_rate"
        ],
        "value": [
            len(df),
            round(df["annual_premium"].mean(), 2),
            round(df["claim_amount"].mean(), 2),
            round((df["claims_count"] > 0).mean(), 3),
            round(df["satisfaction_score"].mean(), 2),
            round(df["churn"].mean(), 3),
            round(df["has_dispute"].mean(), 3),
            round((df["payment_delay_days"] > 0).mean(), 3)
        ]
    })
    kpi_df.to_csv(output_dir / "kpis.csv", index=False)

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

    plt.figure(figsize=(8, 5))
    avg_satisfaction = df.groupby("segment")["satisfaction_score"].mean().reset_index()
    sns.barplot(data=avg_satisfaction, x="segment", y="satisfaction_score", hue="segment", palette="viridis", dodge=False, legend=False)
    plt.title("Average satisfaction by segment")
    plt.tight_layout()
    plt.savefig(output_dir / "satisfaction_by_segment.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    churn_by_segment = df.groupby("segment")["churn"].mean().reset_index()
    sns.barplot(data=churn_by_segment, x="segment", y="churn", hue="segment", palette="rocket", dodge=False, legend=False)
    plt.title("Churn rate by segment")
    plt.tight_layout()
    plt.savefig(output_dir / "churn_by_segment.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="annual_premium", y="claim_amount", hue="churn", size="claims_count", alpha=0.6)
    plt.title("Premium vs claim amount by churn status")
    plt.tight_layout()
    plt.savefig(output_dir / "premium_vs_claims.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=kpi_df, x="kpi", y="value", hue="kpi", palette="Set2", dodge=False, legend=False)
    plt.xticks(rotation=45, ha="right")
    plt.title("Business KPI overview")
    plt.tight_layout()
    plt.savefig(output_dir / "kpi_overview.png", dpi=180)
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
