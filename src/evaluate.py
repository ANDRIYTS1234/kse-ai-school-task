"""Final evaluation on the holdout test period (Step 7). The only module allowed to read test rows."""
import pandas as pd

from src.split import final_test_rows


def describe_period(df: pd.DataFrame) -> dict:
    """Descriptive statistics of a period (no model, no decision input)."""
    onset = df[df["regime"] == "onset"]
    return {
        "from": df["t"].min(), "to": df["t"].max(), "rows": len(df),
        "base_rate": df["y"].mean(), "onset_share": len(onset) / len(df), "onset_base_rate": onset["y"].mean(),
    }


def describe_test_period(features: pd.DataFrame) -> dict:
    """Descriptive base rates of the test period (allowed before Step 7: not a decision input)."""
    return describe_period(final_test_rows(features))
