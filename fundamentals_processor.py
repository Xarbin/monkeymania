import pandas as pd
import os

def normalize_columns(df):
    """Standardize column names by stripping whitespace and lowering case."""
    df.columns = [col.strip().lower() for col in df.columns]
    return df

def load_csv_safe(path):
    """Safely load a CSV and normalize its headers."""
    try:
        df = pd.read_csv(path)
        df = normalize_columns(df)
        return df
    except Exception as e:
        return f"❌ Error loading file {path}: {e}"

def clean_dataframe(df, key='ticker'):
    """Drop rows with missing ticker and convert to uppercase for merge safety."""
    df = df.dropna(subset=[key])
    df[key] = df[key].astype(str).str.upper().str.strip()
    return df

def merge_dataframes(pre_df, overview_df, post_df):
    """Merge three clean dataframes on 'ticker'."""
    try:
        merged = pd.merge(pre_df, overview_df, on="ticker", how="inner")
        merged = pd.merge(merged, post_df, on="ticker", how="inner")
        return merged
    except Exception as e:
        return f"❌ Merge error: {e}"

def load_and_process_all(pre_path, overview_path, post_path):
    """
    Entry point for full pipeline: load, normalize, clean, and merge all 3 CSVs.
    Returns: Cleaned and merged dataframe, or error message.
    """
    # Load and validate
    pre = load_csv_safe(pre_path)
    overview = load_csv_safe(overview_path)
    post = load_csv_safe(post_path)

    for name, df in zip(["Premarket", "Overview", "Postmarket"], [pre, overview, post]):
        if isinstance(df, str):  # Error message
            return f"❌ {name} Load Failed: {df}"

    # Clean
    pre = clean_dataframe(pre)
    overview = clean_dataframe(overview)
    post = clean_dataframe(post)

    # Merge
    final = merge_dataframes(pre, overview, post)
    return final

# Optional: standalone test
if __name__ == "__main__":
    # Example paths (you can replace these when wiring up the GUI)
    pre_path = "movers_pre.csv"
    overview_path = "daily_overview.csv"
    post_path = "movers_post.csv"

    result = load_and_process_all(pre_path, overview_path, post_path)

    if isinstance(result, pd.DataFrame):
        print("✅ Successfully loaded and merged fundamentals.")
        print(result.head())
    else:
        print(result)