# visualize.py
import argparse
import sys
from typing import Optional

import pandas as pd

# Optional, helps ANSI colors work reliably on Windows terminals
try:
    from colorama import init as colorama_init
    colorama_init()
except Exception:
    pass

# ANSI colors
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"

def load_dataframe(path: str) -> pd.DataFrame:
    # Use forward slashes or raw string on Windows paths
    df = pd.read_parquet(path)

    # If we have a MultiIndex with names that include id/time, normalize to columns
    if isinstance(df.index, pd.MultiIndex):
        idx_names = [n.lower() if isinstance(n, str) else n for n in df.index.names]
        if "id" in idx_names and "time" in idx_names:
            df = df.reset_index()  # bring id, time out of the index into columns

    # Normalize column names for convenience
    df.columns = [c.strip().lower() if isinstance(c, str) else c for c in df.columns]
    return df

def filter_by_id(df: pd.DataFrame, target_id: int) -> pd.DataFrame:
    if "id" not in df.columns:
        # If id is still not a column, try to grab from index if present
        if isinstance(df.index, pd.MultiIndex) and "id" in (df.index.names or []):
            return df.xs(target_id, level="id").reset_index()
        else:
            raise KeyError(
                "Could not find 'id' column or index level. "
                f"Available columns: {list(df.columns)} | index names: {df.index.names}"
            )
    return df[df["id"] == target_id]

def print_color_table(df: pd.DataFrame) -> None:
    # Ensure we have time/value/period columns to display
    missing = [c for c in ["time", "value", "period"] if c not in df.columns]
    if missing:
        print(f"Warning: missing expected columns: {missing}", file=sys.stderr)

    # Sort by time if it exists
    if "time" in df.columns:
        df = df.sort_values("time")

    # Header
    print(f"{'time':>6}  {'value':>12}  {'period':>6}")
    print("-" * 30)

    for _, row in df.iterrows():
        t = row.get("time", "")
        v = row.get("value", "")
        p = row.get("period", "")

        # Color period
        if p == 0:
            p_str = f"{GREEN}{p}{RESET}"
        elif p == 1:
            p_str = f"{BLUE}{p}{RESET}"
        else:
            p_str = str(p)

        # Format value if numeric
        try:
            v_str = f"{float(v):>12.6f}"
        except Exception:
            v_str = f"{str(v):>12}"

        print(f"{str(t):>6}  {v_str}  {p_str:>6}")

def maybe_plot(df: pd.DataFrame, target_id: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot. Install with: pip install matplotlib")
        return

    if "time" not in df.columns or "value" not in df.columns or "period" not in df.columns:
        print("Cannot plot: need 'time', 'value', 'period' columns.", file=sys.stderr)
        return

    # Simple scatter, color by period (0=green, 1=blue)
    colors = df["period"].map({0: "green", 1: "blue"}).fillna("gray")
    plt.scatter(df["time"], df["value"], c=colors)
    plt.xlabel("time")
    plt.ylabel("value")
    plt.title(f"id={target_id} (green=period 0, blue=period 1)")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Filter Parquet by id and color-code period.")
    parser.add_argument("path", help="Path to the Parquet file (e.g., ./project-data/X_test.reduced.parquet)")
    parser.add_argument("--id", type=int, required=True, help="Target id to filter (e.g., --id 10015)")
    parser.add_argument("--plot", action="store_true", help="Show a matplotlib scatter")
    args = parser.parse_args()

    df = load_dataframe(args.path)
    filtered = filter_by_id(df, args.id)

    if filtered.empty:
        print(f"No rows found for id={args.id}")
        return

    print_color_table(filtered)

    if args.plot:
        maybe_plot(filtered, args.id)

if __name__ == "__main__":
    main()
