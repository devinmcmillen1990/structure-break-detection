# visualize.py
import argparse
import sys
# from typing import Optional

import pandas as pd

# Optional, helps ANSI colors work reliably on Windows terminals
# try:
#     from colorama import init as colorama_init
#     colorama_init()
# except Exception:
#     pass

# # ANSI colors
# GREEN = "\033[92m"
# BLUE = "\033[94m"
# RESET = "\033[0m"

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

# def print_color_table(df: pd.DataFrame) -> None:
#     # Ensure we have time/value/period columns to display
#     missing = [c for c in ["time", "value", "period"] if c not in df.columns]
#     if missing:
#         print(f"Warning: missing expected columns: {missing}", file=sys.stderr)

#     # Sort by time if it exists
#     if "time" in df.columns:
#         df = df.sort_values("time")

#     # Header
#     print(f"{'time':>6}  {'value':>12}  {'period':>6}")
#     print("-" * 30)

#     for _, row in df.iterrows():
#         t = row.get("time", "")
#         v = row.get("value", "")
#         p = row.get("period", "")

#         # Color period
#         if p == 0:
#             p_str = f"{GREEN}{p}{RESET}"
#         elif p == 1:
#             p_str = f"{BLUE}{p}{RESET}"
#         else:
#             p_str = str(p)

#         # Format value if numeric
#         try:
#             v_str = f"{float(v):>12.6f}"
#         except Exception:
#             v_str = f"{str(v):>12}"

#         print(f"{str(t):>6}  {v_str}  {p_str:>6}")

def maybe_plot(full_df: pd.DataFrame, target_id: int) -> None:
    """
    Show an interactive matplotlib figure for the provided dataframe.

    The figure includes a small text box where the user can type a new id and
    press Enter to update the plot (and print the table to stdout).
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import TextBox
    except ImportError:
        print("matplotlib not installed; skipping plot. Install with: pip install matplotlib")
        return

    # Validate required columns exist somewhere in the dataframe
    if "time" not in full_df.columns or "value" not in full_df.columns or "period" not in full_df.columns:
        print("Cannot plot: need 'time', 'value', 'period' columns.", file=sys.stderr)
        return

    # Create figure and a textbox for entering a new id
    fig, ax = plt.subplots(figsize=(9, 4))
    plt.subplots_adjust(bottom=0.22)

    def render_id(id_value: int) -> None:
        try:
            filtered = filter_by_id(full_df, id_value)
        except KeyError as e:
            ax.clear()
            ax.text(0.5, 0.5, str(e), ha="center", va="center")
            fig.canvas.draw_idle()
            return

        if filtered.empty:
            ax.clear()
            ax.text(0.5, 0.5, f"No rows found for id={id_value}", ha="center", va="center")
            fig.canvas.draw_idle()
            return

        colors = filtered["period"].map({0: "green", 1: "blue"}).fillna("gray")
        ax.clear()
        ax.scatter(filtered["time"], filtered["value"], c=colors, s=4)
        ax.set_xlabel("time")
        ax.set_ylabel("value")
        ax.set_title(f"id={id_value} — type a new id below and press Enter")
        fig.canvas.draw_idle()

        # Also print the textual table in the console for the newly selected id
        # print_color_table(filtered)

    # Initial render
    render_id(target_id)

    # Text box axes (at the bottom of the figure)
    axbox = plt.axes([0.2, 0.06, 0.6, 0.05])
    text_box = TextBox(axbox, "id", initial=str(target_id))

    def submit(text: str) -> None:
        text = text.strip()
        if not text:
            return
        try:
            new_id = int(text)
        except ValueError:
            print(f"Invalid id (not an integer): '{text}'", file=sys.stderr)
            return
        render_id(new_id)

    text_box.on_submit(submit)

    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Filter Parquet by id and color-code period.")
    parser.add_argument("path", help="Path to the Parquet file (e.g., ./project-data/X_test.reduced.parquet)")
    parser.add_argument("--id", type=int, required=True, help="Target id to filter (e.g., --id 10015)")
    #parser.add_argument("--plot", action="store_true", help="Show a matplotlib scatter")
    args = parser.parse_args()

    df = load_dataframe(args.path)
    filtered = filter_by_id(df, args.id)

    if filtered.empty:
        print(f"No rows found for id={args.id}")
        return

    #print_color_table(filtered)

    #if args.plot:
        # Pass the full dataframe so the interactive plot can select different ids
    maybe_plot(df, args.id)

if __name__ == "__main__":
    main()
