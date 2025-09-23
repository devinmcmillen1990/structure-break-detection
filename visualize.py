# visualize.py
import argparse
import pandas as pd
import numpy as np
import sys

def load_dataframe(path: str) -> pd.DataFrame:
    """
    Load as-is: We *expect* a MultiIndex with levels ('id','time')
    and columns: ['value', 'period'].
    """
    df = pd.read_parquet(path)

    if not isinstance(df.index, pd.MultiIndex) or df.index.names != ["id", "time"]:
        raise ValueError(
            f"Expected MultiIndex ['id','time']; got {df.index.names} and type {type(df.index)}"
        )

    return df

def filter_by_id(df: pd.DataFrame, target_id: int) -> pd.DataFrame:
    """
    Efficient slice by the 'id' level without copying to columns.
    Returns a *view* when possible (no reset_index), preserving the MultiIndex.
    """
    # .xs is very fast for MultiIndex selection:
    # drop_level=False keeps the 'id' level; True drops it so index is just 'time'.
    # For plotting, it's convenient to drop the 'id' level:
    try:
        return df.xs(target_id, level="id", drop_level=True)
    except KeyError:
        # No rows for that id
        return df.iloc[0:0]  # empty frame with same columns


def view_plot(full_df: pd.DataFrame, target_id: int) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import TextBox

    def render_id(id_value: int) -> None:
        filtered = filter_by_id(full_df, id_value)  # index now just 'time'
        ax.clear()

        if filtered.empty:
            ax.text(0.5, 0.5, f"No rows found for id={id_value}", ha="center", va="center")
            fig.canvas.draw_idle()
            print(f"No rows found for id={id_value}")
            return

        times = filtered.index.to_numpy()                  # 'time' from index
        values = filtered["value"].to_numpy()
        p = filtered["period"].to_numpy()

        # Assign colors based on 'period' column
        colors = np.where(p == 0, "green", "blue")

        ax.scatter(times, values, c=colors, s=4)              # change s to adjust dot size
        ax.set_xlabel("time")
        ax.set_ylabel("value")
        ax.set_title(f"id={id_value} — type a new id below and press Enter")
        fig.canvas.draw_idle()
        print(f"Showing data for id={id_value}")

    fig, ax = plt.subplots(figsize=(9, 4))
    plt.subplots_adjust(bottom=0.22)
    render_id(target_id)

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

    #if args.plot:
    view_plot(df, args.id)

if __name__ == "__main__":
    main()
