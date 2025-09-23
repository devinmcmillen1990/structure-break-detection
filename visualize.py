# visualize.py (additions/edits)

import argparse
import sys
import numpy as np
import pandas as pd

def load_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.MultiIndex) or df.index.names != ["id", "time"]:
        raise ValueError(f"Expected MultiIndex ['id','time']; got {df.index.names}")
    return df

def filter_by_id(df: pd.DataFrame, target_id: int) -> pd.DataFrame:
    try:
        # index becomes just 'time'
        return df.xs(target_id, level="id", drop_level=True).sort_index()
    except KeyError:
        return df.iloc[0:0]

# ---------- NEW: experiment helpers ----------

def _reciprocal(values: np.ndarray) -> np.ndarray:
    # Safe reciprocal: 1/x, NaN where x==0 or invalid
    with np.errstate(divide="ignore", invalid="ignore"):
        y = 1.0 / values.astype(float)
    y[~np.isfinite(y)] = np.nan
    return y

def _running_sum(values: np.ndarray) -> np.ndarray:
    return np.cumsum(values.astype(float))

def _running_avg(values: np.ndarray) -> np.ndarray:
    v = values.astype(float)
    denom = np.arange(1, len(v) + 1, dtype=float)
    return np.cumsum(v) / denom

def _dvalue_dt(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    # Uses numpy.gradient for robust dv/dt, handles uneven dt
    t = times.astype(float)
    v = values.astype(float)
    if len(v) == 0:
        return v
    if len(v) == 1:
        return np.array([np.nan])
    return np.gradient(v, t)

def apply_experiment(times: np.ndarray, values: np.ndarray, kind: str) -> np.ndarray:
    kind = (kind or "").lower()
    print(kind)
    if kind in ("", "none", "identity"):
        return values
    if kind == "reciprocal":
        return _reciprocal(values)
    if kind == "running_sum":
        return _running_sum(values)
    if kind == "running_avg":
        return _running_avg(values)
    if kind in ("dvalue_dt", "derivative", "dvdt"):
        return _dvalue_dt(times, values)
    raise ValueError(f"Unknown experiment '{kind}'")

# ---------- plotting (keeps your TextBox UI) ----------

def view_plot(full_df: pd.DataFrame, target_id: int, experiment: str, overlay: bool, size: float) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import TextBox

    fig, ax = plt.subplots(figsize=(9, 4))
    plt.subplots_adjust(bottom=0.24)

    def render_id(id_value: int) -> None:
        ax.clear()
        filtered = filter_by_id(full_df, id_value)
        if filtered.empty:
            ax.text(0.5, 0.5, f"No rows found for id={id_value}", ha="center", va="center")
            fig.canvas.draw_idle()
            print(f"No rows found for id={id_value}")
            return

        # pull arrays
        times  = filtered.index.to_numpy()
        values = filtered["value"].to_numpy()
        p      = filtered["period"].to_numpy()

        # experiment output
        y_exp = apply_experiment(times, values, experiment)

        # color by period (0=green, 1=blue)
        colors = np.where(p == 0, "green", "blue")

        # plot
        ax.scatter(times, y_exp, c=colors, s=size, label=experiment or "value")

        if overlay and experiment not in ("", "none", "identity"):
            # overlay original values in a neutral style
            ax.scatter(times, values, s=max(size*0.9, 1), alpha=0.4, label="original")

        ax.set_xlabel("time")
        ax.set_ylabel(experiment if experiment else "value")
        ttl = f"id={id_value}"
        if experiment:
            ttl += f" — {experiment}"
        ttl += " — type a new id below and press Enter"
        ax.set_title(ttl)
        ax.legend(loc="best")
        fig.canvas.draw_idle()
        print(f"Showing data for id={id_value} ({experiment or 'value'})")

    # first render
    render_id(target_id)

    # controls
    ax_id = plt.axes([0.15, 0.08, 0.25, 0.06])
    tb_id = TextBox(ax_id, "id", initial=str(target_id))

    ax_exp = plt.axes([0.50, 0.08, 0.35, 0.06])
    tb_exp = TextBox(ax_exp, "experiment", initial=experiment or "")

    def submit_id(text: str) -> None:
        text = text.strip()
        if not text:
            return
        try:
            new_id = int(text)
        except ValueError:
            print(f"Invalid id: {text}", file=sys.stderr)
            return
        render_id(new_id)

    def submit_exp(text: str) -> None:
        render_id(int(tb_id.text.strip()))

    tb_id.on_submit(submit_id)
    tb_exp.on_submit(submit_exp)

    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Filter Parquet by id and run experiments.")
    parser.add_argument("path", help="Path to the Parquet file")
    parser.add_argument("--id", type=int, required=True, help="Target id (e.g., 10015)")
    parser.add_argument(
        "--experiment",
        choices=["none", "reciprocal", "running_sum", "running_avg", "dvalue_dt"],
        default="none",
        help="Transform to apply before plotting",
    )
    parser.add_argument("--overlay", action="store_true", help="Overlay original values too")
    parser.add_argument("--size", type=float, default=6.0, help="Marker size for scatter")
    args = parser.parse_args()

    df = load_dataframe(args.path)
    view_plot(df, args.id, args.experiment, args.overlay, args.size)

if __name__ == "__main__":
    main()
