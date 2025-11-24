import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    data_path = os.path.join(root, "data", "demo_logs.csv")
    figs_path = os.path.join(root, "figs")

    os.makedirs(figs_path, exist_ok=True)

    df = pd.read_csv(data_path)

    # Basic summary
    print("=== Demo logs head ===")
    print(df.head())
    print("\n=== Counts by assistance style ===")
    print(df["assist_style"].value_counts())
    print("\n=== Accuracy by assistance style ===")
    df["correct"] = (df["y_true"] == df["y_hat"]).astype(int)
    print(df.groupby("assist_style")["correct"].mean())

    # Simple bar plot: accuracy by assistance style
    acc_by_style = df.groupby("assist_style")["correct"].mean().sort_values()
    plt.figure()
    acc_by_style.plot(kind="bar")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Demo accuracy by assistance style")
    plt.tight_layout()
    out_path = os.path.join(figs_path, "accuracy_by_style.png")
    plt.savefig(out_path)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
