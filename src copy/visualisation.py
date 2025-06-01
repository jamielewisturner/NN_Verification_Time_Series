import matplotlib.pyplot as plt

def plot_pandas_series(series_list, labels=None, title="Plot of Pandas Series", xlabel="Date", ylabel="Value"):
    plt.figure(figsize=(12, 6))

    # Plot each series in the list
    for i, series in enumerate(series_list):
        label = labels[i] if labels and i < len(labels) else f"Series {i+1}"
        plt.plot(series.index, series.values, label=label)

    # Formatting the plot
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Show the plot
    plt.show()