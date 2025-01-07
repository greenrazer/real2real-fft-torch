import matplotlib as plt

def visualize_signals(original_signal, reconstructed_signal, title="Signal Comparison"):
    """
    Visualize the original and reconstructed signals on top of each other.

    Args:
        original_signal (torch.Tensor): Original signal of shape (N, 2).
        reconstructed_signal (torch.Tensor): Reconstructed signal of shape (N, 2).
        title (str): Title for the plot.
    """
    # Ensure the signals are numpy arrays for plotting
    original_signal = original_signal.cpu().numpy()
    reconstructed_signal = reconstructed_signal.cpu().numpy()

    # Create the figure
    plt.figure(figsize=(12, 6))

    # Plot channel 1
    plt.subplot(1, 1, 1)
    plt.plot(original_signal, label="Original Signal (Ch 1)", alpha=0.7)
    plt.plot(
        reconstructed_signal,
        label="Reconstructed Signal (Ch 1)",
        linestyle="dashed",
        alpha=0.7,
    )
    plt.title(f"{title} - Channel 1")
    plt.legend()
    plt.grid(True)

    # Show the plots
    plt.tight_layout()
    plt.show()