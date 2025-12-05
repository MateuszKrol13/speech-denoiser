import matplotlib.pyplot as plt
import matplotlib
import librosa
import numpy as np

from denoiser.config import DATA_ROOT

matplotlib.use("TkAgg")

def crop(signal, sr: int, enegry_threshold: float = 0.9999):
    step_size = sr // 100  # 10 ms
    signal_energy = sum(signal ** 2)
    energy_tmp = 0
    first, last = 0, len(signal)
    # front
    while (signal_energy - energy_tmp) / signal_energy > enegry_threshold:
        first += step_size
        energy_tmp = sum(signal[:first] ** 2)

    signal_energy = sum(signal[first:] ** 2)
    energy_tmp = 0
    # back
    while (signal_energy - energy_tmp) / signal_energy > enegry_threshold:
        last -= step_size
        energy_tmp = sum(signal[last:] ** 2)

    return signal[first:last], first, last

def pie_chart(lenghts, cropped_lengths):
    sizes = [sum(cropped_lengths) / sum(lenghts), (sum(lenghts) - sum(cropped_lengths)) / sum(lenghts)]
    labels = ["Sygnał mowy", "Cisza"]

    plt.figure(figsize=(4, 4))
    plt.pie(sizes, labels=labels, autopct='%1.0f%%', startangle=90)
    plt.title("Pie Chart Example")
    plt.tight_layout()
    plt.show()


def plot():
    demo_pth = DATA_ROOT / "demos/demo1/audio_demo.mp3"
    y, sr = librosa.load(demo_pth, sr=32_000)
    y_c, f, l = crop(y, sr=32_000, enegry_threshold=0.9999)
    fs = 32_000
    t = np.arange(len(y)) / fs
    start_t = f / fs
    end_t = l / fs
    plt.figure(figsize=(10, 4))
    plt.plot(t, y, linewidth=1)
    plt.axvspan(start_t, end_t, facecolor='green', alpha=0.3)
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.tight_layout()
    plt.show()

def trim():
    pass

def eval_plots():
    # Example data: each subplot has its own set of three values
    data = [
        [1.5387, 1.6811, 1.9526],
        [0.6901, 0.6574, 0.7552],
        [-0.0019, 5.7417, 10.2063]
    ]

    labels = ["Szum", "Z-norm", "Min-Max"]
    metric = ["PESQ", "STOI", "SI-SDR"]
    colors = ["#6B7A8F", "#384655", "#506070"]

    x = np.arange(3) * 0.25  # keep bars close together
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)

    for j in range(3):
        ax = axes[j]
        values = data[j]

        # plot thin bars using normalized values
        ax.grid(True, axis='y', linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.bar(x, values, color=colors, width=0.2)

        # add actual values above bars
        for i in range(3):
            ax.text(
                x[i],
                values[i] * 1.03,
                f"{values[i]}",
                ha="center",
                va="bottom"
            )

        # cosmetic settings
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylim(0, max(values) * 1.2)
        ax.set_title(metric[j])
    plt.tight_layout()
    plt.show()

if "__main__" == __name__:
    pass