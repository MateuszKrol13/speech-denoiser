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

if "__main__" == __name__:

