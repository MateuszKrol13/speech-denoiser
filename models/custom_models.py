import tensorflow as tf

def cnn_ced(model_name: str = "CNN-CED", show_model: bool = False) -> tf.keras.Sequential:
    """
    Fully convolutional network, based on https://arxiv.org/pdf/1609.07132 .

    Layers
    ------
        - (Conv1D, ReLU, BN) x 15
        - (Conv1D, ReLU)

    Kernels
    -------
         - Count: [18, 30, 8] x 5
         - Sizes: [[9], [5], [9]] x 5

    Optimizer
    ---------
        - Adam

    Returns
    -------
        tf.keras.Sequential
    """

    m = tf.keras.Sequential(name=model_name)
    m.add(tf.keras.layers.Input((129, 8)))
    filter_counts = [18, 30, 8] * 5
    filter_widths = [9, 5, 9] * 5

    for i in range(len(filter_widths)):
        m.add(tf.keras.layers.Conv1D(filter_counts[i], kernel_size=filter_widths[i], activation='relu', padding='same'))
        m.add(tf.keras.layers.BatchNormalization())

    m.add(tf.keras.layers.Conv1D(1, 1, activation='relu'))
    m.compile(optimizer='adam', loss="MeanSquaredError", metrics=[])

    if show_model:
        m.summary()

    return m

def mlp():
    m = tf.keras.Sequential()

    m.add(tf.keras.layers.Input(800))  # 100ms, fs=8kHz -> 800
    for i in range(5):
        m.add(tf.keras.layers.Dense(800/ (i +1), activation='linear'))
        m.add(tf.keras.layers.BatchNormalization(axis=-1))

    m.add(tf.keras.layers.Dense(800, activation='linear'))
    m.compile(optimizer='adam', loss="MeanSquaredError", metrics=[])
    m.summary()