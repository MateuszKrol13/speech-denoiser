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

def cnn_unet(model_name: str = "CNN-CED", show_model: bool = False) -> tf.keras.Sequential:
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
    # input layer
    input = tf.keras.Input(shape=(129, 8))
    x = input
    block_outputs = []

    # Build the blocks
    for block_idx in range(0, 5):

        if block_idx == 3:
            x = tf.keras.layers.Add()([x, block_outputs[1]])
        if block_idx == 4:
            x = tf.keras.layers.Add()([x, block_outputs[0]])

        l1 = tf.keras.layers.Conv1D(filters=18, kernel_size=9, padding='same', activation='relu', name="l1-b"+str(block_idx))(x)
        l1 = tf.keras.layers.BatchNormalization(name="l1-b"+str(block_idx)+"-bn")(l1)
        l2 = tf.keras.layers.Conv1D(filters=30, kernel_size=5, padding='same', activation='relu', name="l2-b"+str(block_idx))(l1)
        l2 = tf.keras.layers.BatchNormalization(name="l2-b"+str(block_idx)+"-bn")(l2)
        l3 = tf.keras.layers.Conv1D(filters=8, kernel_size=9, padding='same', activation='relu', name="l3-b"+str(block_idx))(l2)
        x = tf.keras.layers.BatchNormalization(name="l3-b"+str(block_idx)+"-bn")(l3)

        block_outputs.append(x)



    # Final Conv1D layer
    outputs = tf.keras.layers.Conv1D(1, 1, activation='relu')(x)

    # Build model
    model = tf.keras.Model(inputs=input, outputs=outputs, name="residual_cnn_1d")
    model.summary()

    return model
