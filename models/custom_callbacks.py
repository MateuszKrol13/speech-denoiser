from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K

class LearningRateStopping(Callback):
    def __init__(self, min_lr):
        super().__init__()
        self.min_lr = min_lr

    def on_train_begin(self, logs=None):
        print(f"Loading custom callback LearingRateStopping, min_lr:{self.min_lr}")

    def on_epoch_begin(self, epoch, logs=None):
        lr = K.get_value(self.model.optimizer.lr)

        if lr <= self.min_lr:
            print(f"\nStopping training: learning rate has reached the minimum threshold of {self.min_lr}.")
            self.model.stop_training = True