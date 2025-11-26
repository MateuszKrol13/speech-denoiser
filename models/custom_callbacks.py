from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K

class LearningRateStopping(Callback):
    """
    must be called after ReduceLRonPlateau
    """
    def __init__(self, min_lr):
        super().__init__()
        self.min_lr = min_lr

    def on_train_begin(self, logs=None):
        print(f"Loading custom callback LearingRateStopping, min_lr:{self.min_lr}")

    def on_epoch_end(self, epoch, logs=None):
        lr = K.get_value(self.model.optimizer.lr)

        if lr <= self.min_lr:
            print(f"\nLearningnRateStopping: learning rate has reached the minimum threshold of {self.min_lr}, stopping training...")
            self.model.stop_training = True