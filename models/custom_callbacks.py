from tensorflow.keras.callbacks import Callback, TensorBoard
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
            print(f"\nStopping training: learning rate {lr} has reached the minimum threshold {self.min_lr}.")
            self.model.stop_training = True


class LearningRateLogger(Callback):
    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or "learning_rate" in logs:
            return
        logs["learning_rate"] = K.eval(self.model.optimizer.lr)