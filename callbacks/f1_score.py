"""
Callback to calculate the F1 score on the validation set after each epoch.
"""
from sklearn.metrics import f1_score
from tensorflow.keras.callbacks import Callback
import numpy as np


class F1ScoreCallback(Callback):
    """
    Callback to calculate the F1 score on the validation set after each epoch.
    """

    def __init__(self, validation_data):
        super(F1ScoreCallback, self).__init__()
        self.validation_data = validation_data
        self.validation_labels = np.argmax(validation_data[1], axis=1)

    def on_epoch_end(self, epoch, logs={}):
        """
        Calculate the F1 score on the validation set after each epoch.
        """
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), axis=1)
        f1 = f1_score(self.validation_labels, val_predict, average="macro")
        print(f"Epoch {epoch+1} - val_f1: {f1}")
