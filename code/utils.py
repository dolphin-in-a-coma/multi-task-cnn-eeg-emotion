import numpy as np
from sklearn.utils import class_weight

from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.platform import tf_logging as logging


class ReduceLRBacktrack(ReduceLROnPlateau):
    def __init__(self, best_path, *args, **kwargs):
        super(ReduceLRBacktrack, self).__init__(*args, **kwargs)
        self.best_path = best_path

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)

        #print(log.keys())
        if current is None:
            logging.warning('Reduce LR on plateau conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                             self.monitor, ','.join(list(logs.keys())))
        if not self.monitor_op(current, self.best): # not new best
            if not self.in_cooldown(): # and we're not in cooldown
                if self.wait+1 >= self.patience: # going to reduce lr
                    # load best model so far
                    if self.verbose:
                        print("Backtracking to best model before reducting LR")
                    self.model.load_weights(self.best_path)

        super().on_epoch_end(epoch, logs) # actually reduce LR


def get_sample_weights(list_of_y_trains):

    sample_weights = np.ones(list_of_y_trains[0].shape[0])

    for y_train in list_of_y_trains:
        y_ints = y_train.argmax(1)
        class_weights = class_weight.compute_class_weight('balanced',
                                                      classes=np.unique(y_ints),
                                                      y=y_ints)
    for i in np.unique(y_ints):
        sample_weights[y_ints == i] =\
            sample_weights[y_ints == i] * class_weights[i]

    return sample_weights