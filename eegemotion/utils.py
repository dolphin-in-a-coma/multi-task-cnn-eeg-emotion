import pickle

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

        if current is None:
            logging.warning('Reduce LR on plateau conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                            self.monitor, ','.join(list(logs.keys())))
        if not self.monitor_op(current, self.best):  # not new best
            if not self.in_cooldown():  # and we're not in cooldown
                if self.wait + 1 >= self.patience:  # going to reduce lr
                    # load best model so far
                    if self.verbose:
                        print("Backtracking to best model before reducting LR")
                    self.model.load_weights(self.best_path)

        super().on_epoch_end(epoch, logs)  # actually reduce LR


def get_sample_weights(list_of_y_trains):
    sample_weights = np.ones(list_of_y_trains[0].shape[0])

    for y_train in list_of_y_trains:
        y_ints = y_train.argmax(1)
        class_weights = class_weight.compute_class_weight('balanced',
                                                          classes=np.unique(y_ints),
                                                          y=y_ints)
    for i in np.unique(y_ints):
        sample_weights[y_ints == i] = \
            sample_weights[y_ints == i] * class_weights[i]

    return sample_weights


def print_results(scores_dict, fine_tuning):
    if fine_tuning:
        valence_avg_scores_list = []
        valence_std_scores_list = []
        arousal_avg_scores_list = []
        arousal_std_scores_list = []

        for scores_name, scores_path in scores_dict.items():

            with open(scores_path, 'rb') as fl:
                scores = pickle.load(fl)

            a_acc_list = []
            v_acc_list = []
            a_std_list = []
            v_std_list = []

            for subject_scores in scores:

                a_list = []
                v_list = []
                for valence, arousal in subject_scores:
                    a_list.append(arousal * 100)
                    v_list.append(valence * 100)
                a_acc_list.append(np.mean(a_list))
                a_std_list += a_list
                v_acc_list.append(np.mean(v_list))
                v_std_list += v_list

            print(f'\n{scores_name}')
            print(f'\tArousal\n\t\tmean: {np.mean(a_acc_list):3.4f} %\n\t\tstd: {np.std(a_std_list):3.4f} %')
            print(f'\tValence\n\t\tmean: {np.mean(v_acc_list):3.4f} %\n\t\tstd: {np.std(v_std_list):3.4f} %')
            valence_avg_scores_list.append(np.mean(v_acc_list))
            valence_std_scores_list.append(np.std(v_std_list))
            arousal_avg_scores_list.append(np.mean(a_acc_list))
            arousal_std_scores_list.append(np.std(a_std_list))
    else:
        for scores_name, scores_path in scores_dict.items():
            with open(scores_path, 'rb') as fl:
                scores = pickle.load(fl)

            valence_score = np.mean([score[-2] for score in scores]) * 100
            arousal_score = np.mean([score[-1] for score in scores]) * 100

            print(f'\n{scores_name}')
            print(f'\tArousal mean: {arousal_score:3.4f} %')
            print(f'\tValence mean: {valence_score:3.4f} %')
