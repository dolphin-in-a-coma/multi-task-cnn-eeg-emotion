import os
import random

import numpy as np
import scipy.io as sio

from tensorflow.keras.utils import to_categorical


def load_data(dataset_dir, subject_n=32, img_size=(8, 9, 8), number_of_inputs=1,
              features_type='multi', num_classes=2, frames_per_subject=4800, seed=7):
    img_rows, img_cols, num_chan = img_size

    prefixs = ['DE_s', 'PSD_s']

    if features_type == 'DE':
        prefixs = prefixs[:1]
    elif features_type == 'PSD':
        prefixs = prefixs[1:]
    elif features_type != 'multi':
        raise NotImplementedError()

    samples_number_per_subject = int(frames_per_subject / number_of_inputs)  # tested only for [1...6] range
    samples_numbers_list = list(range(samples_number_per_subject))

    random.seed(seed)

    y_a_list = []
    y_v_list = []

    x_list = []

    subject_id_list = []

    for i in range(subject_n):
        short_name = f'{i + 1:02}'
        random.shuffle(samples_numbers_list)
        print("\nprocessing: ", short_name, "......")
        file_path = os.path.join(dataset_dir, prefixs[0] + short_name)
        file = sio.loadmat(file_path)
        data = file['data']
        y_v = file['valence_labels'][0]
        y_a = file['arousal_labels'][0]
        y_v = to_categorical(y_v, num_classes)
        y_a = to_categorical(y_a, num_classes)

        if len(prefixs) > 0:
            for prefix in prefixs[1:]:
                file_path = os.path.join(dataset_dir, prefix + short_name)
                file = sio.loadmat(file_path)
                data = np.concatenate([data, file['data']], axis=1)

        one_falx = data.transpose([0, 2, 3, 1])
        one_falx = one_falx.reshape((-1, number_of_inputs, img_rows, img_cols, num_chan))
        one_y_v = np.empty([0, 2])
        one_y_a = np.empty([0, 2])

        for j in range(int(len(y_a) // number_of_inputs)):
            one_y_v = np.vstack((one_y_v, y_v[j * number_of_inputs]))
            one_y_a = np.vstack((one_y_a, y_a[j * number_of_inputs]))

        one_falx = one_falx[samples_numbers_list]
        one_y_a = one_y_a[samples_numbers_list]
        one_y_v = one_y_v[samples_numbers_list]

        subject_id = np.array([i] * samples_number_per_subject)

        y_a_list.append(one_y_a)
        y_v_list.append(one_y_v)

        x_list.append(one_falx)
        subject_id_list.append(subject_id)

    y_a_all_subject = np.concatenate(y_a_list)
    y_v_all_subject = np.concatenate(y_v_list)

    x_all_subject = np.concatenate(x_list)
    all_subject_id = np.concatenate(subject_id_list)

    return y_a_all_subject, y_v_all_subject, x_all_subject, all_subject_id
