import os
import random

import numpy as np
import scipy.io as sio

from tensorflow.keras.utils import to_categorical


###
# NUM_CLASSES - это что?
# features = "multi" or "DE" or "PSD"
#img_rows, img_cols, num_chan = 8, 9, 8
# как-то вставить это в функцию

# еще и нум классес и t = 1, NUMBER_OF_INPUTS


# выше то что из снаружи


# def load_data()
def load_data(dataset_dir, short_names, img_size=(8, 9, 8), number_of_inputs=1, features_type='multi', num_classes=2, frames_per_subject=4800, seed=7):
    
    img_rows, img_cols, num_chan = img_size
    
    prefixs = ['DE_s', 'PSD_s']
    
    if features_type == 'DE': prefixs = prefixs[:1]
    if features_type == 'PSD': prefixs = prefixs[1:]
    
    # all_acc = [] # используешься ли ты где-нибудь?
    
    # short_names = ['01', '02', '03', '04', '05', '06', '07', '08', 
    #               '09', '10', '11', '12', '13', '14', '15', '16', 
    #               '17', '18', '19', '20', '21', '22', '23', '24',
    #               '25', '26', '27', '28', '29', '30', '31', '32']
    
    samples_number_per_subject = int(frames_per_subject / number_of_inputs) # tested only for [1...6] range
    samples_numbers_list = list(range(samples_number_per_subject))
    
    # вроде нахуй не нужно 
    # K.clear_session()
    
    random.seed(seed)
    
    # инициализатион
    
    y_a_list = []
    y_v_list = []
    # y_a_regression_list = []
    # y_v_regression_list = []
    
    x_list = []
    
    subject_id_list = []
    
    
    # такс такс
    
    for i, short_name in enumerate(short_names):
        random.shuffle(samples_numbers_list)
        # start = time.time()
        # удолить # print("\nprocessing: ", short_name[i], "......")
        # удолить .dat??
        file_path = os.path.join(dataset_dir, prefixs[0]+short_name + '.dat')
        file = sio.loadmat(file_path)
        data = file['data']
        y_v = file['valence_labels'][0]
        y_a = file['arousal_labels'][0]
        y_v = to_categorical(y_v, num_classes)
        y_a = to_categorical(y_a, num_classes)
    
        if len(prefixs) > 0:
          for prefix in prefixs[1:]:
            # удолить .dat??
            file_path = os.path.join(dataset_dir, prefix+short_name)
            file = sio.loadmat(file_path)
            data = np.concatenate([data, file['data']], axis=1)
            
            # regression удолить тоже?
            # y_v_regression = file['valence_values'][0]
            # y_a_regression = file['arousal_values'][0]
    
        one_falx = data.transpose([0, 2, 3, 1])
        one_falx = one_falx.reshape((-1, number_of_inputs, img_rows, img_cols, num_chan))
        one_y_v = np.empty([0,2])
        one_y_a = np.empty([0,2])
    
        # регр на удоление 
        # if len(prefixs) > 0:
        #     one_y_v_regression = np.empty([0,1])
        #     one_y_a_regression = np.empty([0,1])
    
        for j in range(int(len(y_a)//number_of_inputs)):
            one_y_v = np.vstack((one_y_v, y_v[j*number_of_inputs]))
            one_y_a = np.vstack((one_y_a, y_a[j*number_of_inputs]))
            
            # и это
            # if len(prefixs) > 0:
            #     one_y_v_regression = np.vstack((one_y_v_regression, y_v_regression[j*t]))
            #     one_y_a_regression = np.vstack((one_y_a_regression, y_a_regression[j*t]))
    
        # хз что ето, вроде нигде не примелькается, удолить?
        # cvscores = []
    
        one_falx = one_falx[samples_numbers_list]
        one_y_a = one_y_a[samples_numbers_list]
        one_y_v = one_y_v[samples_numbers_list]
    
        subject_id = np.array([i] * samples_number_per_subject)
    
        y_a_list.append(one_y_a)
        y_v_list.append(one_y_v)
    
        # y_a_regression_list.append(one_y_a_regression)
        # y_v_regression_list.append(one_y_v_regression)
    
        x_list.append(one_falx)
        subject_id_list.append(subject_id)
    
    y_a_all_subject = np.concatenate(y_a_list)
    y_v_all_subject = np.concatenate(y_v_list)
    # y_a_regression_all_subject = np.concatenate(y_a_regression_list)
    # y_v_regression_all_subject = np.concatenate(y_v_regression_list)
    
    
    x_all_subject = np.concatenate(x_list)
    all_subject_id = np.concatenate(subject_id_list)
    
    return y_a_all_subject, y_v_all_subject, x_all_subject, all_subject_id
    
