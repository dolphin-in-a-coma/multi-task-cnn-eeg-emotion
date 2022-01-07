from eegemotion.data_load import load_data
from eegemotion.train import train
from eegemotion.utils import print_results

# specify dataset and model dirs
dataset_dir = '.'  # path of the folder with PSD_s and DE_s files
model_dir = '.'  # path where the model and metrics will be stored
metrics_dir = '.'  # path to the folder were all metrics will be stored

img_size = img_rows, img_cols, num_chan = 8, 9, 8  # matrix shape of input data
number_of_inputs = 1  # how many frames is taken into account during one pass

features_type = 'multi'  # 'PSD', 'DE' or 'multi' be carefull with num_chan
num_classes = 2  # number of classes of input data
frames_per_subject = 4800  # how many frames per one subject
seed = 7  # random seed

dropout_rate = .2
model_name = 'MT_CNN'  # will be a filename part
lr_decay_factor = 0.5  # multiplixity factor of lr, where it's stucked in th plateu
lr_decay_patience = 5  # how many epochs without prgress before lr_decay
epochs_n = 200  # maximum number of epochs
verbose = 0  # 0, 1 or 2

subject_n = 32

task = 'multi'  # 'valence', 'arousal' or 'multi'
fine_tuning = True  # fine tune to all subjects specifically

y_a_all_subject, y_v_all_subject, x_all_subject, all_subject_id = \
    load_data(dataset_dir, subject_n, img_size, number_of_inputs, features_type, num_classes, frames_per_subject, seed)

train(x_all_subject, y_a_all_subject, y_v_all_subject, all_subject_id, subject_n,
      dropout_rate, number_of_inputs, model_dir, metrics_dir, model_name, img_size,
      lr_decay_factor, lr_decay_patience, epochs_n, seed, verbose, task, fine_tuning)

if fine_tuning:
    scores_dict = \
        {
            'before_fine_tuning': f'{metrics_dir}/{model_name}_scores_SD_before.pkl',
            'valence_task_fine_tuning': f'{metrics_dir}/{model_name}_valence_scores_SD.pkl',
            'arousal_fine_tuning': f'{metrics_dir}/{model_name}_arousal_scores_SD.pkl',
            'multi_fine_tuning': f'{metrics_dir}/{model_name}_multi_scores_SD.pkl'
        }  # dict with paths of scores files

    print_results(scores_dict, fine_tuning)
else:
    scores_dict = {'results': f'{metrics_dir}/{model_name}_scores_SI.pkl'}  # dict with path of scores file
    print_results(scores_dict, fine_tuning)
