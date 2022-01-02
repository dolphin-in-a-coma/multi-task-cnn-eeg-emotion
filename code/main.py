from data_load import load_data 
from train import train


dataset_dir = '.'
model_dir = '.'

img_rows, img_cols, num_chan = 8, 9, 8
number_of_inputs=1

features_type='multi'
num_classes=2
frames_per_subject=4800
seed=7

dropout_rate=.2
model_name='MT_CNN'
lr_decay_factor=0.5
lr_decay_patience=5
epochs_n=200
verbose=0

# НЕ ХВАТАЕТ MULTI_TASK

short_names = ('01', '02', '03', '04', '05', '06', '07', '08', 
              '09', '10', '11', '12', '13', '14', '15', '16', 
              '17', '18', '19', '20', '21', '22', '23', '24',
              '25', '26', '27', '28', '29', '30', '31', '32')

if __name__ == '__main__':
    img_size = (img_rows, img_cols, num_chan)
    
    y_a_all_subject, y_v_all_subject, x_all_subject, all_subject_id =\
        load_data(dataset_dir, short_namesi, img_size, number_of_inputs, features_type, num_classes, frames_per_subject, seed)

    train(x_all_subject, y_a_all_subject, y_v_all_subject, all_subject_id, short_names,
          dropout_rate, number_of_inputs, model_dir, model_name, img_size, lr_decay_factor,
          lr_decay_patience, epochs_n, seed, verbose)
    
