from eegemotion.data_load import load_data 
from eegemotion.train import train


# specify dataset and model dirs
dataset_dir = '.' # path of the folder with PSD_s and DE_s files
model_dir = '.' # path where the model and metrics will be stored 


img_size = img_rows, img_cols, num_chan = 8, 9, 8 # matrix shape of input data
number_of_inputs=2 # how many frames is taken into account during one pass

features_type='multi' # 'PSD', 'DE' or 'multi' 
num_classes=2 # number of classes of input data
frames_per_subject=4800 # how many frames per one subject
seed=7 # random seed

dropout_rate=.2
model_name='MT_CNN' # will be a filename part
lr_decay_factor=0.5 # multiplixity factor of lr, where it's stucked in th plateu
lr_decay_patience=5 # how many epochs without prgress before lr_decay
epochs_n=200 # maximum number of epochs
verbose=0 # 0, 1 or 2

# НЕ ХВАТАЕТ MULTI_TASK и файн-тюн
# и еще чего-то не хватает

short_names = ('01', '02', '03', '04', '05', '06', '07', '08', 
              '09', '10', '11', '12', '13', '14', '15', '16', 
              '17', '18', '19', '20', '21', '22', '23', '24',
              '25', '26', '27', '28', '29', '30', '31', '32') # наверное нужно поменять


y_a_all_subject, y_v_all_subject, x_all_subject, all_subject_id =\
    load_data(dataset_dir, short_names, img_size, number_of_inputs, features_type, num_classes, frames_per_subject, seed)
    
train(x_all_subject, y_a_all_subject, y_v_all_subject, all_subject_id, short_names,
      dropout_rate, number_of_inputs, model_dir, model_name, img_size, lr_decay_factor,
      lr_decay_patience, epochs_n, seed, verbose)

# какая-то залупа с памятью