from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.layers import Flatten, Dense, Reshape, Average
from keras.models import Sequential, Model


def create_base_network(input_dim, dropout_rate):
    seq = Sequential()
    seq.add(Conv2D(64, 5, activation='relu', padding='same', name='conv1', input_shape=input_dim))
    if True: seq.add(BatchNormalization())
    seq.add(Dropout(dropout_rate))
    seq.add(Conv2D(128, 4, activation='relu', padding='same', name='conv2'))
    if True: seq.add(BatchNormalization())
    seq.add(Dropout(dropout_rate))
    seq.add(Conv2D(256, 4, activation='relu', padding='same', name='conv3'))
    if True: seq.add(BatchNormalization())
    seq.add(Dropout(dropout_rate))
    seq.add(Conv2D(64, 1, activation='relu', padding='same', name='conv4'))
    seq.add(MaxPooling2D(2, 2, name='pool1'))
    if True: seq.add(BatchNormalization())
    seq.add(Dropout(dropout_rate))
    seq.add(Flatten(name='fla1'))
    seq.add(Dense(512, activation='relu', name='dense1'))
    seq.add(Reshape((1, 512), name='reshape'))
    if True: seq.add(BatchNormalization())
    seq.add(Dropout(dropout_rate))

    return seq


def create_MT_CNN(img_size=(8, 9, 8), dropout_rate=0.2, number_of_inputs=1):

    base_network = create_base_network(img_size, dropout_rate)

    inputs = [Input(shape=img_size) for i in range(number_of_inputs)]

    if number_of_inputs == 1:
        x = base_network(inputs[0])
    else:
        x = Average()([base_network(input_) for input_ in inputs])

    x = Flatten(name='flat')(x)

    out_v = Dense(2, activation='softmax', name='out_v')(x)
    out_a = Dense(2, activation='softmax', name='out_a')(x)

    model = Model(inputs, [out_v, out_a])

    return model
