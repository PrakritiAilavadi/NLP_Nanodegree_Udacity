from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, CuDNNGRU, MaxPooling1D, Dropout)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = CuDNNGRU(units, 
                        return_sequences=True, 
                        name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    simp_rnn = CuDNNGRU(units, 
                        return_sequences=True)(input_data)
    bn_rnn = BatchNormalization()(simp_rnn)
    if recur_layers > 1:
        for i in range(recur_layers -1):
            simp_rnn = CuDNNGRU(units, 
                                return_sequences=True)(bn_rnn)
            bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn) 
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    # bidir_rnn = Bidirectional(CuDNNGRU(units, return_sequences=True))(input_data)
    #bidir_rnn = Bidirectional(LSTM(units, return_sequences=True, activation='relu'), merge_mode='concat')(input_data)
#     forward_layer = LSTM(10, return_sequences=True)
#     backward_layer = LSTM(10, activation='relu', return_sequences=True,
#                        go_backwards=True)
#     bidir_rnn = Bidirectional(CuDNNGRU(units, name='bidir_rnn'), forward_layer, backward_layer=backward_layer)(input_data)
#     # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    simp_rnn = GRU(units, return_sequences=True)
    bidir_rnn = Bidirectional(simp_rnn)(input_data)
    
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim, 
                filters, 
                kernel_size, 
                conv_stride,
                conv_border_mode, 
                units, recur_layers, 
                output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO:
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    
    #     conv_stride=1
#     conv_border_mode='same'
#     # Specify the layers in your network
#     conv_0 = Conv1D(filters, 1, strides=1, padding=conv_border_mode,
#                      activation='relu')(input_data)
#     conv_0 = BatchNormalization()(conv_0)
#     conv_1 = Conv1D(filters, 3, strides=1, padding=conv_border_mode,
#                     activation='relu')(conv_0)
#     conv_1 = BatchNormalization()(conv_1)
#     conv_2 = Conv1D(filters, 1, strides=1, padding=conv_border_mode,
#                     activation='relu')(conv_1)

    
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d) #bn
    
    mp_cnn = MaxPooling1D()(bn_cnn) #max pooling
    
    do_cnn = Dropout(0.2)(mp_cnn) # dropout
    
    # Add bidirectional recurrent layers, each with batch normalization
    bidir_rnn = Bidirectional(GRU(units, return_sequences=True))(do_cnn)
    bn_rnn = BatchNormalization()(bidir_rnn) # bn
    do_rnn = Dropout(0.2)(bn_rnn) #drop out
    if recur_layers > 1:
        for i in range(recur_layers -1):
            bidir_rnn = Bidirectional(GRU(units, return_sequences=True))(do_rnn)
            bn_rnn = BatchNormalization()(bidir_rnn)
            do_rnn = Dropout(0.2)(bn_rnn)
            
    # TODO: Add a TimeDistributedlayer
    time_dense = TimeDistributed(Dense(output_dim))(do_rnn)

    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)//2
    print(model.summary())
    return model