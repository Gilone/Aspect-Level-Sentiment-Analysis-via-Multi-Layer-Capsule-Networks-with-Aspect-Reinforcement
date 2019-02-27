import keras.backend as K
from keras import  layers
import tensorflow as tf

class Length(layers.Layer):

    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


class Mask(layers.Layer):

    def __init__(self,  rank = 1, reverse = False, **kwargs):
        super(Mask, self).__init__(**kwargs)
        self.rank = rank
        self.reverse = reverse

    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            assert len(inputs) == 2
            inputs, mask = inputs
        else:
            x = K.sqrt(K.sum(K.square(inputs), -1))
            mask = K.one_hot(indices=tf.contrib.framework.argsort(x,axis=1, direction='DESCENDING', stable=True)[:,self.rank-1],
                             num_classes=x.get_shape().as_list()[1])
        if self.reverse == False:
            masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        else:
            masked = K.batch_flatten(inputs * (K.ones_like(K.expand_dims(mask, -1))-K.expand_dims(mask, -1)))
        return masked

    def compute_output_shape(self, input_shape):
        # true label provided
        if type(input_shape[0]) is tuple:
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        # true label not provided
        else:
            return tuple([None, input_shape[1] * input_shape[2]])

def Slice(dimension, start, end):

    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return layers.Lambda(func)