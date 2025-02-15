from tensorflow.keras.layers import SeparableConv2D as KerasSeparableConv2D

class SeparableConv2DWrapper(KerasSeparableConv2D):
    def __init__(self, *args, **kwargs):
        # Remove unsupported keyword arguments
        kwargs.pop('groups', None)
        kwargs.pop('kernel_initializer', None)
        kwargs.pop('kernel_regularizer', None)
        kwargs.pop('kernel_constraint', None)
        super().__init__(*args, **kwargs)