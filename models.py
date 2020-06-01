from tensorflow import keras
from tensorflow.keras import backend as K



class CRNN:
    def __init__(self,
                 char_len: int,
                 target_len: int,
                 input_shape: tuple or list = (64, 128, 3)):
        self.char_len = char_len
        self.target_len = target_len
        self.input_tensor = keras.layers.Input(input_shape)

    def __call__(self, return_base=False, *args, **kwargs):
        return self.get_crnn(return_base)

    @staticmethod
    def ctc_lambda_func(args):
        y_true, y_pred, input_length, label_length = args
        return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

    def get_crnn(self, return_base=False):
        x = self.input_tensor
        for i, n_cnn in enumerate([2, 2, 2, 2, 2]):
            for j in range(n_cnn):
                x = keras.layers.Conv2D(32 * 2 ** min(i, 3), kernel_size=3, padding='same', kernel_initializer='he_uniform')(x)
                x = keras.layers.BatchNormalization()(x)
                x = keras.layers.Activation('relu')(x)
            x = keras.layers.MaxPooling2D(2 if i < 3 else (2, 1))(x)

        x = keras.layers.Permute((2, 1, 3))(x)
        x = keras.layers.TimeDistributed(keras.layers.Flatten())(x)

        x = keras.layers.Bidirectional(keras.layers.GRU(128, return_sequences=True))(x)
        x = keras.layers.Bidirectional(keras.layers.GRU(128, return_sequences=True))(x)
        y = keras.layers.Dense(self.char_len + 1, activation='softmax')(x)
        base_model = keras.models.Model(inputs=self.input_tensor, outputs=y)
        if return_base:
            return base_model

        y_true = keras.layers.Input(name='y_true', shape=[self.target_len + 1], dtype='float32')
        # (1, )
        input_length = keras.layers.Input(name='input_length', shape=[1], dtype='int64')
        # (1, )
        label_length = keras.layers.Input(name='label_length', shape=[1], dtype='int64')
        loss_out = keras.layers.Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')(
            [y_true, base_model.outputs[-1], input_length, label_length])

        final_model = keras.models.Model(inputs=[self.input_tensor, y_true, input_length, label_length],
                                         outputs=loss_out)
        return base_model, final_model
