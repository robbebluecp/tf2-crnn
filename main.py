from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np
from keras.preprocessing.sequence import pad_sequences

chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def generator(batch_size):
    while 1:
        X = []
        Y = []
        for i in range(batch_size):
            text = ''
            for i in range(random.choice([4, 5])):
                text += random.choice(chars)

            image = Image.new("RGB", (128, 64), "white")
            draw_table = ImageDraw.Draw(im=image)
            draw_table.text(xy=(0, 0), text=text, fill='#000000', font=ImageFont.truetype('SimHei.ttf', 50))
            img = np.asarray(image)
            img = img / 255.
            X.append(img)
            Y.append([chars.find(i) for i in text])
        X = np.asarray(X)
        Y = pad_sequences(Y, 8, padding='post', value=len(chars) + 1)
        input_length = np.ones(batch_size) * 16
        label_length = np.ones(batch_size) * 8
        yield [X, Y, input_length, label_length], np.zeros(batch_size)


from keras.models import *
from keras.layers import *
from keras.optimizers import *


def ctc_lambda_func(args):
    y_true,  y_pred, input_length, label_length = args
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)



input_tensor = Input((64, 128, 3))
x = input_tensor
for i, n_cnn in enumerate([2, 2, 2, 2, 2]):
    for j in range(n_cnn):
        x = Conv2D(32 * 2 ** min(i, 3), kernel_size=3, padding='same', kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x = MaxPooling2D(2 if i < 3 else (2, 1))(x)

x = Permute((2, 1, 3))(x)
x = TimeDistributed(Flatten())(x)

rnn_size = 128
x = Bidirectional(GRU(rnn_size, return_sequences=True))(x)
x = Bidirectional(GRU(rnn_size, return_sequences=True))(x)
y = Dense(len(chars) + 1, activation='softmax')(x)
base_model = Model(inputs=input_tensor, outputs=y)


y_true = Input(name='the_labels', shape=[8], dtype='float32')
# (1, )
input_length = Input(name='input_length', shape=[1], dtype='int64')
# (1, )
label_length = Input(name='label_length', shape=[1], dtype='int64')
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_true, y, input_length, label_length])

model = Model(inputs=[input_tensor, y_true, input_length, label_length], outputs=loss_out)

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(1e-3, amsgrad=True))
model.fit_generator(generator=generator(16),
                    epochs=100,
                    steps_per_epoch=100,
                    validation_steps=10,
                    validation_data=generator(4),
                    )
base_model.save('base_model.h5')
