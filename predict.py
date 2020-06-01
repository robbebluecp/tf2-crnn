import numpy as np
import models
from generator import generator

chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
char_len = len(chars)
target_len = 7
input_shape = (64, 128, 3)
batch_size = 16

base_model = models.CRNN(char_len, target_len, input_shape)(return_base=True)

base_model.load_weights('model_train/base_model.h5')

g = generator(1, input_shape, chars, target_len, if_print=True)

for [X, _, _, _], _ in g:
    pred = base_model.predict(X)
    pred = np.argmax(pred, -1)

    print(pred[0])
    for i in pred[0]:
        if i != len(chars):
            print(chars[i])
    break
