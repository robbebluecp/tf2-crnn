import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
from tensorflow import keras



def generator(batch_size, input_shape, chars, target_len, if_print=False):
    h, w = input_shape[:2]
    while 1:

        X = []
        Y = []
        for i in range(batch_size):
            text = ''
            for i in range(random.choice([2, target_len])):
                text += random.choice(chars)
            if if_print:
                print(text)

            image = Image.new("RGB", (w, h), "white")
            draw_table = ImageDraw.Draw(im=image)
            draw_table.text(xy=(0, 0), text=text, fill='#000000', font=ImageFont.truetype('SimHei.ttf', 50))
            img = np.asarray(image)
            img = img / 255.
            X.append(img)
            Y.append([chars.find(i) for i in text])
        X = np.asarray(X)
        Y = keras.preprocessing.sequence.pad_sequences(Y, 8, padding='post', value=len(chars) + 1)
        input_length = np.ones(batch_size) * 16
        label_length = np.ones(batch_size) * 8

        yield [X, Y, input_length, label_length], np.zeros(batch_size)