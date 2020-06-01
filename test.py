from PIL import Image, ImageDraw, ImageFont
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


batch_size = 16
X = []
Y = []
for i in range(batch_size):
    text = ''
    for i in range(random.choice([2, 7])):
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
print(X.shape, Y.shape)
input_length = np.ones(batch_size) * 16
label_length = np.ones(batch_size) * 8
print(input_length.shape, label_length.shape)