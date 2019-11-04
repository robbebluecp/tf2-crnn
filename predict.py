import keras
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from keras.preprocessing.sequence import pad_sequences


chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


model = keras.models.load_model('base_model.h5')

X = []
Y = []
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
print(X.shape)
print(Y.shape)

pred = model.predict(X)
pred = np.argmax(pred, -1)
for i in pred[0]:
    if i != len(chars) + 1:
        print(chars.find(i))