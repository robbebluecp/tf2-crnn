import models
from tensorflow import keras
from generator import generator


chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
char_len = len(chars)
target_len = 7
input_shape = (64, 128, 3)
batch_size = 16


data_gen_train = generator(batch_size, input_shape, chars, target_len)
data_gen_valid = generator(batch_size, input_shape, chars, target_len)

checkpoint = keras.callbacks.ModelCheckpoint(filepath='model_train/ep{epoch:03d}-loss{loss:.3f}.h5',
                                             monitor='loss',
                                             save_weights_only=False,
                                             save_best_only=True)
lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=1)
base_model, final_model = models.CRNN(char_len, target_len, input_shape)()
final_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=keras.optimizers.Adam(1e-4, amsgrad=True))
final_model.fit(data_gen_train,
                epochs=50,
                steps_per_epoch=100,
                # extract validation_steps * batch_size samples to validate
                validation_steps=5,
                validation_data=data_gen_valid,
                callbacks=[checkpoint, lr]
                )


