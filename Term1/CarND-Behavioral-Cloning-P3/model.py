import numpy as np
import process_data
from matplotlib import pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Conv2D
from keras.layers import Dropout, Flatten, Dense, BatchNormalization, Lambda
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

#########################################################################################

# Initialize the file path of the image dataset
udacity_data, my_data = process_data.initialize_file_paths()

# read in the training data and split into train and validation
train_data, validation_data = process_data.process_csv(udacity_data, my_data)

# Prepare data for the data generators
train_data = process_data.uniform_split_data(train_data)

# Call the data generators on the training and validation data
train_generator = process_data.data_generator(train_data)
validation_generator = process_data.data_generator(validation_data)


# Build the NVIDIA network in keras framework

###################################MODEL###############################################

model = Sequential()

# Image normalization using the Lambda layer
model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(80,320,3)))

# After reading on many forums, I sandwiched BatchNormalization layer between
# the Convolution and Activation layer

model.add(BatchNormalization())

# Add 5 Convolution layers with filter size 5x5 and 4x4 for the first 3 and last 2 resp.
# Pair it with BatchNormalization

model.add(Conv2D(24, (5, 5), strides=[2,2], activation='elu'))
model.add(BatchNormalization())
model.add(Conv2D(36, (5, 5), strides=[2,2], activation='elu'))
model.add(BatchNormalization())
model.add(Conv2D(48, (5, 5), strides=[2,2], activation='elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='elu'))
model.add(BatchNormalization())

# Dropout for regularisation
model.add(Dropout(rate=0.5))

# Flatten the output and use 3 FC Layers
model.add(Flatten())
model.add(Dense(units=100, activation='elu'))
model.add(Dense(units=50, activation='elu'))
model.add(Dense(units=10, activation='elu'))
model.add(Dense(units=1))

# Print the network summary( very helpful! )
model.summary()

# Initialize Adam with learning decay
_optimizer = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6, amsgrad=False)

# Compile the model using Mean Squared Error as loss function ( as given in the NVIDIA paper )
model.compile(loss='mse', optimizer=_optimizer, metrics=['mse'])

# checkpoint and save best model
model_path = "model_kk.h5"
checkpoint = ModelCheckpoint(model_path, verbose=1, save_best_only=True)
callbacks_list = [checkpoint]

# Compute the step per epoch values for training and validation  data
steps_per_epoch_fit = np.ceil(len(train_data) / process_data.BATCH_SIZE)
steps_per_epoch_val = np.ceil(len(validation_data) / process_data.BATCH_SIZE)

# Fit the model
history = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch_fit, validation_data=validation_generator,
                    validation_steps=steps_per_epoch_val, epochs=50, callbacks=callbacks_list)

# list all data in history
#print(history.history.keys())

# Doesnt work on the AWS

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Added to avoid some session related error, just a trick!
from keras import backend as K
K.clear_session()