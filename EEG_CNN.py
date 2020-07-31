
# Project Name: EEG Data for Mental State Detection
# Researchers: Jeffrey Chau, Apala Thakur

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model

base_dir = './'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Our input feature map is 500x500x3: 500x500 for the image pixels, and 3 for
# the three color channels: R, G, and B
img_input = Input(shape=(500, 500, 3))

# First convolution extracts 4 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window

c1 = Conv2D(4, (3,3), activation='relu', input_shape=(500, 500, 3))(img_input)
p1 = MaxPooling2D(pool_size=(2,2))(c1)

# Second convolution extracts 5 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window

c2 = Conv2D(5, (3,3), activation='relu')(p1)
p2 = MaxPooling2D(pool_size=(2,2))(c2)

# Third convolution extracts 10 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window

c3 = Conv2D(10, (3,3), activation='relu')(p2)
p3 = MaxPooling2D(pool_size=(2,2))(c3)

# Flatten feature map to a 1-dim tensor so we can add fully connected layers

f = Flatten()(p3)
# Create a fully connected (dense) layer with ReLU activation and 20 hidden units

x1 = Dense(20, activation='relu')(f)
# Create output layer with a softmax activation having three output neurons

output = Dense(3, activation='softmax')(x1)
# Create model:

model = Model(img_input, output)

# Compile the model
model.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop(lr = 0.001),
              metrics = ['acc'])



# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size = (500, 500),  # All images will be resized to 500x500
        batch_size = 5,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode = 'categorical')

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size = (500, 500),
        batch_size = 5,
        class_mode = 'categorical')

history = model.fit_generator(
      train_generator,
      steps_per_epoch = 9,  # 45 images = batch_size * steps
      epochs = 15,
      validation_data = validation_generator,
      validation_steps = 3,  # 15 images = batch_size * steps
      verbose = 2)

# Retrieve a list of accuracy results on training and test data
# sets for each training epoch


if history:
	print("History Found")
else:
	print ("No History")
acc = history.history['acc']
val_acc = history.history['val_acc']

# Retrieve a list of list results on training and test data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.legend(['train', 'test'], loc='upper left')
plt.title('Training and validation accuracy')

plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.legend(['train', 'test'], loc='lower left')
plt.title('Training and validation loss')
plt.show()
