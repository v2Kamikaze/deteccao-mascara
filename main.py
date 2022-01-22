"""
    install keras, tensorflow, pillow, matplotlib
"""

import os

#  Para ignorar os warnings, caso não seja possível usar a GPU com o tensorflow.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras import Sequential
from keras_preprocessing.image.directory_iterator import DirectoryIterator
from keras_preprocessing.image.image_data_generator import ImageDataGenerator



EPOCHS: int = 10
PATH = "./dataset"
BATCH_SIZE: int = 16
UNITS: int = 256  # 256, 128, 64

data_generator: ImageDataGenerator = ImageDataGenerator(
    rescale=1/255, validation_split=0.3)

# Dados para treino
train_generator: DirectoryIterator = data_generator.flow_from_directory(
    directory=PATH,
    shuffle=True,
    seed=9,
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    subset="training",
)

# Dados para validação
validation_generator: DirectoryIterator = data_generator.flow_from_directory(
    directory=PATH,
    shuffle=True,
    seed=9,
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    subset="validation",
)

net_model: Sequential = Sequential()
net_model.add(
    Conv2D(
        filters=64,
        kernel_size=2,
        activation="relu",
        input_shape=(256, 256, 3),
    )
)

net_model.add(MaxPool2D(pool_size=(2, 2)))
net_model.add(Dropout(0.3))

net_model.add(
    Conv2D(
        filters=128,
        kernel_size=2,
        activation="relu",
    )
)

net_model.add(MaxPool2D(pool_size=(2, 2)))
net_model.add(Dropout(0.3))

net_model.add(Flatten())

net_model.add(Dense(units=UNITS, activation="relu"))
net_model.add(Dropout(0.5))

# Duas classes [com ou sem máscara] e sigmoid para dar a probabilidade de acerto.
net_model.add(Dense(units=2, activation="sigmoid"))

net_model.summary()

net_model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

checkpoint: ModelCheckpoint = ModelCheckpoint(
    filepath="mask_model.hdf5",
    monitor="val_loss",
    verbose=1,
    mode="min",
    save_best_only=True,

)

early_stop: EarlyStopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=5,
    verbose=1,
    mode="min",
)

net_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples/BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples/BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop],
)

"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 255, 255, 64)      832

 max_pooling2d (MaxPooling2D  (None, 127, 127, 64)     0
 )

 dropout (Dropout)           (None, 127, 127, 64)      0

 conv2d_1 (Conv2D)           (None, 126, 126, 128)     32896

 max_pooling2d_1 (MaxPooling  (None, 63, 63, 128)      0
 2D)

 dropout_1 (Dropout)         (None, 63, 63, 128)       0

 flatten (Flatten)           (None, 508032)            0

 dense (Dense)               (None, 256)               130056448

 dropout_2 (Dropout)         (None, 256)               0

 dense_1 (Dense)             (None, 2)                 514

=================================================================
"""
