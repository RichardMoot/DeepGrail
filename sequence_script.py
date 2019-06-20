import numpy as np

from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Bidirectional, Masking, Dense, Input, Dropout, LSTM, Activation, TimeDistributed, BatchNormalization, concatenate, Concatenate
from keras.layers.embeddings import Embedding
from keras.constraints import max_norm
from keras import regularizers
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, TensorBoard
from keras.utils import to_categorical
from keras import backend as K
from my_classes import DataGenerator

np.random.seed(1)

params = {'n_pos1_classes':30,
          'n_pos2_classes':31,
          'n_super_classes':891,
          'shuffle':True,
          'batch_size':32}

all = ["sent%06d" %i for i in range(15748)]
train, testdev = train_test_split(all, test_size=0.4)
test, dev = train_test_split(testdev, test_size=0.5)

print("Train: "+str(len(train)))
print("Dev  : "+str(len(dev)))
print("Test : "+str(len(test)))

training_generator = DataGenerator(train, **params)
validation_generator = DataGenerator(dev, **params)

embLen = 1024
numPos1Classes = 30
numPos2Classes = 31
numSuperClasses = 891

sentence_embeddings = Input(shape = (None,embLen,), dtype = 'float32')
mask = Masking(mask_value=0.0)(sentence_embeddings)
X = Dropout(0.5)(mask)

# first bi-directional LSTM layer 

X = Bidirectional(LSTM(128, recurrent_dropout=0.2, kernel_constraint=max_norm(5.), return_sequences=True))(X)
X = BatchNormalization()(X)
X = Dropout(0.2)(X)

# Pos1 output

Pos1 = TimeDistributed(Dense(32,kernel_constraint=max_norm(5.)))(X)
Pos1 = TimeDistributed(Dropout(0.2))(Pos1)
Pos1 = TimeDistributed(Dense(numPos1Classes, name='pos1', activation='softmax',kernel_constraint=max_norm(5.)))(Pos1)

# Pos2 output

Pos2 = TimeDistributed(Dense(32,kernel_constraint=max_norm(5.)))(X)
Pos2 = TimeDistributed(Dropout(0.2))(Pos2)
Pos2 = TimeDistributed(Dense(numPos2Classes, name='pos2', activation='softmax',kernel_constraint=max_norm(5.)))(Pos2)

# second bi-directional LSTM layer

X = Bidirectional(LSTM(128, recurrent_dropout=0.25, kernel_constraint=max_norm(5.), return_sequences=True))(X)
X = BatchNormalization()(X)
X = Dropout(0.25)(X)

# supertag output

X = TimeDistributed(Dense(32,kernel_constraint=max_norm(5.)))(X)
X = TimeDistributed(Dropout(0.25))(X)
X = TimeDistributed(Dense(numSuperClasses, name='super', activation='softmax',kernel_constraint=max_norm(5.)))(X)

model = Model(sentence_embeddings, [Pos1,Pos2,X])

model.summary()

model.compile(loss=['categorical_crossentropy','categorical_crossentropy','categorical_crossentropy'], optimizer='rmsprop', metrics=['accuracy'])


history = model.fit_generator(training_generator,\
                              epochs=30, shuffle=True, workers=2, validation_data=validation_generator)

