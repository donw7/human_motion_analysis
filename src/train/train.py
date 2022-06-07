'''
Train Script

example usage:
---------------
python src/train/train.py -m model_path

'''

import argparse
from tensorflow import keras
from keras import layers
import pickle as pkl
import dill as dl
# time
import time
import datetime
import os

parser = argparse.ArgumentParser()

# get time now
now = datetime.datetime.now()
time_stamp = now.strftime("%Y-%m-%d_%H-%M-%S")

# defaults (assumes root dir)
OUT_DIR = os.path.join(os.getcwd(), 'models', time_stamp)
MODEL_PATH = os.path.join('models', 'templates', '1convd_5layers.h5')
X_TRAIN_PATH = os.path.join('data', 'compiled', 'x_train.pkl')
Y_TRAIN_PATH = os.path.join('data', 'compiled', 'y_train.pkl')
LEARNING_RATE = 0.01
LOSS = "mse"
EPOCHS=50
BATCH_SIZE=32
VALIDATION_SPLIT=0
CALLBACKS=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
]
SAVE_DATA = False

# optional input overrides
parser.add_argument("-o", "--outdir", dest = "outdir", default = OUT_DIR, help="out directory")
parser.add_argument("-m", "--model", dest = "modelpath", default = MODEL_PATH, help="model path")
parser.add_argument("-x", "--xtrain", dest = "xtrainpath", default = X_TRAIN_PATH, help="x train path")
parser.add_argument("-y", "--ytrain", dest = "ytrainpath", default = Y_TRAIN_PATH, help="x val path")
parser.add_argument("-lr", "--learningrate", dest = "learningrate", default = LEARNING_RATE, help="learning rate")
parser.add_argument("-l", "--loss", dest = "loss", default = LOSS, help="loss function")
parser.add_argument("-e", "--epochs", dest = "epochs", default = EPOCHS, help="epochs")
parser.add_argument("-b", "--batchsize", dest = "batchsize", default = BATCH_SIZE, help="batch size")
parser.add_argument("-vs", "--valsplit", dest = "valsplit", default = VALIDATION_SPLIT, help="validation split")
parser.add_argument("-c", "--callbacks", dest = "callbacks", default = CALLBACKS, help="callbacks")
parser.add_argument("-s", "--save", dest = "save", default = SAVE_DATA, help="save data")

args = parser.parse_args()

print("----------------------------------------------------")
print("model path: " + args.modelpath)
print("x train path: " + args.xtrainpath)
print("x val path: " + args.ytrainpath)
print("learning rate: " + str(args.learningrate))
print("loss function: " + args.loss)
print("epochs: " + str(args.epochs))
print("batch size: " + str(args.batchsize))
print("validation split: " + str(args.valsplit))
print("----------------------------------------------------")

'''---------------------------------------------------------------------------------------------------------------------
set up data, models
---------------------------------------------------------------------------------------------------------------------'''
with open(args.xtrainpath, 'rb') as f:
  x_train = pkl.load(f) 
with open(args.ytrainpath, 'rb') as f:
  y_train = pkl.load(f)

print("x_train shape: " + str(x_train.shape))
print("y_train shape: " + str(y_train.shape))

model = keras.models.load_model(args.modelpath)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.learningrate), loss=args.loss)
model.summary()


'''---------------------------------------------------------------------------------------------------------------------
train
---------------------------------------------------------------------------------------------------------------------'''
history = model.fit(
  x_train,
  y_train,
  epochs=int(args.epochs),
  batch_size=int(args.batchsize),
  validation_split=float(args.valsplit),
  callbacks=args.callbacks,
)

'''---------------------------------------------------------------------------------------------------------------------
save
---------------------------------------------------------------------------------------------------------------------'''
# mkdir
if not os.path.exists(args.outdir):
  os.makedirs(args.outdir)

# save model as h5
model.save(os.path.join(OUT_DIR, 'model.h5'))

# # save all params
# with open(os.path.join(OUT_DIR, 'params.pkl'), 'wb') as f:
#   pkl.dump(args, f)

# save all params dill
with open(os.path.join(OUT_DIR, 'params.pkl'), 'wb') as f:
  dl.dump(args, f)

# optional save data (default=False)
if args.save:
  with open(os.path.join(OUT_DIR, 'x_train.pkl'), 'wb') as f:
    pkl.dump(x_train, f)
  with open(os.path.join(OUT_DIR, 'y_train.pkl'), 'wb') as f:
    pkl.dump(y_train, f)



