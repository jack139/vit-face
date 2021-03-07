import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import random
from keras.optimizers import *
from keras.callbacks import *
import skimage.io as io
import skimage.transform as trans

from transformer import VisionTransformer

from keras.datasets import cifar10

input_size = (128,128)
batch_size = 64
epochs = 30

layers = 8
d_model = 256   # Hidden_size_D
n_head = 16
mlp_dim = 128   # MLP_size
patch_size = 8
dropout = 0.2 

'''
Model  Layers Hidden_size_D MLP_size    Heads   Params
Base    12      768         3072        12      86M
Large   24      1024        4096        16      307M
Huge    32      1280        5120        16      632M
'''

train_dir = '../datasets/AFDB_face_dataset'
#test_dir = '../datasets/AFDB_face_dataset'

# label 数量
classes_num = len(os.listdir(train_dir))

# 数据生成器
train_datagen = ImageDataGenerator(
    preprocessing_function=lambda x: x.astype(np.float32),
)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    class_mode="categorical",
    target_size=input_size,
    batch_size=batch_size,
)

#test_datagen = ImageDataGenerator(
#    preprocessing_function=lambda x: x.astype(np.float32),
#)
#test_generator = test_datagen.flow_from_directory(
#    test_dir,
#    class_mode="categorical",
#    target_size=input_size,
#    batch_size=batch_size,
#)

# Load the dataset cifar
#(X_train, y_train), (X_val, y_val) = cifar10.load_data()

# 生成模型
vit = VisionTransformer(input_size[0], classes_num, d_model=d_model, d_inner_hid=mlp_dim, \
                   n_head=n_head, layers=layers, dropout=dropout, patch_size=patch_size)

class LRSchedulerPerStep(Callback):
    def __init__(self, d_model, warmup=4000):
        self.basic = d_model**-0.5
        self.warm = warmup**-1.5
        self.step_num = 0
    def on_batch_begin(self, batch, logs = None):
        self.step_num += 1
        lr = self.basic * min(self.step_num**-0.5, self.step_num*self.warm)
        K.set_value(self.model.optimizer.lr, lr)
    def on_epoch_begin(self, epoch, logs = None):
        print('lr=', K.get_value(self.model.optimizer.lr))
lr_scheduler = LRSchedulerPerStep(d_model, 4000) 

mfile = 'vit-face.h5'
model_saver = ModelCheckpoint(mfile, save_best_only=True, save_weights_only=True, monitor='val_loss', verbose=1)

early_stop = EarlyStopping(patience=10, monitor='val_loss')

vit.compile(Adam(1e-4, 0.9, 0.98, epsilon=1e-9))

vit.model.summary()

if __name__ == '__main__':    
    #try: vit.model.load_weights(mfile)
    #except: print('\n\nnew model')

    vit.model.fit(train_generator,
        batch_size=batch_size,
        epochs=epochs, 
        validation_data=test_generator,
        callbacks=[early_stop, lr_scheduler, model_saver])
