import numpy as np
import os
import matplotlib.pyplot as plt

from keras import callbacks as cbks

base_dir = '/home/Unet-ants/'
os.chdir(base_dir+'code/')

# local imports
from sampling import DataLoader, CSVDataset
from sampling import transforms as tx
from models import create_unet_model3D

data_dir = base_dir + 'data_3D/'
results_dir = base_dir+'results_3D/'
try:
    os.mkdir(results_dir)
except:
    pass

# tx.Compose lets you string together multiple transforms
co_tx = tx.Compose([tx.TypeCast('float32'),
                    tx.ExpandDims(axis=-1),
                    tx.RandomAffine(rotation_range=(0,0), # rotate btwn -15 & 15 degrees
                                    translation_range=(0,0), # translate btwn -10% and 10% horiz, -10% and 10% vert
                                    shear_range=(0,0), # shear btwn -10 and 10 degrees
                                    zoom_range=None, # between 15% zoom-in and 15% zoom-out
                                    flip_range=None,
                                    turn_off_frequency=1,
                                    fill_value=0,
                                    target_fill_mode='nearest',
                                    target_fill_value=0) # how often to just turn off random affine transform (units=#samples)
                    ])

input_tx = tx.MinMaxScaler((0,1)) # scale between -1 and 1

target_tx = tx.BinaryMask(cutoff=0.5) # convert segmentation image to One-Hot representation for cross-entropy loss

# use a co-transform, meaning the same transform will be applied to input+target images at the same time 
# this is necessary since Affine transforms have random parameter draws which need to be shared
dataset = CSVDataset(filepath=data_dir+'image_l.csv', 
                    base_path=os.path.join(data_dir,'images'), # this path will be appended to all of the filenames in the csv file
                    input_cols=['Images'], # column in dataframe corresponding to inputs (can be an integer also)
                    target_cols=['Segmentations'],# column in dataframe corresponding to targets (can be an integer also)
                    input_transform=input_tx, target_transform=target_tx, co_transform=co_tx,
                    co_transforms_first=True) # run co transforms before input/target transforms

val_data, train_data = dataset.split_by_column('TrainTest')
# split into train and test set based on the `train-test` column in the csv file
# this splits alphabetically by values, and since 'test' comes before 'train' thus val_data is returned before train_data

# overwrite co-transform on validation data so it doesnt have any random augmentation
val_data.set_co_transform(tx.Compose([tx.TypeCast('float32'),
                                      tx.ExpandDims(axis=-1)]))

# create a dataloader .. this is basically a keras DataGenerator -> can be fed to `fit_generator`
batch_size = 1
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

#write an example batch to a folder as JPEG
#train_loader.write_a_batch(data_dir+'example_batch/')

n_labels = train_data[0][1].shape[-1]
# create model

model = diunet3D_LeakyReLU(train_data[0][0].shape, output_activation='sigmoid', init_lr=0.001, weight_decay=1e-5)
model.summary()
