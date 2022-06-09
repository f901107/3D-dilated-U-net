import os
import glob
import matplotlib.pyplot as plt
import nibabel as nib
from keras import callbacks as cbks
from sampling import DataLoader, CSVDataset
from sampling import transforms as tx
from models import *
from keras.utils import multi_gpu_model

np.random.seed(87)
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
base_dir = '/home/Unet-ants/'
os.chdir(base_dir+'code/')

# local imports
data_dir = base_dir + 'data_3D/'
results_dir = base_dir+'results_3D/'
try:
    os.mkdir(results_dir)
except:
    pass

# tx.Compose lets you string together multiple transforms
co_tx = tx.Compose([tx.TypeCast('float32'),
                    tx.ExpandDims(axis=-1),
                    tx.RandomAffine(rotation_range=None, # rotate btwn -15 & 15 degrees
                                    translation_range=None, # translate btwn -10% and 10% horiz, -10% and 10% vert
                                    shear_range=None, # shear btwn -10 and 10 degrees
                                    zoom_range=None, # between 15% zoom-in and 15% zoom-out
                                    flip_range=(0,0),
                                    turn_off_frequency=1,
                                    fill_value=0,
                                    target_fill_mode='nearest',
                                    target_fill_value=0) # how often to just turn off random affine transform (units=#samples)
                    ])

input_tx = tx.MinMaxScaler((0,1)) # scale between -1 and 1

target_tx = tx.BinaryMask(cutoff=0.5) # convert segmentation image to One-Hot representation for cross-entropy loss

dataset = CSVDataset(filepath=data_dir+'image.csv', 
                    base_path=os.path.join(data_dir,'images'), # this path will be appended to all of the filenames in the csv file
                    input_cols=['Images'], # column in dataframe corresponding to inputs (can be an integer also)
                    target_cols=['Segmentations'],# column in dataframe corresponding to targets (can be an integer also)
                    input_transform=input_tx, target_transform=target_tx, co_transform=co_tx,
                    co_transforms_first=True) # run co transforms before input/target transforms
val_data, train_data = dataset.split_by_column('TrainTest')
val_data.set_co_transform(tx.Compose([tx.TypeCast('float32'),
                                      tx.ExpandDims(axis=-1)]))
batch_size = 1
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
n_labels = train_data[0][1].shape[-1]

# diunet3D
model = diunet3D(train_data[0][0].shape, output_activation='sigmoid', init_lr=0.001, weight_decay=0.00001)
callbacks = [cbks.ModelCheckpoint(results_dir+'diunet3D.h5', monitor='val_loss', save_best_only=True),
            cbks.ReduceLROnPlateau(monitor='val_loss', factor=0.1)]
history_L1 = model.fit_generator(generator=iter(train_loader), steps_per_epoch=np.ceil(len(train_data)/batch_size), 
                    epochs=120, callbacks=callbacks, shuffle=True, validation_data=iter(val_loader), validation_steps=np.ceil(len(val_data)/batch_size))
                    
print(history_L1.history.keys())
# summarize history for accuracy
plt.plot(history_L1.history['dice_coefficient'])
plt.plot(history_L1.history['val_dice_coefficient'])
plt.title('model dice')
plt.ylabel('dice coefficient')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('model_dice.png')
# summarize history for loss
plt.plot(history_L1.history['loss'])
plt.plot(history_L1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')  
plt.savefig('model_loss.png')

real_train_x, real_train_y = train_data.load()
real_val_x, real_val_y = val_data.load()
listfilename = sorted(glob.glob('/home/f901107/New/h/hmf*_T1.nii.gz'))
for i in range(len(real_train_x)):
    real_train_y_pred = model.predict(real_train_x[i:i+1])
    real_train_y_pred[real_train_y_pred>=0.5] = 1
    real_train_y_pred[real_train_y_pred<0.5] = 0
    real_train_y_pred = (real_train_y_pred[0,:,:,:,0]*1.)
    img = nib.load(listfilename[i])
    filename = listfilename[i][20:]
    nifti_affine = img.affine
    nifti_data = img.get_data()
    nin = nib.Nifti1Image(real_train_y_pred, nifti_affine)
    nin.to_filename('01_'+filename)
    print(filename)
# test data predict
for j in range(len(real_val_x)):
    real_val_y_pred = model.predict(real_val_x[j:j+1])
    real_val_y_pred[real_val_y_pred>=0.5] = 1
    real_val_y_pred[real_val_y_pred<0.5] = 0
    real_val_y_pred = (real_val_y_pred[0,:,:,:,0]*1.)
    img = nib.load(listfilename[i+j+1])
    filename = listfilename[i+j+1][20:]
    nifti_affine = img.affine
    nifti_data = img.get_data()  
    nin = nib.Nifti1Image(real_val_y_pred, nifti_affine)
    nin.to_filename('01_'+filename)
    print(filename)               
