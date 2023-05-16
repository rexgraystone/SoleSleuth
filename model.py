import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping

path = 'Dataset/'
train_path = path + "train/"
val_path = path + "validation"

data_gen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=30, 
    width_shift_range=0.2, 
    validation_split=0.2, 
    height_shift_range=0.2, 
    shear_range=0.2
)
train_gen = data_gen.flow_from_directory(
    directory=train_path, 
    class_mode='categorical', 
    target_size=(400, 400), 
    subset='training', 
    batch_size=20
)
val_gen = data_gen.flow_from_directory(
    directory=train_path, 
    class_mode='categorical', 
    target_size=(400, 400), 
    subset='validation', 
    batch_size=20
)