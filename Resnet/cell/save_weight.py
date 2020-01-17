import os
from keras import Input
import glob
from keras.models import load_model,Model
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D,Dense
from PIL import Image
import numpy as np
from classification_models.keras import Classifiers
from collections import Counter
import csv
from keras.preprocessing.image import ImageDataGenerator
import shutil
from sklearn.metrics import confusion_matrix

model_path = r'E:\多目标跟踪\deep_sort_yolov3-master\deep_sort_yolov3-master\model_data\trained_weights_final.h5'
model = load_model(model_path)
model.save_weights('cell_9class_weight.weights')
