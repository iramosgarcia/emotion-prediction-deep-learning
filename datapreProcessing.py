
from scipy.io import loadmat
import pandas as pd
import numpy as np
from random import shuffle
import os
import cv2

# emotion labels from FER2013:
emotion = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3,
           'Sad': 4, 'Surprise': 5, 'Neutral': 6}
           
emo     = ['Angry', 'Fear', 'Happy','Sad', 'Surprise', 'Neutral']

class DataManager(object):
    """Class for loading fer2013 emotion classification dataset or
        imdb gender classification dataset."""
    def __init__(self, dataset_name='imdb', dataset_path=None, image_size=(48, 48)):

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.image_size = image_size
        if self.dataset_path != None:
            self.dataset_path = dataset_path
        elif self.dataset_name == 'fer2013':
            self.dataset_path = '/data/fer2013.csv'
        else:
            raise Exception('Incorrect dataset name, please input fer2013')
            
    def get_data(self):
        return self._load_data()
            
    def _load_data(self):
        data = pd.read_csv(self.dataset_path)
        pixels = data['pixels'].tolist()
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'), self.image_size)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        emotions = pd.get_dummies(data['emotion']).as_matrix()
        return faces, emotions
        
        
    def split_data(x, y, validation_split=.2):
        num_samples = len(x)
        num_train_samples = int((1 - validation_split)*num_samples)
        train_x = x[:num_train_samples]
        train_y = y[:num_train_samples]
        val_x = x[num_train_samples:]
        val_y = y[num_train_samples:]
        train_data = (train_x, train_y)
        val_data = (val_x, val_y)
        return train_data, val_data
        
    def get_class_to_arg(dataset_name='fer2013'):
        if dataset_name == 'fer2013':
            return {'angry':0, 'disgust':1, 'fear':2, 'happy':3, 'sad':4,
                    'surprise':5, 'neutral':6}
        else:
            raise Exception('Invalid dataset name')