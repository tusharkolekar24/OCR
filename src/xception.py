import pandas as pd
import cv2
import numpy as np
import keras
import json
import matplotlib.pyplot as plt
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, model_from_json
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import os

class DataPreperation:
    def __init__(self,file_path):
        self.file_path = file_path
    
    def gether_info(self):
        paths = self.file_path
        dataset_detail = []
        for folder_name in os.listdir(paths):
            # print(folder_name)
            for image_path in os.listdir(os.path.join(paths,folder_name)):
                # print(folder_name,image_path)
                images = cv2.imread(os.path.join(paths,folder_name,image_path))
                stretch_near = cv2.resize(images, (80,80), 
                            interpolation = cv2.INTER_LINEAR)
                
                # Use the cvtColor() function to grayscale the image 
                gray_image = cv2.cvtColor(stretch_near, cv2.COLOR_BGR2GRAY)/255.0 
                
                dataset_detail.append([gray_image,folder_name.replace("_","")])
            # break  
        # break
        images_set,label_set = zip(*dataset_detail)
        
        return np.array([f for f in images_set]),np.array(label_set)


def plot_myimages(images_set,label_bucket,idx):
    plt.title("Categories:{}".format(label_bucket[idx]))
    plt.imshow(images_set[idx])

class SplitDataset:
    def __init__(self,original_dataset, label_bucket, test_size):
        self.original_dataset = original_dataset
        self.label_bucket     = label_bucket
        self.test_size        = test_size

    def train_test_split(self):
        X_train, X_test, y_train, y_test = train_test_split(self.original_dataset, 
                                                               self.label_bucket, 
                                                               test_size=0.2, 
                                                               stratify=self.label_bucket)
        return X_train, X_test, y_train, y_test
    
def decode_labels(y_train,y_test,num_classes):
    encoder1 = LabelEncoder()

    y_train_encoded = encoder1.fit_transform(y_train)
    y_test_encoded  = encoder1.fit_transform(y_test) 


    # num_classes =  53

    y_train_image = keras.utils.to_categorical(y_train_encoded, num_classes)
    y_test_image  = keras.utils.to_categorical(y_test_encoded, num_classes)

    return y_train_image, y_test_image, encoder1

class Xception_Model:
    def __init__(self,input_size,num_classes,train_info):
        self.input_size    = input_size
        self.num_classes   = num_classes
        self.X_train_image = train_info['X_train_image']
        self.y_train_image = train_info['y_train_image']
        self.X_test_image  = train_info['X_test_image']
        self.y_test_image  = train_info['y_test_image']
        self.batch_size    = train_info['batch_size']
        self.epochs        = train_info['epochs']

    def Model_Info(self):
        base_model = Xception(weights='imagenet', include_top=False, input_shape=(80, 80, 3))

        for layer in base_model.layers:
            layer.trainable = False

        x = Flatten()(base_model.output)
        x = Dense(250, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=output)
        model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])
        
        model.summary() 

        return model 
    
    def model_fit(self):
        model   = self.Model_Info()

        history = model.fit(self.X_train_image, self.y_train_image,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            verbose=1,
                            validation_data=(self.X_test_image, self.y_test_image)
                        )    
        
        return model

    def save_model(self,model,json_file_path,weight_file_path):
        # serialize model to JSON
        model_json = model.to_json()
        with open(json_file_path, "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights(weight_file_path)

        return model
    
if __name__=='__main__':
    obj1 = DataPreperation(file_path=r'D:\TrafficAdBar\Original') 
    images_bucket, label_bucket = obj1.gether_info()

    original_dataset = np.array([np.stack((row,) * 3, axis=-1) for row in np.array(images_bucket)])

    obj2 = SplitDataset(original_dataset = original_dataset,
                        label_bucket     = label_bucket,
                        test_size        = 0.2)

    X_train, X_test, y_train, y_test = obj2.train_test_split()
    

    y_train_image, y_test_image, encoder_info = decode_labels(y_train     = y_train,
                                                              y_test      = y_test,
                                                              num_classes = 53)
    
    num_classes = 53
    train_info  = {'X_train_image':X_train,
                   'y_train_image':y_train_image,
                   'X_test_image':X_test,
                   'y_test_image':y_test_image,
                   'batch_size':124,
                   'epochs':10}
    
    input_size  = (80, 80, 3)
    xception_model = Xception_Model(input_size  = input_size,
                          num_classes = num_classes,
                          train_info  = train_info)
    
    train_xception_model =  xception_model.model_fit()

    # D:\upgraded_git_repo\test_info
    xception_model.save_model(model            = train_xception_model,
                         json_file_path   = r'D:\upgraded_git_repo\test_info\xception.json',
                         weight_file_path = r'D:\upgraded_git_repo\test_info\xception.h5')
    
    mapping = dict()
    for number, categories in zip(np.arange(0,53),encoder_info.inverse_transform(np.arange(0,53))):
        mapping[str(number)]=categories

    with open(r"D:\upgraded_git_repo\test_info\label_encoder_xception.json", "w") as json_file:
        json.dump(mapping, json_file, indent = 4)