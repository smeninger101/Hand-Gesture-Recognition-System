#Imports
import cv2
import keras
import os
import numpy as np
from sklearn.model_selection import train_test_split
from glob import glob
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D


#Training Directory and Testing Directory
train_directory = 'asl-alphabet/asl_alphabet_train/asl_alphabet_train'
test_directory = 'asl-alphabet/asl_alphabet_test/asl_alphabet_test'

#Gets Image Size
def image_size():
    image = cv2.imread('asl-alphabet/asl_alphabet_train/asl_alphabet_train/A/A1.jpg',0)
    return image.shape

#Gets Number of Classes
def number_of_classes():
    return len(glob('asl-alphabet/asl_alphabet_train/asl_alphabet_train/*'))
image_x, image_y = image_size()

#Labels Dictionary
labels_dictionary = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,'Z':25,'space':26,'del':27,'nothing':28}

#Load Data  
def load_data():
    
    #How my images and labels will be loaded
    images = []
    labels = []
    
    #Size of the images
    size = 64,64
    
    #Printing out the statement "LOADING DATA FROM"
    print("LOADING DATA FROM : ",end = "")
    for folder in os.listdir(train_directory):
        
        #Print out folder name with a line in between each folder name
        print(folder, end = ' | ')
        for image in os.listdir(train_directory + "/" + folder):
           
            #Read the images from the specified path
            ASL_Image = cv2.imread(train_directory + '/' + folder + '/' + image)
            
            #Resize the images
            ASL_Image = cv2.resize(ASL_Image, size)
            
            #Append those images
            images.append(ASL_Image)
            
            #Append the labels from the label dictionary
            labels.append(labels_dictionary[folder])
    
    #Implements numpy slicing and converts the images to float 32
    images = np.array(images)
    images = images.astype('float32')/255.0
    
    #One-Hot Encoding
    labels = keras.utils.to_categorical(labels)
    
    #Splitting up my dataset
    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size = 0.01)
    
    #Printing out the shape of my X_train and X_test
    print()
    print('Loaded', len(X_train),'images for training,','Train data shape =',X_train.shape)
    print('Loaded', len(X_test),'images for testing','Test data shape =',X_test.shape)
    
    return X_train, X_test, Y_train, Y_test

#Makes these values equal to load_data()
X_train, X_test, Y_train, Y_test = load_data()


#Creatung my model
def create_model():
    
    #Creating a Sequential Model
    model = Sequential()
    
    #Add Model Layers
    model.add(Conv2D(16, kernel_size = [3,3], padding = 'same', activation = 'relu', input_shape = (64,64,3)))
    model.add(Conv2D(32, kernel_size = [3,3], padding = 'same', activation = 'relu'))
    
    #Add Pooling Layer
    model.add(MaxPooling2D(pool_size = [3,3]))
    
    #Add Regularization Layer(new)
    model.add(Dropout(0.5))
    
    #Add Model Layers
    model.add(Conv2D(32, kernel_size = [3,3], padding = 'same', activation = 'relu'))
    model.add(Conv2D(64, kernel_size = [3,3], padding = 'same', activation = 'relu'))
    
    #Add Pooling Layer
    model.add(MaxPooling2D(pool_size = [3,3]))
    
    #Add Regularization Layer(new)
    model.add(Dropout(0.5))
    
    #Add Model Layers
    model.add(Conv2D(128, kernel_size = [3,3], padding = 'same', activation = 'relu'))
    model.add(Conv2D(256, kernel_size = [3,3], padding = 'same', activation = 'relu'))
    
    #Add Pooling Layer
    model.add(MaxPooling2D(pool_size = [3,3]))
    
    #Add Normailization Layer
    model.add(BatchNormalization())
    
    #Flatten the images
    model.add(Flatten())
    
    #Add Regularization Layer
    model.add(Dropout(0.5))
    
    #Add Core Layers
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(29, activation = 'softmax'))
    
    #Compiling my Model
    model.compile(optimizer = 'adam', loss = keras.losses.categorical_crossentropy, metrics = ["accuracy"])
    
    #Print out "MODEL CREATED" and give a model summary
    print("MODEL CREATED")
    model.summary()
    
    #Return my Model
    return model

#Fitting my Model
def fit_model():
    
    #Fitting my Model
    model_history = model.fit(X_train, Y_train, batch_size = 64, epochs = 5, validation_split = 0.1)
    
    #Evaluating my Model
    score = model.evaluate(x = X_test, y = Y_test, verbose = 0)
    
    #Printing out the test images accuracy
    print('Accuracy for test images:', round(score[1]*100, 3), '%')
    return model_history


#Set my model equal to build_model()
model = create_model()

#Set my model_history to fit_model()
model_history = fit_model()

#Print out accuracies
if model_history:
    print('Final Accuracy: {:.2f}%'.format(model_history.history['acc'][4] * 100))
    print('Validation Set Accuracy: {:.2f}%'.format(model_history.history['val_acc'][4] * 100))
    model.save('ASL_Model.h5')
