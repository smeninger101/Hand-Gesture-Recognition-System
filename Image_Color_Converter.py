import cv2
import os
from os import listdir,makedirs
from os.path import isfile,join

#Path to the Original Images
path = r'C:\Users\smeni\OneDrive\Desktop\Fall_2020_Classes\Capstone\asl-alphabet\asl_alphabet_train\asl_alphabet_train\A' 

#The Destination Path to put the Images in a new folder
dstpath = r'C:\Users\smeni\OneDrive\Desktop\Fall_2020_Classes\Capstone\asl-alphabet\asl_alphabet_train_resize\A' 

try:
    
    #Make a directory
    makedirs(dstpath)
except:
    
    #If directory already exists, say displayed message
    print ("Directory already exist, images will be written in directory folder")

# Folder won't used
files = [f for f in listdir(path) if isfile(join(path,f))] 

for image in files:
    try:
        
        #Join the path to the images
        img = cv2.imread(os.path.join(path,image))
        
        #thresh = 128
        #img_binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
        
        #Resize Images
        image = image.resize((64, 64))
        
        #Join modified images to the Destination Path
        dstPath = join(dstpath,image)
        
        #Put the new images in the Destination Path
        cv2.imwrite(dstPath,image)
    except:
        
        #If the images won't convert, say displayed message
        print ("{} is not converted".format(image))
