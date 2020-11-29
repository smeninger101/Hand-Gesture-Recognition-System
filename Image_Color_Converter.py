import cv2
import os
from os import listdir,makedirs
from os.path import isfile,join

path = r'C:\Users\smeni\OneDrive\Desktop\Fall_2020_Classes\Capstone\asl-alphabet\asl_alphabet_train\asl_alphabet_train\A' # Source Folder
dstpath = r'C:\Users\smeni\OneDrive\Desktop\Fall_2020_Classes\Capstone\asl-alphabet\asl_alphabet_train_resize\A' # Destination Folder

try:
    makedirs(dstpath)
except:
    print ("Directory already exist, images will be written in directory folder")

# Folder won't used
files = [f for f in listdir(path) if isfile(join(path,f))] 

for image in files:
    try:
        img = cv2.imread(os.path.join(path,image))
        #thresh = 128
        #img_binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
        image = image.resize((64, 64))
        dstPath = join(dstpath,image)
        cv2.imwrite(dstPath,image)
    except:
        print ("{} is not converted".format(image))