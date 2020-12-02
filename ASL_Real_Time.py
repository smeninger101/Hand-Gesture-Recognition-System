#Imports
import cv2 
import numpy as np
from keras.models import load_model


#Disable scientific notation for clarity
np.set_printoptions(suppress=True)

#Load ASL Model
asl_model = load_model('ASL_Model.h5')

#Create the array of the right shape to feed into the keras model
dataset_data = np.ndarray(shape=(1, 64, 64, 3), dtype=np.float32)

#Turn Camera Feed on
camera = cv2.VideoCapture(0)

#Video Quality
camera.set(640,480)

#Prediction Letter Font
font = cv2.FONT_HERSHEY_SIMPLEX

#Dimensions of Blackboard
x, y, w, h = 70, 70, 200, 200

#Camera reads the frame
#Flips the camera 
#Region of Interest Square where my hand is going to go in
while True:
    ret, frame = camera.read()
    frame = cv2.flip(frame,1)
    region_of_interest = frame[70:270,70:270]  

#Resizes the array to 64 x 64
    asl_image = cv2.resize(region_of_interest,(64,64))  #roi.resize((224, 224))
    asl_image_array = np.asarray(asl_image)

#Normalize the image
    normalized_image_array = (asl_image_array.astype(np.float32) / 127.0) - 1

#Load the image into the array
    dataset_data[0] = normalized_image_array

#Run the Program
#Predict the model 
#Print prediction letter
    asl_letter_prediction = asl_model.predict(dataset_data)
    #print(asl_letter_prediction)
    if asl_letter_prediction[0][0] == max(asl_letter_prediction[0]):
        letter = 'A'
    elif asl_letter_prediction[0][1] == max(asl_letter_prediction[0]):
        letter = 'B'
    elif asl_letter_prediction[0][2] == max(asl_letter_prediction[0]):
        letter = 'C'
    elif asl_letter_prediction[0][3] == max(asl_letter_prediction[0]):
        letter = 'D'
    elif asl_letter_prediction[0][4] == max(asl_letter_prediction[0]):
        letter = 'E'
    elif asl_letter_prediction[0][5] == max(asl_letter_prediction[0]):
        letter = 'F'
    elif asl_letter_prediction[0][6] == max(asl_letter_prediction[0]):
        letter = 'G'
    elif asl_letter_prediction[0][7] == max(asl_letter_prediction[0]):
        letter = 'H'
    elif asl_letter_prediction[0][8] == max(asl_letter_prediction[0]):
        letter = 'I'
    elif asl_letter_prediction[0][9] == max(asl_letter_prediction[0]):
        letter = 'J'
    elif asl_letter_prediction[0][10] == max(asl_letter_prediction[0]):
        letter = 'K'
    elif asl_letter_prediction[0][11] == max(asl_letter_prediction[0]):
        letter = 'L'
    elif asl_letter_prediction[0][12] == max(asl_letter_prediction[0]):
        letter = 'M'
    elif asl_letter_prediction[0][13] == max(asl_letter_prediction[0]):
        letter = 'N'
    elif asl_letter_prediction[0][14] == max(asl_letter_prediction[0]):
        letter = 'O'
    elif asl_letter_prediction[0][15] == max(asl_letter_prediction[0]):
        letter = 'P'
    elif asl_letter_prediction[0][16] == max(asl_letter_prediction[0]):
        letter = 'Q'
    elif asl_letter_prediction[0][17] == max(asl_letter_prediction[0]):
        letter = 'R'
    elif asl_letter_prediction[0][18] == max(asl_letter_prediction[0]):
        letter = 'S'
    elif asl_letter_prediction[0][19] == max(asl_letter_prediction[0]):
        letter = 'T'
    elif asl_letter_prediction[0][20] == max(asl_letter_prediction[0]):
        letter = 'U'
    elif asl_letter_prediction[0][21] == max(asl_letter_prediction[0]):
        letter = 'V'
    elif asl_letter_prediction[0][22] == max(asl_letter_prediction[0]):
        letter = 'W'
    elif asl_letter_prediction[0][23] == max(asl_letter_prediction[0]):
        letter = 'X'
    elif asl_letter_prediction[0][24] == max(asl_letter_prediction[0]):
        letter = 'Y'
    elif asl_letter_prediction[0][25] == max(asl_letter_prediction[0]):
        letter = 'Z'
    elif asl_letter_prediction[0][26] == max(asl_letter_prediction[0]):
        letter = 'del'
    elif asl_letter_prediction[0][27] == max(asl_letter_prediction[0]):
        letter = 'nothing'
    elif asl_letter_prediction[0][28] == max(asl_letter_prediction[0]):
        letter = 'space'
    
    #Creating the Blackboard
    blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
    
    #Putting text into the Blackboard
    cv2.putText(blackboard, letter, (30, 200), cv2.FONT_HERSHEY_TRIPLEX, 1.3, (255, 255, 255))
    
    #cv2.putText(frame, letter, (320,55), font, 2 , (255,255,255), 3, cv2.LINE_AA)
    
    #Creating a rectangle to fit the Region of Interest
    cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
    
    #Combines the camera feed with the Blackboard next to it
    res = np.hstack((frame, blackboard))
    
    #Show frame
    cv2.imshow('Camera Feed',frame)
    
    #Show Region of Interest where it's just my hand frame
    cv2.imshow('Region of Interest',region_of_interest)
    
    #Shows the camera feed and the Region of Interest when running the program
    cv2.imshow('Real Time Implementation', res)
    
    #Waitkey to break program
    k = cv2.waitKey(10) & 0xFF
    if k == ord('q'):
        break
    
#Release camera and destroy all windows
camera.release()
cv2.destroyAllWindows()
