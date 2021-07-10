import cv2
import os
import numpy as np
import face_recognition1 as fr
test_img = cv2.imread('E:/Study Material/Projects/Machine Learning/OpenCV Project/Testing Image/test.jpg')
faces_detected,gray_img= fr.faceDetection(test_img)
print('faces detected',faces_detected)


#comment thecode when you are running the code for 2nd time 
faces, faceID = fr.labels_for_training_images('E:/Study Material/Projects/Machine Learning/OpenCV Project/baseimage')
face_recognizer = fr.train_classifier(faces, faceID)
face_recognizer.write('E:/Study Material/Projects/Machine Learning/OpenCV Project/trainingdata.yml')


#uncomment this line when running code for 2nd time 

#face_recognizer=cv2.face.LBPHFaceRecognizer_create()
#face_recognizer.read('E:/Study Material/Projects/Machine Learning/OpenCV Project/trainingdata.yml')

name = { 0:'Sagnik'} # creating Dictionary containing names for label

for face in faces_detected:
    (x,y,w,h)= face
    roi_gray=gray_img[y:y+h,x:x+h]
    label, confidence = face_recognizer.predict(roi_gray) # predict label of the image 
    
    fr.draw_react(test_img, face)
    predicted_name = name[label]
    
    if(confidence>35):#if more than 35 no name print
         continue
    fr.put_text(test_img, predicted_name, x, y)
    print('Confidence:',confidence)
    print('Label:',label)
    resized_img = cv2.resize(test_img,(500,700))
    cv2.imshow('face detection',resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows
     
    