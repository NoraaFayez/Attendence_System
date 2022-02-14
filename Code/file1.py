import cv2
import os
import face_recognition
import numpy as np

imgElon = face_recognition.load_image_file("Image_database\Elon_Musk.jpg")
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file("Image_database\Elon_Musk_2.jpg")
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

imgTest_2 = face_recognition.load_image_file("Image_database\Gates.jpg")
imgTest_2 = cv2.cvtColor(imgTest_2,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(0,255,12),12)
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeElonTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(12,255,12),12)

faceLocTest_2 = face_recognition.face_locations(imgTest_2)[0]
encodeBillTest = face_recognition.face_encodings(imgTest_2)[0]
cv2.rectangle(imgTest,(faceLocTest_2[3],faceLocTest_2[0]),(faceLocTest_2[1],faceLocTest_2[2]),(12,12,12),12)

res = face_recognition.compare_faces([encodeElon],encodeElonTest)
simiar = face_recognition.face_distance([encodeElon],encodeElonTest)
cv2.putText(imgElon,f'{res}{round(simiar[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0))

cv2.imshow("Elon",imgElon)
#cv2.imshow("Test",imgTest)
cv2.waitKey(0)

""" cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(1) """

""" while True:
    # Capture frame-by-frame
    ret, frames = video_capture.read()

    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frames)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break """

    