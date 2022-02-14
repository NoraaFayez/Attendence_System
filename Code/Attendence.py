from ctypes.wintypes import RGB
import cv2
import os
import face_recognition
import numpy as np
from datetime import datetime

path = "Image_database"
img=[]
User_names = []
mylist = os.listdir(path)

############### Read the images ##################
for cls in mylist:
    currentImage = cv2.imread(f'{path}/{cls}')
    img.append(currentImage)
    User_names.append(os.path.splitext(cls)[0])

def find_encodings(img):
    encodings = []
    for image in img:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image)[0]
        encodings.append(encode)
    return encodings

def markAttendence(name):
    with open("Code\Attendence.csv","r+") as f:
        dataList = f.readlines()
        print(dataList)
        nameList = []
        for line in dataList:
            entry= line.split(",")
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dateStr = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dateStr}')

imgEncodings = find_encodings(img)
video_capture = cv2.VideoCapture(1)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detectIamge(gray):
    faces = detector.detectMultiScale(RGB, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frames,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.waitKey(1)
        if cv2.waitKey(100) & 0xFF == ord('s'):
            per_name = input("Please Enter your name")
            pathNewImage = cv2.imwrite("Image_database/"+per_name+".jpg",RGB[y:y+h,x:x+w]) 
            mylist.append(pathNewImage)

def test(imgRGB):
    faceLocCam = face_recognition.face_locations(imgRGB)
    encodeCam = face_recognition.face_encodings(imgRGB,faceLocCam)
    for encodeLoop,locLoop in zip(encodeCam,faceLocCam):
        match = face_recognition.compare_faces(imgEncodings,encodeLoop)
        distMatch = face_recognition.face_distance(imgEncodings,encodeLoop)
        matchIndex = np.argmin(distMatch)

        if match[matchIndex]:
            name = User_names[matchIndex].upper()
            markAttendence(name)
        else:
            name = detectIamge(frames)
        print(name)

        y1,x2,y2,x1 = locLoop
        cv2.rectangle(frames,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frames,name,(x1+6,y2+6),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0))

            
while True:
    # Capture frame-by-frame
    ret, frames = video_capture.read()
    imgRGB = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    detectIamge(imgRGB)    

    cv2.imshow("Camera",frames)
    cv2.waitKey(1)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break 

