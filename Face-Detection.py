import cv2
import numpy as np
import face_recognition

imgV = face_recognition.load_image_file('Images/bts-jungkook.jpg')
imgV = cv2.cvtColor(imgV,cv2.COLOR_BGR2RGB)
imgVTest = face_recognition.load_image_file('Images/BTS-RM.jpg')
imgVTest = cv2.cvtColor(imgVTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgV)[0]
encodeV = face_recognition.face_encodings(imgV)[0]
cv2.rectangle(imgV,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)


faceLocTest = face_recognition.face_locations(imgVTest)[0]
encodeVTest = face_recognition.face_encodings(imgVTest)[0]
cv2.rectangle(imgVTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeV],encodeVTest)
faceDistance= face_recognition.face_distance([encodeV],encodeVTest)
print(results,faceDistance)
cv2.putText(imgVTest,f'{results}{round(faceDistance[0],2)}',(50,50),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(0,0,255),2)


cv2.imshow('BTS V',imgV)
imgVTest1 = cv2.resize(imgVTest, (960, 540))
cv2.imshow("IMAGEV", imgVTest1)
#cv2.imshow('BTS V TEST',imgVTest)

cv2.waitKey(0)
