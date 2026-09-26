import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import tensorflow
import numpy as np
import math
import time
import socket
import struct
cap = cv.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5","Model/labels.txt")
imgSize = 300
offset = 5
counter = 0
Labels = ["go_ahead", "stop" , "rotate" ]

Label_already_sent = None


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("192.168.185.196",3033)) 
while True:
    success, img = cap.read()
    if not success:
        break
    
    imgOutput = img.copy()
    hands,img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Validar que los valores estén dentro de los límites de la imagen
        y1 = max(0, y)
        y2 = min(img.shape[0], y + h)
        x1 = max(0, x)
        x2 = min(img.shape[1], x + w)
       
        if y2 > y1 and x2 > x1:
            imgCrop = img[y1:y2, x1:x2]    # Solo si el recorte tiene tamaño válido
            imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255
            aspectRatio = h/w
            if aspectRatio > 1:
                k = imgSize/h
                ratio  = math.ceil(k*w)
            else:
                k = imgSize/w
                ratio = math.ceil(k*h)
                
            imgResize = cv.resize(imgCrop,(ratio,imgSize)) if aspectRatio > 1  else cv.resize(imgCrop,(imgSize,ratio))
            imgResizeShape = imgResize.shape
            Gap = math.ceil((imgSize - ratio)/2)
            if aspectRatio > 1:
                imgWhite[:,Gap:ratio + Gap] = imgResize
               
            else:
                imgWhite[Gap:ratio + Gap,:]  = imgResize
            
            prediction, index = classifier.getPrediction(imgWhite)
            confianza = 0.95
            if(Labels[index] == "go_ahead"):
                confianza = 0.7
                
            if (prediction[index] > confianza):
                print(Labels[index])
                cv.putText(imgOutput,Labels[index],(x,y-20),cv.FONT_HERSHEY_COMPLEX,2,(255,0,0))
                #enviar mensaje por TCP
                if Label_already_sent != Labels[index]:
                    Value = struct.pack('i',index)
                    print("index:", index)
                    s.send(Value)
                    Label_already_sent = Labels[index]
            cv.imshow("ImageWhite",imgWhite)
            cv.rectangle(imgOutput,(x - 20,y - 20),(x+w + 20,y+h+20),(255,0,255),4)
    cv.imshow("Image", imgOutput)

    # Salir con tecla 'q'
    key = cv.waitKey(1) 
    if key == ord("q"):
        break
    
cap.release()
cv.destroyAllWindows()
