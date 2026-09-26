import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
cap = cv.VideoCapture(0)
detector = HandDetector(maxHands=1)
imgSize = 300
offset = 20
Folder = "Data/go_ahead"
counter = 0
while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img)
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
                imgWhite[Gap:ratio + Gap,:] = imgResize
            
            
            cv.imshow("ImageCrop", imgCrop)
            cv.imshow("IMageWhite",imgWhite)

    cv.imshow("Image", img)

    # Salir con tecla 'q'
    key = cv.waitKey(1) 

    if key == ord("s"):
        counter += 1
        cv.imwrite(f'{Folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)
    if key == ord("q"):
        break
    
cap.release()
cv.destroyAllWindows()
