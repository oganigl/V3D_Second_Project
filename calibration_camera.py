import cv2
import numpy as np
import glob
import os

# Parámetros del patrón del tablero de ajedrez
chessboard_size = (8, 6)  # columnas internas, filas internas
square_size = 1.0  # puede ser en cm, mm, lo que uses como escala

# Preparar puntos de objeto (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
objp = np.zeros((chessboard_size[1]*chessboard_size[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays para almacenar los puntos de objeto y los puntos de imagen de todas las imágenes
objpoints = []  # puntos 3D en el mundo real
imgpoints = []  # puntos 2D en la imagen

# Leer imágenes
ruta_imagenes = "./capturas_left/*.jpg"
imagenes = glob.glob(ruta_imagenes)

print(f"Encontradas {len(imagenes)} imágenes...")

gray = None

for fname in imagenes:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Encontrar las esquinas del tablero
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    cv2.imshow('Esquinas detectadas', img)
    cv2.waitKey(100)

    # Si se encuentran, añadir puntos
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), 
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)

        # Dibuja y muestra las esquinas
        img = cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        
        

cv2.destroyAllWindows()

# Calibrar la cámara
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(ret)
# Guardar los parámetros de calibración
np.savez("calibration_left.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
print("Calibración guardada en 'calibracion.npz'")

# Mostrar imágenes corregidas
for fname in imagenes:
    img = cv2.imread(fname)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # Corregir distorsión
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # Recortar la imagen resultante
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    cv2.imshow('Imagen corregida', dst)
    cv2.waitKey(300)

cv2.destroyAllWindows()
