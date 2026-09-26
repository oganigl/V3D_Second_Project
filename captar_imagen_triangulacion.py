import cv2 as cv
import numpy as np
import pickle
import socket
import struct
import os
import pyzed.sl as sl

ground = np.load('calibration_ground.npz')
Rg = ground['Rg']
Origin = ground['Origin']
relacion_casilla_metro = 0.108


def transform_to_ground(points, R, origin):
    translated = points - origin
    transformed = R @ translated.T
    return transformed.T


def load_stereo_calibration(filename):
    data = np.load(filename)
    return data['K1'], data['D1'], data['K2'], data['D2'], data['R'], data['T']


def rectify_images(img1, img2, K1, D1, K2, D2, R, T):
    size = (img1.shape[1], img1.shape[0])
    R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(
        K1, D1, K2, D2, size, R, T,
        flags=cv.CALIB_ZERO_DISPARITY, alpha=0
    )
    map1x, map1y = cv.initUndistortRectifyMap(K1, D1, R1, P1, size, cv.CV_32FC1)
    map2x, map2y = cv.initUndistortRectifyMap(K2, D2, R2, P2, size, cv.CV_32FC1)
    img1_rect = cv.remap(img1, map1x, map1y, cv.INTER_LINEAR)
    img2_rect = cv.remap(img2, map2x, map2y, cv.INTER_LINEAR)
    return img1_rect, img2_rect, P1, P2, Q


def segment_black_objects(img, min_area=1000, max_aspect_ratio=5.0):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Aplicar morfologías para limpiar la imagen (suavizar, eliminar ruido)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel, iterations=5)
    cleaned = cv.morphologyEx(cleaned, cv.MORPH_CLOSE, kernel, iterations=5)

    # Binarizar con umbral fijo para separar negro
    _, thresh = cv.threshold(cleaned, 60, 255, cv.THRESH_BINARY_INV)

    # Encontrar contornos para filtrar por tamaño y forma
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    filtered_contours = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv.boundingRect(cnt)
        aspect_ratio = max(w/h, h/w)  # Relación de aspecto
        if aspect_ratio > max_aspect_ratio:
            continue

        filtered_contours.append(cnt)

    # Crear máscara vacía y dibujar solo los contornos filtrados
    mask_clean = np.zeros_like(thresh)
    cv.drawContours(mask_clean, filtered_contours, -1, 255, thickness=cv.FILLED)

    # Mostrar la imagen con los contornos filtrados para depurar
    mask_bgr = cv.cvtColor(mask_clean, cv.COLOR_GRAY2BGR)
    cv.drawContours(mask_bgr, filtered_contours, -1, (0, 255, 0), 3)
    cv.imshow("Mascara segmentada y contornos filtrados", mask_bgr)
    cv.waitKey(1)

    return mask_clean


def extract_corners_with_extremes(mask):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    objects = []
    for c in contours:
        if cv.contourArea(c) > 75:
            obj_mask = np.zeros_like(mask)
            cv.drawContours(obj_mask, [c], -1, 255, -1)

            corners = cv.goodFeaturesToTrack(obj_mask, maxCorners=30, qualityLevel=0.03, minDistance=5)
            if corners is None or len(corners) < 3:
                continue
            corners = corners.reshape(-1, 2)

            contour_points = c.reshape(-1, 2)

            # Calcular centroide
            M = cv.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroid = (cx, cy)
            else:
                centroid = (0, 0)

            # Obtener orientación y dimensiones
            rect = cv.minAreaRect(c) 
            ((x_center, y_center), (w, h), angle) = rect

            # Ajustar ángulo si es necesario
            if w < h:
                angle += 90
                w, h = h, w  # Hacemos que w sea el largo y h el ancho, por consistencia

            objects.append({
                'contour': c,
                'all_vertices_2d': corners.astype(int),
                'all_contour_points': contour_points.astype(int),
                'centroid': centroid,
                'rotation_z': angle,     # en grados
                'width': w,              # ancho (el menor lado)
                'length': h              # largo (el mayor lado)
            })
    return objects


def get_rotated_rectangle_vertices(center, width, length, angle_deg):
    angle = np.deg2rad(angle_deg)
    w, h = width / 2, length / 2  # mitad de dimensiones

    # Ejes rotados
    dx = np.cos(angle)
    dy = np.sin(angle)

    # Vectores de dirección
    vec_w = np.array([dx, dy]) * w
    vec_h = np.array([-dy, dx]) * h  # perpendicular

    # Centro
    cx, cy = center

    # 4 vértices alrededor del centro
    p1 = (cx + vec_w[0] + vec_h[0], cy + vec_w[1] + vec_h[1])
    p2 = (cx - vec_w[0] + vec_h[0], cy - vec_w[1] + vec_h[1])
    p3 = (cx - vec_w[0] - vec_h[0], cy - vec_w[1] - vec_h[1])
    p4 = (cx + vec_w[0] - vec_h[0], cy + vec_w[1] - vec_h[1])

    return np.array([p1, p2, p3, p4], dtype=np.float32)


def triangulate_full_object_vertices(objects_left, objects_right, P1, P2):
    all_objects_3d = []

    for objL, objR in zip(objects_left, objects_right):
        centerL = objL['centroid']
        centerR = objR['centroid']

        widthL = objL['width']
        lengthL = objL['length']
        angleL = objL['rotation_z']

        widthR = objR['width']
        lengthR = objR['length']
        angleR = objR['rotation_z']

        # Obtener 4 vértices para cada cámara
        vertsL = get_rotated_rectangle_vertices(centerL, widthL, lengthL, angleL)
        vertsR = get_rotated_rectangle_vertices(centerR, widthR, lengthR, angleR)

        object_3d_pts = []

        for pL, pR in zip(vertsL, vertsR):
            # Homogeneizar para triangulación
            pL_h = np.array([[pL[0]], [pL[1]]], dtype=np.float32)
            pR_h = np.array([[pR[0]], [pR[1]]], dtype=np.float32)

            point_4d_hom = cv.triangulatePoints(P1, P2, pL_h, pR_h)
            point_3d = point_4d_hom[:3] / point_4d_hom[3]  # Normalizar homogéneo
            point_3d = transform_to_ground(point_3d.flatten(), Rg, Origin) * relacion_casilla_metro  # Transformar a coordenadas del suelo   
            object_3d_pts.append(point_3d)
            
        all_objects_3d.append(np.array(object_3d_pts))  

    return all_objects_3d

class ZEDCamera():
    def __init__(self, camera_name, fps):
        self.name = 'camera_' + camera_name
        self.fps = fps
        self.img_count = 0
        self.cam = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        init_params.coordinate_units = sl.UNIT.METER
        init_params.camera_fps = self.fps
        status = self.cam.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print("Error al abrir la cámara ZED")
            exit(1)
    def capture_images(self):
        key = -1
        #socket para enviar las imagenes en tiempo real
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        runtime = sl.RuntimeParameters()
        mat = sl.Mat()
        key = -1
        K1, D1, K2, D2, R, T = load_stereo_calibration('calibration_stereo_data.npz')
        img_left = None
        img_right = None
        while key != 113:  # esperar tecla 'q'
            err = self.cam.grab(runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                for orientation in ["l", "r"]:
                    if orientation == "l":
                        self.cam.retrieve_image(mat, sl.VIEW.LEFT)
                        img_left = mat.get_data()
                    else:
                        self.cam.retrieve_image(mat, sl.VIEW.RIGHT)
                        img_right = mat.get_data()
                        
            # Disminuir calidad en la imagen, asi disminuimos el tamaño de la imagen y la enviamos para la realidad aumentada
            encode_param = [int(cv.IMWRITE_JPEG_QUALITY), 50]  # Calidad 50%
            result, encimg = cv.imencode('.jpg', img_left, encode_param)
            s.sendto(encimg.tobytes(),("localhost", 5030)) 
            
            # Cargar calibración
            K1, D1, K2, D2, R, T = load_stereo_calibration('calibration_stereo_data.npz')

            # Rectificar imágenes
            imgL_rect, imgR_rect, P1, P2, Q = rectify_images(img_left, img_right, K1, D1, K2, D2, R, T)
            # Segmentar objetos negros
            maskL = segment_black_objects(imgL_rect)
            maskR = segment_black_objects(imgR_rect)

            # Extraer vértices
            objectsL = extract_corners_with_extremes(maskL)
            objectsR = extract_corners_with_extremes(maskR)

            # Dibujar contornos y vértices en la imagen izquierda
            img_right_pts = imgR_rect.copy()
            img_left_pts = imgL_rect.copy()
            print(len(objectsL), len(objectsR))
            for obj in objectsL:
                cv.drawContours(img_left_pts, [obj['contour']], -1, (0, 255, 0), 2)
                for pt in obj['all_vertices_2d']:
                    cv.circle(img_left_pts, tuple(pt), 5, (255, 0, 0), -1)
            
            cv.imshow('Left Rectified - Puntos 2D de contorno y esquinas', img_left_pts)

            # Triangular vértices
            objects_3d = triangulate_full_object_vertices(objectsL, objectsR, P1, P2)
            

            # Guardar resultados
            flat_floats = []
            for obj in objects_3d:
                flat_floats.extend(obj.flatten())

            #empaquetar todos los puntos 3D en un solo paquete
            data = struct.pack(f'{len(flat_floats)}f', *flat_floats)
            #enviar paquete con los vertices de los objetos detectados
            s.sendto(data, ('192.168.185.196', 5050)) #la dirección IP era esa pero depende de la red
            print(len(data))
            key = cv.waitKey(1)

        cv.destroyAllWindows()


def main():
    camera_name = "ZED1"
    fps = 30
    zed_camera = ZEDCamera(camera_name, fps)
    zed_camera.capture_images()

if __name__ == '__main__':
    main()