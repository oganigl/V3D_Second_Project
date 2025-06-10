import cv2 as cv
import os
import numpy as np
import pickle
import pyzed.sl as sl

import open3d as o3d
from PIL import Image
import socket
import copy
import time
import math

# Variables globales
K1, D1, K2, D2, R, T = None, None, None, None, None, None
K = None
t = None

P0_world = np.array([0, 0, 0])
P1_world = np.array([1, 0, 0])
P2_world = np.array([0, 1, 0])

moneda_positions = []
moneda = None
monedas = []
axis = None
params = None
sock_robot = None
vis = None
robot_cube = None

width = 640
height = 480
R_wc = None

R_cw = None
t_cw = None
    
#de camara a mundo

R_cg = None
t_cg = None
R_gc = None
t_gc = None

# --- FUNCIONES AUXILIARES ---
def world_to_cam(P_world, R, t):
    P_world = np.asarray(P_world).reshape(3,)
    return R @ P_world + t

# FUNCIONES (algunas corregidas):
def load_stereo_calibration(filename):
    global K1, D1, K2, D2, R, T, K, t
    data = np.load(filename)
    K1 = data['K1']
    D1 = data['D1']
    K2 = data['K2']
    D2 = data['D2']
    R = data['R']
    T = data['T'] 
    K = K1  # Asumimos K1 como la cámara principal
    t = T.reshape(3,)
    return K1, D1, K2, D2, R, T


def ground_to_world(p_ground):
    global t_gc, t_cw, R_gc, R_cg, R_cw, T_cw
    
    p_ground = np.asarray(p_ground).reshape(3,1)
    t_gc = np.array(t_gc).reshape(3,1)
    t_cw = np.asarray(t_cw).reshape(3,1)

    #ground -> cam
    p_cam = R_cg @ p_ground + t_gc
    #cam -> world
    p_world = R_cw @ p_cam + t_cw

    return p_world.flatten()
    
def cargar_monedas():
    global moneda_positions, moneda, axis, t_gc, t_cw, R_cg, R_gc, R_cw, t_cw, params

    # --- UDP para recibir monedas ---
    server_ip = '127.0.0.1'
    port = 5005
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((server_ip, port))
    print("Esperando mensaje UDP con posiciones de monedas...")

    try:
        data, _ = sock.recvfrom(1024)
    except socket.error as e:
        print("Error recibiendo UDP:", e)
        return

    mensaje = data.decode('utf-8')
    print("Mensaje recibido:", mensaje)
    moneda_positions = [np.fromstring(m, sep=',') for m in mensaje.strip().split(';')]

    data = np.load("calibration_ground.npz")
    rvec = R[0].reshape(3)
    tvec = T.reshape(3,1)

    R_wc,_ = cv.Rodrigues(rvec) #de mundoacamara

    R_cw = R_wc.T
    t_cw = -R_wc @ tvec
    
    #de camara a mundo

    R_cg=data['Rg']
    t_cg = data['Origin']
    R_gc = np.linalg.inv(R_cg)
    t_gc = -R_cg @ t_cg



    p0 = np.array([0.0,0.0,0.0])
    p1 = np.array([1.0,0.0,0.0])
    p2 = np.array([0.0,1.0,0.0])


    # --- Transformaciones de cámara

    x_axis = (p1 - p0)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = (p2 - p0)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    z_axis = -z_axis
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    R_cam = np.stack([x_axis, y_axis, z_axis], axis=1)
    t_cam = p0.reshape((3, 1))

    


    T_matrix = np.eye(4)
    T_matrix[:3, :3] = R_cam
    T_matrix[:3, 3] = t_cam[:, 0]

    extrinsic = np.linalg.inv(T_matrix)
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2])

    params = o3d.camera.PinholeCameraParameters()
    params.intrinsic = intrinsic
    params.extrinsic = extrinsic

    # --- Cargar moneda 3D
    moneda = o3d.io.read_triangle_mesh("moneda.ply")
    moneda.scale(1, center=moneda.get_center())
    color = [1.0, 1.0, 0.0]  # Amarillo
    moneda.paint_uniform_color(color)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

def inicializar_datosrobot():
    global sock_robot
    robot_ip = '127.0.0.1'
    robot_port = 6006
    sock_robot = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_robot.bind((robot_ip, robot_port))
    sock_robot.setblocking(False)


def visualizer():
    global vis, robot_cube, monedas, params
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(axis)
    monedas = []

    for pos in moneda_positions:
        moneda_copia = copy.deepcopy(moneda)
        pos = np.array([pos[0],pos[1],pos[2]])
        p_world = ground_to_world(pos)
        moneda_copia.translate(p_world, relative=False)
        vis.add_geometry(moneda_copia)
        monedas.append(moneda_copia)

    robot_cube = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.2, depth=0.2)
    robot_cube.paint_uniform_color([0.5, 0.5, 0.5])
    robot_cube.translate([1, 1, 0], relative=False)
    vis.add_geometry(robot_cube)

    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
    vis.get_render_option().background_color = np.array([0, 0, 0])

def fusion_images(frame):
    global vis, robot_cube, monedas

    frame = cv.resize(frame, (width, height))

    try:
        data, _ = sock_robot.recvfrom(1024)
        mensaje = data.decode('utf-8')
        pos = np.fromstring(mensaje, sep=',')
        if pos.shape == (3,):
            robot_cube.translate(pos - robot_cube.get_center(), relative=True)
            robot_cube.compute_vertex_normals()
            vis.update_geometry(robot_cube)
    except BlockingIOError:
        pass

    angle = math.radians(20)
    Rz = moneda.get_rotation_matrix_from_axis_angle([0, 0, angle])

    for moneda_copia in monedas:
        moneda_copia.rotate(Rz, center=moneda_copia.get_center())
        vis.update_geometry(moneda_copia)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image("render.png")
    
    render = Image.open("render.png").convert("RGBA")
    render_np = np.array(render)
    render_bgra = render_np[:, :, [2, 1, 0, 3]]

    frame_rgba = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)

    mask = ~((render_np[:, :, 0] == 0) & (render_np[:, :, 1] == 0) & (render_np[:, :, 2] == 0))
    mask = mask.astype(np.uint8)

    for c in range(3):
        frame_rgba[:, :, c] = np.where(mask, render_bgra[:, :, c], frame_rgba[:, :, c])

    frame_rgba[:, :, 3] = 255
    return frame_rgba




def rectify_images(img1, img2, K1, D1, K2, D2, R, T):
    size = (img1.shape[1], img1.shape[0])
    R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(K1, D1, K2, D2, size, R, T, flags=cv.CALIB_ZERO_DISPARITY, alpha=0)
    map1x, map1y = cv.initUndistortRectifyMap(K1, D1, R1, P1, size, cv.CV_32FC1)
    map2x, map2y = cv.initUndistortRectifyMap(K2, D2, R2, P2, size, cv.CV_32FC1)
    img1_rect = cv.remap(img1, map1x, map1y, cv.INTER_LINEAR)
    img2_rect = cv.remap(img2, map2x, map2y, cv.INTER_LINEAR)
    return img1_rect, img2_rect, P1, P2, Q

def segment_black_objects(img):
    # 1. Convertir a escala de grises
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 2. Suavizado para evitar ruido
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # 3. Umbral fijo ligeramente más bajo (mejor para negros reales)
    _, binary_mask = cv.threshold(blurred, 60, 255, cv.THRESH_BINARY_INV)

    # 4. Filtrado HSV más permisivo pero útil para negros reales
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 80])  # V=80 permite sombra, sin colarse grises

    mask_hsv = cv.inRange(hsv, lower_black, upper_black)

    # 5. Combinación de máscaras
    combined = cv.bitwise_and(binary_mask, mask_hsv)

    # 6. Morfología: limpiar ruido pero sin pasarse
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv.morphologyEx(combined, cv.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv.morphologyEx(cleaned, cv.MORPH_CLOSE, kernel, iterations=1)

    return cleaned



def extract_ordered_corners(mask, max_corners=10):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    objects = []
    for c in contours:
        if cv.contourArea(c) > 1000:
            obj_mask = np.zeros_like(mask)
            cv.drawContours(obj_mask, [c], -1, 255, -1)
            corners = cv.goodFeaturesToTrack(
                obj_mask,
                maxCorners=max_corners,
                qualityLevel=0.03,
                minDistance=20
            )
            if corners is None or len(corners) < 3:
                continue
            corners = corners.reshape(-1, 2)
            center = np.mean(corners, axis=0)
            angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
            ordered = corners[np.argsort(angles)]
            objects.append({'contour': c, 'vertices_2d': ordered.astype(int)})
    return objects

def triangulate_vertices(objects_left, objects_right, P1, P2):

    P1 = P1.astype(np.float64)
    P2 = P2.astype(np.float64)

    all_objects_3d = []
    for objL, objR in zip(objects_left, objects_right):
        vertices3d = []
        for vL, vR in zip(objL['vertices_2d'], objR['vertices_2d']):
            ptL = np.array(vL, dtype=np.float32).reshape(2, 1)
            ptR = np.array(vR, dtype=np.float32).reshape(2, 1)
            X = cv.triangulatePoints(P1, P2, ptL, ptR)
            X /= X[3]
            vertices3d.append(X[:3].flatten().tolist())
        vertices3d = np.array(vertices3d)
        altura = float(np.max(vertices3d[:, 2]) - np.min(vertices3d[:, 2]))
        all_objects_3d.append({'vertices': vertices3d.tolist(), 'height': altura})
    return all_objects_3d

# --- CLASE ZEDCamera ---

class ZEDCamera():
    def __init__(self, camera_name, fps, left_path, right_path):
        self.name = 'camera_' + camera_name
        self.fps = fps
        self.left_path = left_path
        self.right_path = right_path
        self.img_count = 0
        self.cam = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        init_params.coordinate_units = sl.UNIT.METER
        init_params.camera_fps = self.fps
        status = self.cam.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print("Error al abrir la cámara ZED1")
            exit(1)

    def capture_images(self):
        runtime = sl.RuntimeParameters()
        mat = sl.Mat()
        key = -1
        K1, D1, K2, D2, R, T = load_stereo_calibration('calibration_stereo_data.npz')
        img_left = None
        img_right = None

        while key != 113:  # 'q'
            err = self.cam.grab(runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                for orientation in ["l", "r"]:
                    if orientation == "l":
                        self.cam.retrieve_image(mat, sl.VIEW.LEFT)
                        img_left = mat.get_data()
                    else:
                        self.cam.retrieve_image(mat, sl.VIEW.RIGHT)
                        img_right = mat.get_data()

                self.img_count += 1
                imgL_rect, imgR_rect, P1, P2, Q = rectify_images(img_left, img_right, K1, D1, K2, D2, R, T)

                # Segmentación
                maskL = segment_black_objects(imgL_rect)
                maskR = segment_black_objects(imgR_rect)

                # Aplicar Canny sobre la máscara limpia
                edgesL = cv.Canny(maskL, 100, 200)
                edgesR = cv.Canny(maskR, 100, 200)

                # Extraer vértices sobre bordes
                objectsL = extract_ordered_corners(edgesL)
                objectsR = extract_ordered_corners(edgesR)

                # Triangulación 3D
                objects_3d = triangulate_vertices(objectsL, objectsR, P1, P2)
                #print(objects_3d)

                with open('objetos_3d.pkl', 'wb') as f:
                    pickle.dump(objects_3d, f)

                # Dibujar sobre imagen original rectificada
                for obj in objectsL:
                    cv.drawContours(imgL_rect, [obj['contour']], -1, (0, 255, 0), 2)
                    for v in obj['vertices_2d']:
                        cv.circle(imgL_rect, tuple(v), 5, (0, 0, 255), -1)

                imagen_AR = fusion_images(imgL_rect)

                point = (0,0,0)

                data = np.load("calibration_ground.npz")
                Rg=data['Rg']
                tr = data['Origin']
                Rg = np.linalg.inv(Rg)
                point -= tr
                point = Rg@point
                point2d, _ = cv.projectPoints(point, R, T, K1, None)
                u,v = int(point2d[0][0][0]), int(point2d[0][0][1])
                cv.circle(imagen_AR, (u,v), 10, (0,255,0), 10)

                cv.imshow('Left Rectified - Objetos detectados', imagen_AR)
                #cv.imshow('Canny Left', edgesL)

                key = cv.waitKey(1)
            else:
                print(f"Error al capturar la imagen: {err}")
                break

        self.cam.close()
        cv.destroyAllWindows()

# --- MAIN ---

def main():
    global K1, D1, K2, D2, R, T

    camera_name = "ZED1"
    fps = 30
    save_left_path = "C:/Users/IUBIMUO/Desktop/v3d/capturas_left"
    save_right_path = "C:/Users/IUBIMUO/Desktop/v3d/capturas_right"

    load_stereo_calibration('calibration_stereo_data.npz')  # inicializa K1...T
    cargar_monedas()
    inicializar_datosrobot()
    visualizer()

    zed_camera = ZEDCamera(camera_name, fps, save_left_path, save_right_path)
    zed_camera.capture_images()


if __name__ == '__main__':
    main()
