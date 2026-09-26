import open3d as o3d
import numpy as np
import cv2
from PIL import Image
import socket
import copy
import math
import os
import time
frame = None

def receive_frame(image):
    frame = image.copy()
# --- FUNCIONES ---
def load_calibration(filename):
    with np.load(filename) as data:
        K = data['mtx']
        dist = data['dist']
        rvec = data['rvecs'][0]
        tvec = data['tvecs'][0]
        R, _ = cv2.Rodrigues(rvec)
        RT = np.hstack([R, tvec.reshape(3, 1)])
    return K, RT, dist

def load_ground_calibration(filename):
    with np.load(filename) as data:
        R_ground2cam = data['Rg']         # Matriz de rotación 3x3
        t_ground2cam = data['Origin']     # Vector de traslación (3,)
    return R_ground2cam, t_ground2cam

def world_to_cam(P_world, R, t):
    return R @ P_world + t


# --- CARGA CALIBRACIÓN ---
K, RT, dist = load_calibration('calibration_left.npz')
R = RT[:, :3]
t = RT[:, 3]

width, height = 640, 480

# --- MATRIZ EXTRÍNSECA WORLD -> CAM ---
P0_world = np.array([0, 0, 0])
P1_world = np.array([1, 0, 0])
P2_world = np.array([0, 1, 0])
p0 = world_to_cam(P0_world, R, t)
p1 = world_to_cam(P1_world, R, t)
p2 = world_to_cam(P2_world, R, t)

x_axis = (p1 - p0) / np.linalg.norm(p1 - p0)
y_axis = (p2 - p0) / np.linalg.norm(p2 - p0)
z_axis = -np.cross(x_axis, y_axis)
z_axis /= np.linalg.norm(z_axis)
y_axis = np.cross(z_axis, x_axis)
y_axis /= np.linalg.norm(y_axis)

R_cam = np.stack([x_axis, y_axis, z_axis], axis=1)
t_cam = p0.reshape((3, 1))

T = np.eye(4)
T[:3, :3] = R_cam
T[:3, 3] = t_cam[:, 0]
extrinsic = np.linalg.inv(T)

# --- GROUND CALIBRATION ---
R_ground2cam, t_ground2cam = load_ground_calibration('calibration_ground.npz')

T_ground2cam = np.eye(4)
T_ground2cam[:3, :3] = R_ground2cam
T_ground2cam[:3, 3] = t_ground2cam


T_cam2world = np.linalg.inv(extrinsic)

# --- INTRÍNSECOS OPEN3D ---
intrinsic = o3d.camera.PinholeCameraIntrinsic()
intrinsic.set_intrinsics(width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
params = o3d.camera.PinholeCameraParameters()
params.intrinsic = intrinsic
params.extrinsic = extrinsic


def ground_to_world(P_ground):
    # P_ground: (3,)  -> (4,1) homogéneo
    P_ground_h = np.ones(4)
    P_ground_h[:3] = P_ground
    # Ground -> Cam
    P_cam_h = T_ground2cam @ P_ground_h
    # Cam -> World
    P_world_h = T_cam2world @ P_cam_h
    # Devuelve solo las 3 primeras componentes (x, y, z)
    return P_world_h[:3]

# --- UDP CLIENTE: MONEDAS ---
server_ip, port = '127.0.0.1', 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((server_ip, port))
print("Esperando mensaje UDP con posiciones de monedas...")
data, _ = sock.recvfrom(1024)
mensaje = data.decode('utf-8')
print("Mensaje recibido:", mensaje)
moneda_positions = [np.fromstring(m, sep=',') for m in mensaje.strip().split(';')]

# --- UDP ROBOT ---
robot_ip, robot_port = '127.0.0.1', 6006
sock_robot = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_robot.bind((robot_ip, robot_port))
sock_robot.setblocking(False)

# --- MODELO MONEDA ---
moneda = o3d.io.read_triangle_mesh("moneda.ply")
moneda.scale(10, center=moneda.get_center())
amarillo = [1.0, 1.0, 0.0]
moneda.vertex_colors = o3d.utility.Vector3dVector(np.tile(amarillo, (len(moneda.vertices), 1)))

# --- VISUALIZER ---
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
robot_cube = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
robot_cube.paint_uniform_color([0.5, 0.5, 0.5])
robot_cube.translate([1, 1, 0], relative=False)

vis = o3d.visualization.Visualizer()
vis.create_window(width=width, height=height, visible=False)
vis.add_geometry(axis)
vis.add_geometry(robot_cube)


# --- SISTEMA DE COORDENADAS DEL GROUND ---
# Creamos el frame de coordenadas del ground
ground_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

# Creamos la matriz de transformación ground->world (igual que ground_to_world pero homogénea)
T_ground2world = T_cam2world @ T_ground2cam

# Aplicamos la transformación al frame
ground_frame.transform(T_ground2world)

# Añadimos el frame al visualizador
vis.add_geometry(ground_frame)

monedas = []

 
for pos in moneda_positions:
    
    pos_world = ground_to_world(pos)  # <--- TRANSFORMACIÓN AQUÍ
    moneda_copia = copy.deepcopy(moneda)
    moneda_copia.translate(pos_world, relative=False)
    vis.add_geometry(moneda_copia)
    #monedas.append(moneda_copia)
    monedas.append({'obj': moneda_copia, 'pos': pos_world})


ctr = vis.get_view_control()
ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
vis.get_render_option().background_color = np.array([0, 0, 0])



# Antes del bucle principal
last_time = time.time()

monedas_recolectadas = 0
start_time = time.time()
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(("localhost", 5030))
while True:
    umbral = 1
    data, addr = sock.recvfrom(65535)
    # Convertir bytes a numpy array
    np_arr = np.frombuffer(data, dtype=np.uint8)
    # Decodificar la imagen JPEG
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        print("Error al decodificar la imagen")
        continue
    frame = cv2.resize(frame, (width, height))
    pos_world_rob = [0.0, 0.0, 0.0]
    try:
        data, _ = sock_robot.recvfrom(1024)
        pos = np.fromstring(data.decode('utf-8'), sep=',')
        pos_world_rob = ground_to_world(pos)
        if pos.shape == (3,):
            robot_cube.translate(pos_world_rob - robot_cube.get_center(), relative=True)
            robot_cube.compute_vertex_normals()
            vis.update_geometry(robot_cube)
    except BlockingIOError:
        pass

    # --- ROTACIÓN DE MONEDAS (antes de eliminar) ---
    if monedas:  # Solo si hay monedas
        current_time = time.time()
        delta_time = current_time - last_time
        last_time = current_time
        angle = math.radians(60) * delta_time
        Rz = monedas[0]['obj'].get_rotation_matrix_from_axis_angle([0, 0, angle])
        for moneda_dict in monedas:
            moneda_obj = moneda_dict['obj']
            moneda_obj.rotate(Rz, center=moneda_obj.get_center())
            vis.update_geometry(moneda_obj)
    else:
        current_time = time.time()
        delta_time = current_time - last_time
        last_time = current_time

        # --- ELIMINACIÓN DE MONEDAS ---
    monedas_a_eliminar = []
    for moneda in monedas:
        distancia = np.linalg.norm(np.array(pos_world_rob) - moneda['pos'])
        if distancia < umbral:
            vis.remove_geometry(moneda['obj'])
            monedas_a_eliminar.append(moneda)

    for moneda in monedas_a_eliminar:
        monedas.remove(moneda)

    if monedas_a_eliminar:
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        monedas_recolectadas += len(monedas_a_eliminar)

    vis.poll_events()
    vis.update_renderer()
    
    vis.capture_screen_image("render.png")

    # Superposición
    render = Image.open("render.png").convert("RGBA")
    render_np = np.array(render)
    render_bgra = render_np[:, :, [2, 1, 0, 3]]
    frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    mask = ~((render_np[:, :, 0] == 0) & (render_np[:, :, 1] == 0) & (render_np[:, :, 2] == 0))
    mask = mask.astype(np.uint8)
    for c in range(3):
        frame_rgba[:, :, c] = np.where(mask, render_bgra[:, :, c], frame_rgba[:, :, c])
    frame_rgba[:, :, 3] = 255


    # --- DIBUJAR CONTADORES ---
    # Monedas recolectadas (arriba izquierda)
    cv2.putText(
        frame_rgba,
        f"Monedas: {monedas_recolectadas}",
        (20, 40),  # posición (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,  # tamaño de fuente
        (0, 255, 255, 255),  # color (amarillo, BGRA)
        3,  # grosor
        cv2.LINE_AA
    )

    # Tiempo transcurrido (arriba derecha)
    elapsed = int(time.time() - start_time)
    minutos = elapsed // 60
    segundos = elapsed % 60
    texto_tiempo = f"Tiempo: {minutos:02d}:{segundos:02d}"
    (texto_w, texto_h), _ = cv2.getTextSize(texto_tiempo, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    cv2.putText(
        frame_rgba,
        texto_tiempo,
        (frame_rgba.shape[1] - texto_w - 20, 40),  # posición (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 255, 255),  # color (amarillo, BGRA)
        3,
        cv2.LINE_AA
    )
    
    cv2.imshow("Realidad Aumentada", frame_rgba)
    frame = None
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vis.destroy_window()
cv2.destroyAllWindows()