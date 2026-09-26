import cv2 as cv
import numpy as np


def build_ground_frame(points):
    centroid = np.mean(points, axis=0)  # punto medio
    centered = points - centroid        # centra la nube
    _, _, vh = np.linalg.svd(centered)
    normal = vh[-1]                    
    z_axis =normal / np.linalg.norm(normal)

    # Elige un vector arbitrario no paralelo a Z'
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(arbitrary, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=0)  # 3x3 matriz de rotación
    return R, centroid

def transform_to_ground(points, R, origin):
    """
    Transforma puntos desde sistema de cámara al sistema del suelo
    """
    translated = points - origin  # traslada respecto al origen del suelo
    transformed = R @ translated.T  # aplica rotación
    return transformed.T  # retorna Nx3


def rectify_images(img1, img2, K1, D1, K2, D2, R, T):
    size = (img1.shape[1], img1.shape[0])
    R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(K1, D1, K2, D2, size, R, T, flags=cv.CALIB_ZERO_DISPARITY, alpha=0)
    map1x, map1y = cv.initUndistortRectifyMap(K1, D1, R1, P1, size, cv.CV_32FC1)
    map2x, map2y = cv.initUndistortRectifyMap(K2, D2, R2, P2, size, cv.CV_32FC1)
    img1_rect = cv.remap(img1, map1x, map1y, cv.INTER_LINEAR)
    img2_rect = cv.remap(img2, map2x, map2y, cv.INTER_LINEAR)
    return img1_rect, img2_rect, P1, P2, Q

def extract_vertices_chessboard(img, pattern_size=(8, 6)):
    """
    Extrae las esquinas internas del patrón de tablero de ajedrez.

    :param img: imagen de entrada (BGR o escala de grises)
    :param pattern_size: número de esquinas internas (columnas, filas)
    :return: np.ndarray de forma (N, 2) con coordenadas (x, y)
    """
    # Convertir a escala de grises si es necesario
    if len(img.shape) == 3:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img

    # Buscar patrón de tablero de ajedrez
    found, corners = cv.findChessboardCorners(gray, pattern_size, None)

    if not found:
        print("No se encontraron esquinas.")
        return None

    # Refinar las coordenadas de las esquinas
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    return corners_refined.reshape(-1, 2)

data = np.load("calibration_stereo_data.npz")
K1 = data['K1']
D1 = data['D1']
K2 = data['K2']
D2 = data['D2']
R = data['R']
T = data['T']

imgL = cv.imread("capturas/left/img_000744.jpg")
imgR = cv.imread("capturas/right/img_000744.jpg")


imgL_rect, imgR_rect, P1, P2, Q = rectify_images(imgL, imgR, K1, D1, K2, D2, R, T)

cornersL = extract_vertices_chessboard(imgL_rect)
cornersR = extract_vertices_chessboard(imgR_rect)
if cornersL is not None and cornersR is not None:
    for i, pt in enumerate(cornersL):
        pt = tuple(np.round(pt).astype(int))
        cv.circle(imgL_rect, pt, 5, (0, 255, 0), -1)
        cv.putText(imgL_rect, str(i), pt, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    for i, pt in enumerate(cornersR):
        pt = tuple(np.round(pt).astype(int))
        cv.circle(imgR_rect, pt, 5, (0, 255, 0), -1)
        cv.putText(imgR_rect, str(i), pt, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    
    points4D = cv.triangulatePoints(P1, P2, cornersL.T, cornersR.T)
    points3D = (points4D / points4D[3])[:3].T

    # Rg, origin = build_ground_frame(points3D)

    # np.savez("calibration_ground.npz", Rg=Rg, Origin=origin)


    ground = np.load('calibration_ground.npz')
    Rg = ground['Rg']
    Origin = ground['Origin']

    points3D = transform_to_ground(points3D, Rg, Origin)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # Necesario para proyecciones 3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Desempaquetamos coordenadas X, Y, Z
    X = points3D[:, 0]
    Y = points3D[:, 1]
    Z = points3D[:, 2]

    # Dibujamos los puntos
    ax.scatter(X, Y, Z, c='r', marker='o')

    # Etiquetas opcionales
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Puntos 3D triangulados')

    plt.show()

    cv.imshow("Left with corners", imgL_rect)
    cv.imshow("Right with corners", imgR_rect)
    cv.waitKey(0)
    cv.destroyAllWindows()



