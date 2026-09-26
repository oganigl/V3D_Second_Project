import os
import cv2
import pyzed.sl as sl
import time


class ZEDCamera():
    """
    Cámara ZED2 para capturar imágenes y guardarlas en un directorio.
    """

    def __init__(self, camera_name, fps, left_path, right_path):
        self.name = 'camera_' + camera_name
        self.fps = fps
        self.left_path = left_path
        self.right_path = right_path
        self.img_count = 0

        # Crea el directorio si no existe
        #if not os.path.exists(self.path):
        #    os.makedirs(self.path)

        # Inicializa la cámara ZED
        self.cam = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1080  # Resolución HD1080
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Modo de profundidad
        init_params.coordinate_units = sl.UNIT.METER
        init_params.camera_fps = self.fps
        status = self.cam.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print("Error al abrir la cámara ZED1")
            exit(1)

    def capture_images(self):
        runtime = sl.RuntimeParameters()
        mat = sl.Mat()

        key = -1  # Inicializamos la tecla para evitar que entre en el loop

        while key != 113:  # 'q' para salir
            err = self.cam.grab(runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                # Captura la imagen de la cámara ZED2 
                key = cv2.waitKey(1)  # Espera por 1 ms para una tecla
                for orientation in ["l", "r"]:
                    if orientation == "l": 
                        self.cam.retrieve_image(mat, sl.VIEW.LEFT)
                        filename = os.path.join(self.left_path, f"img_{self.img_count:06d}.jpg")
                    else: 
                        self.cam.retrieve_image(mat, sl.VIEW.RIGHT)
                        filename = os.path.join(self.right_path, f"img_{self.img_count:06d}.jpg")
                    cvImage = mat.get_data()  # Convierte a formato compatible con OpenCV 

                   
                    if key == ord('s'):
                        cv2.imwrite(filename, cvImage)
                        print(f"Imagen guardada como {filename}")
                self.img_count += 1

                # Muestra la imagen en una ventana
                cv2.imshow("ZED Camera - Live Feed", cvImage)
               
            else:
                print(f"Error al capturar la imagen: {err}")
                break

        # Libera los recursos
        self.cam.close()
        cv2.destroyAllWindows()


# Crear una instancia de la clase ZEDCamera
if __name__ == "__main__":
    # Nombre de la cámara, FPS y carpeta de guardado
    camera_name = "ZED1"
    fps =  15 # FPS de la cámara
    save_left_path = "C:/Users/IUBIMUO/Desktop/v3d/images_prueba_r"  # Directorio para guardar las imágenes
    save_right_path = "C:/Users/IUBIMUO/Desktop/v3d/images_prueba_l"

    # Crear objeto de cámara ZED
    zed_camera = ZEDCamera(camera_name, fps, save_left_path, save_right_path)

    # Iniciar la captura de imágenes
    zed_camera.capture_images()

