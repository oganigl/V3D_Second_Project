import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import socket
import struct

IP = '192.168.185.195' 
PORT = 4040

class OdomUDPNode(Node):
    def __init__(self):
        super().__init__('odom_udp_node')         

        # Crear socket UDP
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Suscribirse al topic /odom
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        self.get_logger().info(f'Odom')

    def odom_callback(self, msg: Odometry):
        # Obtener posición (x, y, orientación en z)
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.orientation.z  # orientación como quaternion (z)

        # Empaquetar y enviar por UDP (como 3 floats)
        msg = f'{x},{y},{z}'
        self.sock.sendto(msg.encode('utf-8'), (IP,PORT))
        self.get_logger().info(f'Odom {msg}')


def main():
    rclpy.init()
    node = OdomUDPNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
