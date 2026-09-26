import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time
import socket
import threading
import struct

PORT = 3033

class VelocityPublisher(Node):
    def __init__(self):
        super().__init__('velocity_publisher')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)

        self.ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ss.bind(('192.168.185.196', PORT))

        # Iniciar hilo para escuchar
        self.thread = threading.Thread(target=self.listen_socket)
        self.thread.daemon = True
        self.thread.start()

    def send_velocity(self, linear=0.0, angular=0.0):
        msg = Twist()
        msg.linear.x = linear
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = angular
        self.publisher_.publish(msg)

    def listen_socket(self):
        self.ss.listen(1)
        conn, add = self.ss.accept()
        self.get_logger().info(f'Conexión aceptada desde {add}')
        
        while rclpy.ok():
            data = conn.recv(1024)

            if len(data) != 4:
                continue

            key = struct.unpack('i', data)[0]
            self.get_logger().info(f'key {key}')
            
            if key == 1:
                self.send_velocity(0.0, 0.0)

            elif key == 0:
                self.send_velocity(20.0, 0.0)

            elif key == 2:
                self.send_velocity(0.0, 20.0)


def main():
    rclpy.init()
    node = VelocityPublisher()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
