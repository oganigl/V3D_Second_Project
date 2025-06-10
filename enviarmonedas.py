import socket

import time

 

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

server_ip = '127.0.0.1'

port = 5005

 

while True:

    mensaje = "1.0,0.0,2.0;2.0,0.0,2.0;1.0,1.0,2.0;0.1,0.2,1.0;0.2,0.1,1.0;0.0,0.0,1.2"

    sock.sendto(mensaje.encode('utf-8'), (server_ip, port))

    time.sleep(0.5)