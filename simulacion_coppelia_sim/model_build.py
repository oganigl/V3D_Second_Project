from time import sleep
import math
import struct
import socket
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require('sim')
objects = []
handler_mesh = []

def clear_enviroment():
    idx = 0
    while idx < len(handler_mesh):
        handler = handler_mesh[idx]
        sim.removeObject(handler)
        idx += 1

    handler_mesh.clear()

def create_object(object):
    vertices = [
        object[0][0], object[0][1], object[0][2],
        object[1][0], object[1][1], object[1][2],
        object[2][0], object[2][1], object[2][2],
        object[3][0], object[3][1], object[3][2], 
        object[0][0], object[0][1], 0,
        object[1][0], object[1][1], 0,
        object[2][0], object[2][1], 0,
        object[3][0], object[3][1], 0 
    ]

    indices = [
        0, 1, 4,  4, 5, 1,  
        1, 2, 5,  5, 6, 2, 
        2, 3, 6,  6, 7, 3,  
        3, 0, 7,  7, 4, 0, 
        0, 1, 3,  1, 3, 2
    ]

    shapeHandle = sim.createMeshShape(0, 8, vertices, indices)
    handler_mesh.append(shapeHandle)

def build_enviroment():
    clear_enviroment()
    idx = 0
    while idx < len(objects):
        create_object(objects[idx])
        idx += 1

    objects.clear()

def unpack_packet(raw_data):
    idx = 0
    object = []
    #4 points for each figure
    object_count = 0
    while idx < len(raw_data):
        x, y, z = struct.unpack('fff', data[idx:idx+4*3])
        object_count += 1
        object.append((x, y, z))
        if(object_count == 4):
            object_count = 0
            objects.append(object.copy())
            object.clear()
        idx+=4*3

    build_enviroment()

#conexion tcp
host = "192.168.185.196"
port = 5050
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind((host, port))

data_buff = b''
while 1:
    data, _ = s.recvfrom(1024)
    data_buff += data
    if len(data) == 1024:
        continue
    else:
        unpack_packet(data_buff)
    
    
    data_buff = b''

        

    