import yaml
import numpy as np
from numpy.linalg import norm
from math import pi
import rospy

WAYPOINT_YAW = 100
OBSTACLE = 100
UNKNOWN = -1

def read_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def index_of_point(mapData, Xp):
    resolution = mapData.info.resolution
    Xstartx = mapData.info.origin.position.x
    Xstarty = mapData.info.origin.position.y
    width = mapData.info.width
    Data = mapData.data
    index = int((np.floor((Xp[1]-Xstarty)/resolution) *
                  width)+(np.floor((Xp[0]-Xstartx)/resolution)))
    return index

def _world_to_map_ori(_map, world_pose):
    wx, wy = world_pose[:2]
    # print(_map.info.origin.position.x, _map.info.origin.position.y)  # -9.9 -3.2499999046325683
    if (wx < _map.info.origin.position.x or wy < _map.info.origin.position.y):
        rospy.logwarn("World coordinates out of bounds")  # raise Exception
        # return None

    mx = int((wx - _map.info.origin.position.x) / _map.info.resolution)
    my = int((wy - _map.info.origin.position.y) / _map.info.resolution)
    
    if  (my > _map.info.height or mx > _map.info.width):
        rospy.logwarn("Map height or width out of bounds")  # raise Exception
        # return None
    return (mx, my)  #! to np.array index

def _world_to_map(_map, world_pose):
    wx, wy = world_pose[:2]
    # print(_map.info.origin.position.x, _map.info.origin.position.y)  # -9.9 -3.2499999046325683
    if (wx < _map.info.origin.position.x or wy < _map.info.origin.position.y):
        rospy.logwarn(f"{world_pose}:World coordinates out of bounds")  # raise Exception
        # return None
        rospy.loginfo(f'origin_x:{_map.info.origin.position.x}, origin_y:{_map.info.origin.position.y}')
        return ()
    mx = int((wx - _map.info.origin.position.x) / _map.info.resolution)
    my = int((wy - _map.info.origin.position.y) / _map.info.resolution)
    
    if  my >= _map.info.height or mx >= _map.info.width:
        # rospy.loginfo(f'{world_pose}!!!')
        rospy.logwarn(f"{world_pose}:Map height or width out of bounds")  #raise Exception
        rospy.loginfo(f'map height:{_map.info.height}, width:{_map.info.width}')
        # return None
        return ()
    if my == _map.info.height:
        my -= 1
    if mx == _map.info.width:
        mx -= 1
    return (my, mx)  #! to np.array index

def _map_to_world(_map, map_pose):
    my, mx = map_pose
    wy = (my + 0.5) * _map.info.resolution + _map.info.origin.position.y
    wx = (mx + 0.5)  * _map.info.resolution + _map.info.origin.position.x
    world_pose = [wx, wy]
    if (wx < _map.info.origin.position.x or wy < _map.info.origin.position.y):
        rospy.logwarn(f"{world_pose}:World coordinates out of bounds")  # raise Exception
    if  my > _map.info.height or mx > _map.info.width:
        # rospy.loginfo(f'{world_pose}!!!')
        rospy.logwarn(f"{world_pose}:Map height or width out of bounds")  #raise Exception
    return world_pose
    
    

def point_of_index(mapData, i):
    y = mapData.info.origin.position.y + \
        (i/mapData.info.width)*mapData.info.resolution
    x = mapData.info.origin.position.x + \
        (float(i-(int(i/mapData.info.width)*(mapData.info.width)))*mapData.info.resolution)
    return np.array([x, y])


def informationGain(mapData, point, r):
    infoGain = 0.0
    index = index_of_point(mapData, point)
    r_region = int(r/mapData.info.resolution)
    init_index = index-r_region*(mapData.info.width+1)
    for n in range(0, 2*r_region+1):
        start = n*mapData.info.width+init_index
        end = start+2*r_region
        limit = ((start/mapData.info.width)+2)*mapData.info.width
        for i in range(start, end+1):
            if (i >= 0 and i < limit and i < len(mapData.data)):
                if(mapData.data[i] == -1 and norm(np.array(point)-point_of_index(mapData, i)) <= r):
                    infoGain += 1.0
    return infoGain*(mapData.info.resolution**2)


def _to_radians(angle):
    return angle / 180 * pi
        

def informationGainFOV(mapData, point, r, stride=15, hfov_angle=70.5, count_state=-1):
    infoGain = 0.0
    index = index_of_point(mapData, point)
    r_region = int(r/mapData.info.resolution)
    init_index = index-r_region*(mapData.info.width+1)
    unknown_angles = []
    for n in range(0, 2*r_region+1):
        start = n*mapData.info.width+init_index
        end = start+2*r_region
        limit = ((start/mapData.info.width)+2)*mapData.info.width
        for i in range(start, end+1):
            if (i >= 0 and i < limit and i < len(mapData.data)):
                p_i = point_of_index(mapData, i)
                if(mapData.data[i] == count_state and norm(np.array(point)-p_i) <= r):
                    angle = np.arctan2(p_i[1] - point[1], p_i[0] - point[0])  # odom coord
                    # angle = _norm_to_2pi(angle - pi / 2)  # [0, 2*pi]
                    unknown_angles.append(angle)
                    infoGain += 1.0
    if not len(unknown_angles):
        return 0, 0
    unknown_angles = sorted(unknown_angles) # [-pi, pi]
    # unknown_angles = [_normalize_heading(angle - pi / 2) for angle in unknown_angles]
    # slice_num = 12
    fov = _to_radians(hfov_angle)
    fov_gain = []
    lowb = -pi
    highb = -pi + fov
    l, r = 0, 0
    pre_l, pre_r = 0, 0
    num = 0
    stride_rad = _to_radians(stride)
    # rospy.loginfo(f'unknown_angles:{unknown_angles}')
    for i in range(int(360 / stride)):
        lowb += i * stride_rad
        highb += i * stride_rad
        # for r in range(pre_r, len(unknown_angles)):  # [lowb, highb)
        while l < len(unknown_angles):
            if unknown_angles[l] >= lowb:
                num -= l - pre_l
                pre_l = l
                break
            l += 1
        # rospy.loginfo(f'l:{num}')
        while True:
            if r >= len(unknown_angles):
                highb -= 2 * pi
            if unknown_angles[r % len(unknown_angles)] > highb:
                num += r - pre_r
                pre_r = r
                break
            r += 1
        # rospy.loginfo(f'r:{num}')
        fov_gain.append(num)
    
    # print(np.mean(fov_gain),np.max(norm_fov_gain), np.min(norm_fov_gain))
    # fov_gain = fov_gain - np.mean(fov_gain)  # normalized to 0-1
    max_fov_gain = np.max(fov_gain)
    argmax_yaw = -pi + stride_rad * fov_gain.index(max_fov_gain) + fov / 2
    if argmax_yaw > pi:
        argmax_yaw -= 2 * pi
    return max_fov_gain*(mapData.info.resolution**2), argmax_yaw  # 


def _normalize_heading(heading):
    if heading >= pi:
        heading -= 2 * pi
    elif heading < -pi:
        heading += 2 * pi
    return heading

def bresenham(start, goal):
    # https://github.com/daQuincy/Bresenham-Algorithm/blob/master/bresenham.py
    """Yield integer coordinates on the line from (x0, y0) to (x1, y1).
    Input coordinates should be integers.
    The result will contain both the start and the end point.
    """
    x0, y0 = start
    x1, y1 = goal

    
    dx = x1 - x0
    dy = y1 - y0

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2*dy - dx
    y = 0

    for x in range(dx + 1):
        yield x0 + x*xx + y*yx, y0 + x*xy + y*yy
        if D >= 0:
            y += 1
            D -= 2*dx
        D += 2*dy

def get_list_dist(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
def get_elements_by_idxs(elements, idxs):
    if isinstance(elements, np.ndarray):
        return elements[idxs]
    elif isinstance(elements, list):
        return [elements[i] for i in idxs]
    

def unknown_check(pixel, map_array):
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for _dir in dirs:
        neighbor_point = (pixel[0] + _dir[0], pixel[1] + _dir[1])
        if 0 <= neighbor_point[0] < map_array.shape[0] \
            and 0 <= neighbor_point[1] < map_array.shape[1] \
            and map_array[neighbor_point] == UNKNOWN:
            return True
    else:
        return False



import sys
import select
import tty
import termios

class NonBlockingConsole(object):

    def __enter__(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, type, value, traceback):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)


    def get_data(self):
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1)
        return False
