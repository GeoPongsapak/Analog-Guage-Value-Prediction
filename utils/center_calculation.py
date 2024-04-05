from os import getcwd
import sys
sys.path.append(getcwd())
from config.libaries import *

def find_intersection_point(m1, c1, m2, c2):
    x = (c2 - c1) / (m1 - m2)
    y = m1 * x + c1

    return [x, y]
        
def cal_center(point_b, point_c, distance_b, distance_c, angle_b, angle_c):

    x = point_b[0]
    y = point_b[1]
    endxb = x + distance_b * math.cos(math.radians(angle_b))
    
    endyb = y + distance_b * math.sin(math.radians(angle_b))


    x = point_c[0]
    y = point_c[1]

    endxc = x + distance_c * math.cos(math.radians(angle_c))
    
    endyc = y + distance_c * math.sin(math.radians(angle_c)) 

    m1 = (endyc - point_c[1]) / (endxc - point_c[0])
    c1 = point_c[1] - m1 * point_c[0]

    m2 = (point_b[1] - endyb) / (point_b[0] - endxb)
    c2 = point_b[1] - m2 * point_b[0]

    return find_intersection_point(m1, c1, m2, c2)

def one_quadant_cal_center(point_b, point_c, distance_c, angle_c):
    x = point_c[0]
    y = point_c[1]

    endxc = x + distance_c * math.cos(math.radians(angle_c))
    
    endyc = y + distance_c * math.sin(math.radians(angle_c)) 

    m1 = (endyc - point_c[1]) / (endxc - point_c[0])
    c1 = point_c[1] - m1 * point_c[0]

    m2 = (point_b[1] - point_b[1]) / (0 - point_b[0])
    c2 = point_b[1] - m2 * point_b[0]
    return find_intersection_point(m1, c1, m2, c2)