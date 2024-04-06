from os import getcwd
import sys
sys.path.append(getcwd())
from config.libaries import *

def cal_degree(interest_point :list, middle :list):
    # middle_center = [df[df['cls'] == 'middle']['x'].to_list()[0], df[df['cls'] == 'middle']['y'].to_list()[0]]

    dx = interest_point[0] - middle[0]
    dy = interest_point[1] - middle[1]

    degree_temp = math.atan2(dy,dx)
    degree_temp = math.degrees(degree_temp)

    return degree_temp


def value_calculation(start, end, tips, middle, start_value, max_value):
    start_temp = cal_degree(start, middle)
    end_temp = cal_degree(end, middle)
    tips_temp = cal_degree(tips, middle)

    remove_degree_start = start_temp - 90
    remove_degree_end = 90 - end_temp
    overall_degree = 360 - (remove_degree_start + remove_degree_end)

    print(tips_temp)
    if tips_temp < 0:
        tips_temp = (180 - abs(tips_temp)) + 180
    elif tips_temp < 90:
        tips_temp = 270 + tips_temp + 90
    
    print(tips_temp)
        
    temp_tips_degree = tips_temp - 90 - remove_degree_start
    step_incre = max_value / overall_degree
    value = temp_tips_degree * step_incre

    return value - start_value