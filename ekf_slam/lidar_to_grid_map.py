"""

LIDAR to 2D grid map example

author: Erno Horvath, Csaba Hajdu based on Atsushi Sakai's scripts

"""

import math
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

EXTEND_AREA = 1.0


def bresenham(start, end):
    """
    Implementation of Bresenham's line drawing algorithm
    See en.wikipedia.org/wiki/Bresenham's_line_algorithm
    Bresenham's Line Algorithm
    Produces a np.array from start and end (original from roguebasin.com)
    >>> points1 = bresenham((4, 4), (6, 10))
    >>> print(points1)
    np.array([[4,4], [4,5], [5,6], [5,7], [5,8], [6,9], [6,10]])
    """
    # setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    is_steep = abs(dy) > abs(dx)  # determine how steep the line is
    if is_steep:  # rotate line
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    # swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    dx = x2 - x1  # recalculate differentials
    dy = y2 - y1  # recalculate differentials
    error = int(dx / 2.0)  # calculate error
    y_step = 1 if y1 < y2 else -1
    # iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = [y, x] if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += y_step
            error += dx
    if swapped:  # reverse the list if the coordinates were swapped
        points.reverse()
    points = np.array(points)
    return points


def calc_grid_map_config(ox, oy, xy_resolution):
    """
    Calculates the size, and the maximum distances according to the the
    measurement center
    """
    min_x = round(min(ox) - EXTEND_AREA / 2.0)
    min_y = round(min(oy) - EXTEND_AREA / 2.0)
    max_x = round(max(ox) + EXTEND_AREA / 2.0)
    max_y = round(max(oy) + EXTEND_AREA / 2.0)
    xw = int(round((max_x - min_x) / xy_resolution))
    yw = int(round((max_y - min_y) / xy_resolution))
    print("The grid map is ", xw, "x", yw, ".")
    return min_x, min_y, max_x, max_y, xw, yw


def atan_zero_to_twopi(y, x):
    angle = math.atan2(y, x)
    if angle < 0.0:
        angle += math.pi * 2.0
    return angle


def init_flood_fill(center_point, obstacle_points, xy_points, min_coord,
                    xy_resolution):
    """
    center_point: center point
    obstacle_points: detected obstacles points (x,y)
    xy_points: (x,y) point pairs
    """
    center_x, center_y = center_point
    prev_ix, prev_iy = center_x - 1, center_y
    ox, oy = obstacle_points
    xw, yw = xy_points
    min_x, min_y = min_coord
    occupancy_map = (np.ones((xw, yw))) * 0.5
    for (x, y) in zip(ox, oy):
        # x coordinate of the the occupied area
        ix = int(round((x - min_x) / xy_resolution))
        # y coordinate of the the occupied area
        iy = int(round((y - min_y) / xy_resolution))
        free_area = bresenham((prev_ix, prev_iy), (ix, iy))
        for fa in free_area:
            occupancy_map[fa[0]][fa[1]] = 0  # free area 0.0
        prev_ix = ix
        prev_iy = iy
    return occupancy_map


def flood_fill(center_point, occupancy_map):
    """
    center_point: starting point (x,y) of fill
    occupancy_map: occupancy map generated from Bresenham ray-tracing
    """
    # Fill empty areas with queue method
    sx, sy = occupancy_map.shape
    fringe = deque()
    fringe.appendleft(center_point)
    while fringe:
        n = fringe.pop()
        nx, ny = n
        # West
        if nx > 0:
            if occupancy_map[nx - 1, ny] == 0.5:
                occupancy_map[nx - 1, ny] = 0.0
                fringe.appendleft((nx - 1, ny))
        # East
        if nx < sx - 1:
            if occupancy_map[nx + 1, ny] == 0.5:
                occupancy_map[nx + 1, ny] = 0.0
                fringe.appendleft((nx + 1, ny))
        # North
        if ny > 0:
            if occupancy_map[nx, ny - 1] == 0.5:
                occupancy_map[nx, ny - 1] = 0.0
                fringe.appendleft((nx, ny - 1))
        # South
        if ny < sy - 1:
            if occupancy_map[nx, ny + 1] == 0.5:
                occupancy_map[nx, ny + 1] = 0.0
                fringe.appendleft((nx, ny + 1))


def generate_ray_casting_grid_map(ox, oy, xy_resolution, breshen=True):
    """
    The breshen boolean tells if it's computed with bresenham ray casting
    (True) or with flood fill (False)
    """
    min_x, min_y, max_x, max_y, x_w, y_w = calc_grid_map_config(
        ox, oy, xy_resolution)
    # default 0.5 -- [[0.5 for i in range(y_w)] for i in range(x_w)]
    occupancy_map = np.ones((x_w, y_w)) / 2
    center_x = int(
        round(-min_x / xy_resolution))  # center x coordinate of the grid map
    center_y = int(
        round(-min_y / xy_resolution))  # center y coordinate of the grid map
    # occupancy grid computed with bresenham ray casting
    if breshen:
        for (x, y) in zip(ox, oy):
            # x coordinate of the the occupied area
            ix = int(round((x - min_x) / xy_resolution))
            # y coordinate of the the occupied area
            iy = int(round((y - min_y) / xy_resolution))
            laser_beams = bresenham((center_x, center_y), (
                ix, iy))  # line form the lidar to the occupied point
            for laser_beam in laser_beams:
                occupancy_map[laser_beam[0]][
                    laser_beam[1]] = 0.0  # free area 0.0
            occupancy_map[ix][iy] = 1.0  # occupied area 1.0
            occupancy_map[ix + 1][iy] = 1.0  # extend the occupied area
            occupancy_map[ix][iy + 1] = 1.0  # extend the occupied area
            occupancy_map[ix + 1][iy + 1] = 1.0  # extend the occupied area
    # occupancy grid computed with with flood fill
    else:
        occupancy_map = init_flood_fill((center_x, center_y), (ox, oy),
                                        (x_w, y_w),
                                        (min_x, min_y), xy_resolution)
        flood_fill((center_x, center_y), occupancy_map)
        occupancy_map = np.array(occupancy_map, dtype=float)
        for (x, y) in zip(ox, oy):
            ix = int(round((x - min_x) / xy_resolution))
            iy = int(round((y - min_y) / xy_resolution))
            occupancy_map[ix][iy] = 1.0  # occupied area 1.0
            occupancy_map[ix + 1][iy] = 1.0  # extend the occupied area
            occupancy_map[ix][iy + 1] = 1.0  # extend the occupied area
            occupancy_map[ix + 1][iy + 1] = 1.0  # extend the occupied area
    return occupancy_map, min_x, max_x, min_y, max_y, xy_resolution


def main():
    """
    Example usage
    """
    print(__file__, "start")
    xy_resolution = 0.02  # x-y grid resolution
    dist = [2.4617786407470703, 2.4863545894622803, 2.4668729305267334, 2.4845967292785645, 2.4852302074432373, 2.489358425140381, 2.496401309967041, 2.485363006591797, 2.5192477703094482, 2.517054557800293, 2.5293641090393066, 2.5294651985168457, 2.5483758449554443, 2.547100067138672, 2.5478017330169678, 2.5750632286071777, 2.5799720287323, 0.64629065990448, 0.6451490521430969, 0.6615902185440063, 2.637596368789673, 2.642230987548828, 2.6847786903381348, 2.7035584449768066, 2.735398530960083, 2.744340658187866, 2.7629950046539307, 2.780346155166626, 2.824085235595703, 2.8460617065429688, 2.8849527835845947, 2.8899290561676025, 2.9377336502075195, 2.938626289367676, 3.0019874572753906, 3.044933557510376, 3.0998106002807617, 3.1113803386688232, 3.158255100250244, 3.1898446083068848, 3.242030382156372, 3.2869765758514404, 3.3575236797332764, 3.4144160747528076, 3.4759857654571533, 0, 0, 0, 3.4965286254882812, 3.4487106800079346, 3.402766227722168, 3.3555662631988525, 3.3065125942230225, 3.2480478286743164, 3.2020413875579834, 3.1779024600982666, 3.146052837371826, 3.1198856830596924, 3.063035011291504, 3.0215530395507812, 2.9984841346740723, 2.9629948139190674, 2.9173789024353027, 2.922470808029175, 2.907304525375366, 2.866389274597168, 2.855602502822876, 2.830589532852173, 2.802726984024048, 2.780254602432251, 2.771402597427368, 2.76017689704895, 2.743025779724121, 2.7224855422973633, 2.710709810256958, 2.705629348754883, 2.6876790523529053, 2.6727654933929443, 2.66042160987854, 2.649871826171875, 2.6378235816955566, 2.6445648670196533, 2.6241564750671387, 2.627098560333252, 2.629844903945923, 2.6140334606170654, 2.6167984008789062, 2.629750967025757, 2.6116364002227783, 2.596895217895508, 2.600376844406128, 2.61556339263916, 2.6049394607543945, 2.6074986457824707, 2.624521017074585, 2.6319990158081055, 2.6062488555908203, 2.6110055446624756, 2.6332526206970215, 2.6526689529418945, 2.6637113094329834, 2.649282455444336, 2.6626389026641846, 2.6735620498657227, 2.6911826133728027, 2.7095468044281006, 2.715919256210327, 2.7192440032958984, 2.7281277179718018, 2.771251678466797, 2.758944272994995, 2.7821521759033203, 2.803628921508789, 2.842130422592163, 2.8571677207946777, 2.8637101650238037, 2.9010777473449707, 2.9256458282470703, 2.952845335006714, 2.9875335693359375, 3.0187201499938965, 3.054262638092041, 3.0841052532196045, 3.10180401802063, 3.154301881790161, 3.1759753227233887, 3.2306265830993652, 3.253197193145752, 2.985313892364502, 2.8985702991485596, 2.895754098892212, 2.8687992095947266, 2.8707401752471924, 2.8650426864624023, 2.8984172344207764, 2.9461960792541504, 3.0262818336486816, 3.3450779914855957, 3.309370994567871, 3.244704008102417, 3.219618797302246, 3.163801670074463, 3.159853458404541, 3.224663496017456, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.7212831974029541, 0.7457218766212463, 0.7467686533927917, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.525702476501465, 2.5399622917175293, 2.534615993499756, 2.565213918685913, 2.5607361793518066, 2.5742275714874268, 2.5904150009155273, 2.608058214187622, 2.6215834617614746, 2.636756181716919, 2.6451334953308105, 2.659928798675537, 2.698157787322998, 2.6994712352752686, 2.7272982597351074, 2.7483274936676025, 2.7603859901428223, 2.805335521697998, 2.831007719039917, 2.866759777069092, 2.8985939025878906, 2.937747001647949, 2.9697585105895996, 2.9876387119293213, 3.0219860076904297, 0.8924275040626526, 0.8879349231719971, 3.1627840995788574, 3.190265655517578, 3.222788095474243, 3.2882304191589355, 3.3592283725738525, 3.390155076980591, 3.4641294479370117, 0, 0, 0, 0, 0, 3.5, 3.4417123794555664, 3.381105899810791, 3.347339630126953, 3.297213315963745, 3.2618627548217773, 3.22158145904541, 3.192467212677002, 3.161407947540283, 3.1197597980499268, 3.0945396423339844, 3.052821397781372, 3.027252674102783, 3.015601396560669, 2.9700963497161865, 2.9703304767608643, 2.940908670425415, 2.9121596813201904, 2.897351026535034, 2.853785991668701, 2.8615965843200684, 2.844400405883789, 2.82883358001709, 2.807530403137207, 2.8046929836273193, 2.7787864208221436, 2.777743101119995, 2.769038677215576, 2.7522315979003906, 2.7376058101654053, 2.7236380577087402, 2.7267754077911377, 2.7111127376556396, 2.7163751125335693, 2.716425895690918, 2.7075023651123047, 2.684535503387451, 2.70196795463562, 2.6821818351745605, 2.697136640548706, 2.6903743743896484, 2.687049627304077, 2.700535774230957, 2.7047314643859863, 2.6842052936553955, 2.7033915519714355, 2.717046022415161, 2.712043285369873, 2.727897882461548, 2.7469327449798584, 2.7359063625335693, 2.741384506225586, 2.753512382507324, 2.762138843536377, 2.796175479888916, 2.8143229484558105, 2.82735013961792, 2.819671392440796, 2.850403308868408, 2.852167844772339, 2.892355442047119, 2.8937883377075195, 2.9196982383728027, 2.9150257110595703, 2.9684033393859863, 2.9980056285858154, 3.0231473445892334, 3.040192127227783, 3.0806612968444824, 3.109870672225952, 3.1082277297973633, 3.1758923530578613, 3.213108539581299, 3.239436626434326, 3.2662744522094727, 3.3256266117095947, 3.374119997024536, 3.3972527980804443, 3.4430418014526367, 0, 0, 0, 0, 0, 0, 0, 0, 0.805948793888092, 0.8202430009841919, 0.8017035126686096, 0, 0, 0, 0, 0, 3.010277271270752, 2.9790256023406982, 2.931389331817627, 2.900527000427246, 2.870464324951172, 2.8460302352905273, 2.816157817840576, 2.775315999984741, 2.7634801864624023, 2.7325565814971924, 2.729945421218872, 2.7121689319610596, 2.6779863834381104, 2.6485931873321533, 2.654859781265259, 2.6297667026519775, 2.6041877269744873, 2.600289821624756, 2.5876994132995605, 2.5718753337860107, 2.575157403945923, 2.5369110107421875, 2.527487277984619, 2.542963743209839, 2.5003409385681152, 2.530441999435425, 2.5106141567230225, 2.5381298065185547, 2.490354061126709, 2.478184700012207, 2.5025384426116943, 2.481180191040039, 2.4870376586914062, 2.4736533164978027, 2.496537923812866]
    print(f"len of dist: {len(dist)}")
    angle_increment = 0.01749303564429283
    ang = np.arange(len(dist)) * angle_increment
    ox = np.sin(ang) * dist
    oy = np.cos(ang) * dist
    ox = ox[~np.isnan(ox)]
    ox = ox[~np.isinf(ox)]
    oy = oy[~np.isnan(oy)]
    oy = oy[~np.isinf(oy)]
    occupancy_map, min_x, max_x, min_y, max_y, xy_resolution = \
        generate_ray_casting_grid_map(ox, oy, xy_resolution, True)
    xy_res = np.array(occupancy_map).shape
    plt.figure(1, figsize=(10, 4))
    plt.subplot(122)
    plt.imshow(occupancy_map, cmap="PiYG_r")
    # cmap = "binary" "PiYG_r" "PiYG_r" "bone" "bone_r" "RdYlGn_r"
    plt.clim(-0.4, 1.4)
    plt.gca().set_xticks(np.arange(-.5, xy_res[1], 1), minor=True)
    plt.gca().set_yticks(np.arange(-.5, xy_res[0], 1), minor=True)
    plt.grid(True, which="minor", color="w", linewidth=0.6, alpha=0.5)
    plt.colorbar()
    plt.subplot(121)
    plt.plot([oy, np.zeros(np.size(oy))], [ox, np.zeros(np.size(oy))], "ro-")
    plt.axis("equal")
    plt.plot(0.0, 0.0, "ob")
    plt.gca().set_aspect("equal", "box")
    bottom, top = plt.ylim()  # return the current y-lim
    plt.ylim((top, bottom))  # rescale y axis, to match the grid orientation
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()