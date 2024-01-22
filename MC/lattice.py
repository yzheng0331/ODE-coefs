import numpy as np
import math

def new_lattice(yb_conc, tm_conc, d):
    # Create the lattice
    l_r = int(d/2/0.596)
    l_z = int(d/2/0.353)
    print(l_r, l_z)
    na_points = [(i, j, k) for i in range(-l_r, l_r+1) for j in range(-l_r, l_r+1) for k in range(-l_z, l_z+1)]
    y_coords = [(i+1/2, j+1/2, k+1/2) for i,j,k in na_points]
    na_points = in_diameter(d, na_points)
    y_coords = in_diameter(d, y_coords)
    n_points = len(y_coords)

    # Assign molecules to the lattice points
    n_yb = int(yb_conc*n_points)
    n_tm = int(tm_conc*n_points)
    types =  ['Yb'] * n_yb + ['Tm'] * n_tm + ['Y'] * (n_points - n_yb - n_tm)
    np.random.shuffle(types)
    values = []
    for t in types:
        if t == 'Yb':
            values.append(np.random.choice([0, 1], p=[0.85, 0.15]))  
            ### here, because of absorbation rate of Yb, set the rate manually as 0.85 and 0.15
        elif t == 'Tm':
            values.append(2)  # All type B points have value 2
        else:
            values.append(-1)  # No associated value for type C
    y_points = [[t, tuple(coord), v] for t, coord, v in zip(types, y_coords, values)]

    return na_points, y_points, n_points

def l_distance(p1, p2):
   vec = (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])
   evec = to_euclidean(vec)
   return e_distance(evec)

def to_euclidean(vec):
   a, b, c = vec
   return (0.596*a+0.5*0.596*b, 3**(1/2)/2*0.596*b, 0.353*c)

def e_distance(vec):
   return (vec[0]**2+vec[1]**2+vec[2]**2)**(1/2)

def in_diameter(d, points):
    origin = (0,0,0)
    ret = []
    for point in points:
        if l_distance(point, origin) < d/2:
            ret.append(point)
    return ret

print(new_lattice(0.5,0.5,1))