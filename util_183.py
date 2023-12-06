import numpy as np
from scipy.integrate import odeint

a21 = 1
a31 = 0.27
a32 = 0.73

a41 = 0.18
a42 = 0.24
a43 = 0.58

a51 = 0.24
a52 = 0.23
a53 = 0.2
a54 = 0.33

W2 = 63000
W3 = 20000
W4 = 15000
W5 = 33000
Ws2 = 8000

def system(state, t, x, P980, c1, c2, c3, c4, k31, k41, k51):

    ns2, n1, n2, n3, n4, n5 = state

    #ns2
    ms2 = 1.23*P980*(37*(100-x)-ns2) - Ws2*ns2 - (c1*n1+c2*n2+c3*n3+c4*n4)*ns2  # ns1 = total_Yb - ns2

    # n1
    m1 = -c1*n1*ns2 + a21*W2*n2 + a31*W3*n3 + a41*W4*n4 + a51*W5*n5 - k41*n1*n4 - k31*n1*n3 - k51*n5*n1

    # n2
    m2 = c1*n1*ns2 - c2*n2*ns2 - a21*W2*n2 + a32*W3*n3 + a42*W4*n4 + a52*W5*n5 + k41*n1*n4 + 2*k31*n1*n3

    # n3
    m3 = c2*n2*ns2 - c3*n3*ns2 - (a31+a32)*W3*n3 + a43*W4*n4 + a53*W5*n5 + 2*k51*n1*n5 + k41*n1*n4 - k31*n1*n3

    # n4
    m4 = c3*n3*ns2 - c4*n4*ns2 - (a43+a42+a41)*W4*n4 + a54*W5*n5 - k41*n1*n4

    # n5
    m5 = c4*n4*ns2 - (a54+a53+a52+a51)*W5*n5 - k51*n1*n5


    return [ms2, m1, m2, m3, m4, m5]

def compute_NIR_for_x(x, P980, c1, c2, c3, c4, k31, k41, k51):

    # Compute initial condition
    y = 37 * x
    state0 = [0, y, 0, 0, 0, 0]
    t = np.arange(0.0, 0.001, 0.000001)
    state = odeint(system, state0, t, args=(x, P980, c1, c2, c3, c4, k31, k41, k51))
    NIR = a31 * W3 * state[:, 3][-1]

    return NIR

def compute_blue_for_x(x, P980, c1, c2, c3, c4, k31, k41, k51):

    # Compute initial condition
    y = 37 * x
    state0 = [0, y, 0, 0, 0, 0]
    t = np.arange(0.0, 0.001, 0.000001)
    state = odeint(system, state0, t, args=(x, P980, c1, c2, c3, c4, k31, k41, k51))
    blue_total = a41 * W4 * state[:, 4][-1] + a52 * W5 * state[:, 5][-1]

    return blue_total
