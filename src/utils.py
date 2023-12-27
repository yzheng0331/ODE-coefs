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

P980 = 3*10**4  # our power density

# i = 0
# prev_i = 0
# j = 0

def system(state, t, x, c1, c2, c3, c4, k31, k41, k51):
    
    # global i
    # i += 1
    # print(i)

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

    # print([ms2, m1, m2, m3, m4, m5])
    # raise Exception
    return [ms2, m1, m2, m3, m4, m5]


Cr = 8 / 100  # 8%


def calculate_up(up, C2):
    C2 = C2 / 100
    return 3 * up / (2 + (Cr / C2) ** 2)


def calculate_k(k, C2):
    C2 = C2 / 100
    return k * (C2 / Cr) ** 2


def compute_values_for_x(x, c1, c2, c3, c4, k31, k41, k51):

    # Compute initial condition
    y = 37 * x
    state0 = [0, y, 0, 0, 0, 0]

    # ODEs
    t = np.arange(0.0, 0.001, 0.000001)

    state = odeint(system, state0, t, args=(x, c1, c2, c3, c4, k31, k41, k51))
    
    # global j
    # global prev_i
    # j += 1
    # prev_i = i
    # print(j, ' ', i) 


    # Compute NIR and blue_total
    NIR = a31 * W3 * state[:, 3][-1]
    blue_1 = a41 * W4 * state[:, 4][-1]
    blue_2 = a52 * W5 * state[:, 5][-1]
    blue_total = blue_1 + blue_2

    return NIR, blue_total

def objective(exponents):

    # c1, c2, c3, c4, k31, k41, k51 = params
    c1_exp, c2_exp, c3_exp, c4_exp, k31_exp, k41_exp, k51_exp = exponents

    # Converting exponent values to actual values
    c1, c2, c3, c4, k31, k41, k51 = (10**c1_exp, 10**c2_exp, 10**c3_exp,
                                 10**c4_exp, 10**k31_exp, 10**k41_exp, 10**k51_exp)

    up_values = [c1, c2, c3, c4]

    k_values = [k31, k41, k51]


    C2_range = [4, 6, 8, 10, 12, 15, 50]

    # storing new cross relaxation and new up-conversion values
    nested_up_list = []
    nested_k_list = []

    for C2 in C2_range:

        up_sublist = [calculate_up(up, C2) for up in up_values]

        k_sublist = [calculate_k(k, C2) for k in k_values]

        nested_up_list.append(up_sublist)

        nested_k_list.append(k_sublist)

    x_values = [4, 6, 8, 10, 12, 15, 50]

    exp_blue_total = [4.5*10**3, 5.7*10**3, 5.5*10**3, 3.5*10**3, 2*10**3, 1.2*10**3, 0.6*10**3]
    blue_std=[1*10**3, 1.5*10**3, 1*10**3, 1*10**3, 0.5*10**3, 0.5*10**3, 0.25*10**3]

    #blue_weight=[i/j for i, j in zip(exp_blue_total, blue_std)]
    #blue_weight_2=[x/sum(blue_weight) for x in blue_weight]

    exp_NIR = [5*10**3, 8.5*10**3, 6.6*10**3, 7.5*10**3, 6*10**3, 5.5*10**3, 0.6*10**3]
    NIR_std=[1.1*10**3, 2.25*10**3, 0.7*10**3, 1.5*10**3, 1*10**3, 1.7*10**3, 0.25*10**3]

    #NIR_weight=[i/j for i, j in zip(exp_NIR, NIR_std)]
    #NIR_weight_2=[x/sum(NIR_weight) for x in NIR_weight]


    total_error = 0

    for index, x in enumerate(x_values):

        c1, c2, c3, c4 = nested_up_list[index]

        k31, k41, k51 = nested_k_list[index]

        NIR, blue_total = compute_values_for_x(x, c1, c2, c3, c4, k31, k41, k51)

        error_NIR = ((NIR - exp_NIR[index])/exp_NIR[index])**2
        error_blue = ((blue_total - exp_blue_total[index])/exp_blue_total[index])**2

        total_error += error_NIR + error_blue


    return total_error, # a return statement with a trailing comma, it means the function is returning a tuple with a single element.

