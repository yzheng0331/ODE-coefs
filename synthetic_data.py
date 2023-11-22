import numpy as np
import pandas as pd
from utils import compute_values_for_x, calculate_up, calculate_k

x_values = [4, 6, 8, 10, 12, 15, 50]
def generate(ref, n = 1):
    print("enters here")
    df_list_param = []
    df_list_emi = []
    for _ in range(n):
        syn_param = []
        for i in ref:
            sample = np.random.normal(i, 0.20)
            syn_param.append(sample)
        syn_param = [syn_param]
        df_param = pd.DataFrame(syn_param, columns = ['c1', 'c2', 'c3', 'c4', 'k31', 'k41', 'k51'])
        df_list_param.append(df_param) 

        syn_nir = []
        syn_blue = []
        for x in x_values:
            c1, c2, c3, c4, k31, k41, k51 = syn_param[0]
            c1, c2, c3, c4, k31, k41, k51 = 10**c1, 10**c2, 10**c3, 10**c4, 10**k31, 10**k41, 10**k51
            c1 = calculate_up(c1, x)
            c2 = calculate_up(c2, x)
            c3 = calculate_up(c3, x)
            c4 = calculate_up(c4, x)
            k31 = calculate_k(k31, x)
            k41 = calculate_k(k41, x)
            k51 = calculate_k(k51, x)
            NIR, blue = compute_values_for_x(x, c1, c2, c3, c4, k31, k41, k51)
            print(NIR, blue)
            NIR = np.random.normal(NIR, 100)
            syn_nir.append(NIR)
            blue = np.random.normal(blue, 1)
            syn_blue.append(blue)
        syn = [syn_nir, syn_blue]
        df_syn = pd.DataFrame(syn, columns = x_values)
        df_list_emi.append(df_syn)
    final_df = pd.concat(df_list_param, axis=0, ignore_index=True)
    final_df.to_csv("synthetic_param.csv", index=False)

    final_df = pd.concat(df_list_emi, axis=0, ignore_index=True)
    final_df.to_csv("synthetic_data.csv", index=False)


def objective_syn(exponents):

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

    df_syn_data = pd.read_csv("synthetic_data.csv")
    
    df_syn_data = pd.read_csv("synthetic_data.csv")
    

    exp_blue_total = list(df_syn_data.iloc[1])
    # [4.5*10**3, 5.7*10**3, 5.5*10**3, 3.5*10**3, 2*10**3, 1.2*10**3, 0.6*10**3]
    # blue_std=[1*10**3, 1.5*10**3, 1*10**3, 1*10**3, 0.5*10**3, 0.5*10**3, 0.25*10**3]

    #blue_weight=[i/j for i, j in zip(exp_blue_total, blue_std)]
    #blue_weight_2=[x/sum(blue_weight) for x in blue_weight]

    exp_NIR = list(df_syn_data.iloc[0])
    # [5*10**3, 8.5*10**3, 6.6*10**3, 7.5*10**3, 6*10**3, 5.5*10**3, 0.6*10**3]
    # NIR_std=[1.1*10**3, 2.25*10**3, 0.7*10**3, 1.5*10**3, 1*10**3, 1.7*10**3, 0.25*10**3]

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



reference = [3.5, -0.72, 0.64, 4.9, 4.2, 6.6, -3.9]
# [2.6576577925413774,
#   -1.399210616984687,
#   0.5546134786066016,
#   3.8070708908234026,
#   2.0285724283998414,
#   -4.12526984276854,
#   -3.4756328446599003]
#[3.5, -0.72, 0.64, 4.9, 4.2, 6.6, -3.9]
# generate(reference)