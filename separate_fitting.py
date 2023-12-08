import json
from util_183 import system, compute_blue_for_x, compute_NIR_for_x 

p10_blue = [2.8*10**3, 4.4*10**3, 5*10**3, 6*10**3, 7.5*10**3, 9.1*10**3, 1.1*10**4, 1.4*10**4, 1.7*10**4, 2*10**4, 2.3*10**4, 2.7*10**4, 3*10**4]
p10_nir = [1.2*10**3, 1.8*10**3, 2.8*10**3, 4.4*10**3, 5*10**3, 6*10**3, 7.5*10**3, 9.1*10**3, 1.1*10**4, 1.4*10**4, 1.7*10**4, 2*10**4, 2.3*10**4, 2.7*10**4, 3*10**4]
p12_nir = [1.2*10**3, 1.8*10**3, 2.8*10**3, 4.4*10**3, 5*10**3, 6*10**3, 7.5*10**3, 9.1*10**3, 1.1*10**4, 1.4*10**4, 1.7*10**4, 2*10**4, 2.3*10**4, 2.7*10**4, 3*10**4]
p12_blue = [4.4*10**3, 5*10**3, 6*10**3, 7.5*10**3, 9.1*10**3, 1.1*10**4, 1.4*10**4, 1.7*10**4, 2*10**4, 2.3*10**4, 2.7*10**4, 3*10**4]
p15_nir = [1.2*10**3, 1.8*10**3, 2.8*10**3, 4.4*10**3, 5*10**3, 6*10**3, 7.5*10**3, 9.1*10**3, 1.1*10**4, 1.4*10**4, 1.7*10**4, 2*10**4, 2.3*10**4, 2.7*10**4, 3*10**4]
p15_blue = [4.4*10**3, 5*10**3, 6*10**3, 7.5*10**3, 9.1*10**3, 1.1*10**4, 1.4*10**4, 1.7*10**4, 2*10**4, 2.3*10**4, 2.7*10**4, 3*10**4]

power = {10: (p10_nir, p10_blue), 12: (p12_nir, p12_blue), 15: (p15_nir, p15_blue)}

def gen(conc, ref):
    global power
    syn_nir = []
    syn_blue = []
    p_nir, p_blue = power[conc]
    for p in p_nir:
        c1, c2, c3, c4, k31, k41, k51 = ref
        c1, c2, c3, c4, k31, k41, k51 = 10**c1, 10**c2, 10**c3, 10**c4, 10**k31, 10**k41, 10**k51
        nir = compute_NIR_for_x(conc, p, c1, c2, c3, c4, k31, k41, k51)
        syn_nir.append(nir)
    for p in p_blue:
        c1, c2, c3, c4, k31, k41, k51 = ref
        c1, c2, c3, c4, k31, k41, k51 = 10**c1, 10**c2, 10**c3, 10**c4, 10**k31, 10**k41, 10**k51
        blue = compute_blue_for_x(conc, p, c1, c2, c3, c4, k31, k41, k51)
        syn_blue.append(blue)
    syn = {'param': ref, 'NIR': syn_nir, 'blue': syn_blue}
    json_data = json.dumps(syn)
    with open(f'./synthetic_data/syn_{conc}.json', 'w') as f:
        f.write(json_data)
        
def objective_separate(conc, exponents):
    c1_exp, c2_exp, c3_exp, c4_exp, k31_exp, k41_exp, k51_exp = exponents
    c1, c2, c3, c4, k31, k41, k51 = (10**c1_exp, 10**c2_exp, 10**c3_exp,
                                 10**c4_exp, 10**k31_exp, 10**k41_exp, 10**k51_exp)
    total_error = 0
    with open(f'./synthetic_data/syn_{conc}.json', 'r') as f:
        loaded_json_data = f.read()
    loaded_data = json.loads(loaded_json_data)
    exp_blue_total = loaded_data.get('blue', [])
    exp_NIR = loaded_data.get('NIR', [])
    p_nir, p_blue = power[conc]
    for index, p in enumerate(p_blue):
        blue = compute_blue_for_x(10, p, c1, c2, c3, c4, k31, k41, k51)
        error_blue = ((blue - exp_blue_total[index])/exp_blue_total[index])**2
        total_error += error_blue

    for index, p in enumerate(p_nir):
        nir = compute_NIR_for_x(10, p, c1, c2, c3, c4, k31, k41, k51)
        error_NIR = ((nir - exp_NIR[index])/exp_NIR[index])**2
        total_error += error_NIR

    return total_error,

def obj_separate(conc):
    def f(exponents):
        return objective_separate(conc, exponents)
    return f
