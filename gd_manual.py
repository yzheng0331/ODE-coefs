import numpy as np
import json
from tqdm import tqdm
from scipy.optimize import minimize
from util_183 import system, compute_blue_for_x, compute_NIR_for_x
from separate_fitting import power
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Cost function to minimize
def cost_function(params, x, P980s):
    c1, c2, c3, c4, k31, k41, k51 = params

    P980_NIR, P980_blue = P980s
    cost = 0
    with open(f'./synthetic_data/syn_{x}.json', 'r') as f:
        loaded_json_data = f.read()
    loaded_data = json.loads(loaded_json_data)
    experimental_data_NIR = loaded_data.get('NIR', [])
    experimental_data_blue = loaded_data.get('blue', [])
    for P980, exp_nir in zip(P980_NIR, experimental_data_NIR):
        nir = compute_NIR_for_x(x, P980, c1, c2, c3, c4, k31, k41, k51)
        cost_NIR = (nir - exp_nir)**2
        cost += cost_NIR
    for P980, exp_blue in zip(P980_blue, experimental_data_blue):
        blue = compute_blue_for_x(x, P980, c1, c2, c3, c4, k31, k41, k51)
        cost_blue = (blue - exp_blue)**2
        cost += cost_blue    

    return cost

def gradient_descent(params, learning_rate, iterations, x, P980s):
    costs = []

    for epoch in tqdm(range(iterations)):
        grad = np.zeros_like(params)
        epsilon = 1e-3 # something to change
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += epsilon
            params_minus[i] -= epsilon
            grad[i] = (cost_function(params_plus, x, P980s) -
                       cost_function(params_minus, x, P980s)) / (2 * epsilon)

        params -= learning_rate * grad

        costs.append(cost_function)
    
    return params, costs



conc = 10 ## Something to change
#  2.25551330653112, -1.3056691656198742, 0.4984769347191149, 2.31942225639846, -0.631828119307082, 2.7201431777574223, -0.005981397391665376
initial_params = [3, -1, 0, 3, 0, 2, 0] ## Something to change

learning_rate = 0.01 ## Something to change
iterations = 100 ## Something to change
P980s = power[conc]
optimized_params, costs = gradient_descent(initial_params, learning_rate, iterations, conc, P980s)

plt.plot(range(iterations), costs)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Cost by Epoch')
plt.show()

print("Optimized Parameters:", optimized_params)
