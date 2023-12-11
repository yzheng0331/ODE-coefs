import numpy as np
from scipy.optimize import minimize
from util_183 import system
from scipy.integrate import odeint

# Cost function to minimize
def cost_function(params, t, experimental_data_NIR, experimental_data_blue, x, P980):
    c1, c2, c3, c4, k31, k41, k51 = params

    state0 = [0, 37 * x, 0, 0, 0, 0]
    t_sim = np.arange(0.0, 0.001, 0.000001)
    state = odeint(system, state0, t_sim, args=(x, P980, c1, c2, c3, c4, k31, k41, k51))

    predictions_NIR = a31 * W3 * state[:, 3][-1]
    predictions_blue = a41 * W4 * state[:, 4][-1] + a52 * W5 * state[:, 5][-1]

    cost_NIR = np.sum((predictions_NIR - experimental_data_NIR)**2)
    cost_blue = np.sum((predictions_blue - experimental_data_blue)**2)

    total_cost = cost_NIR + cost_blue
    return total_cost

def gradient_cost(params, t, experimental_data_NIR, experimental_data_blue, x, P980):
    epsilon = 1e-6
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += epsilon
        params_minus[i] -= epsilon
        grad[i] = (cost_function(params_plus, t, experimental_data_NIR, experimental_data_blue, x, P980) -
                   cost_function(params_minus, t, experimental_data_NIR, experimental_data_blue, x, P980)) / (2 * epsilon)
    return grad


conc = 10 ## Something to change
initial_params = [] ## Something to change
with open(f'./synthetic_data/syn_{conc}.json', 'r') as f:
    loaded_json_data = f.read()
loaded_data = json.loads(loaded_json_data)
exp_blue_total = loaded_data.get('blue', [])
experimental_data_NIR = loaded_data.get('NIR', [])
experimental_data_blue = loaded_data.get('blue', [])


# Minimize the cost function using gradient descent
result = minimize(cost_function, initial_params, args=(t, experimental_data_NIR, experimental_data_blue, x, P980), method='BFGS', jac=gradient_cost)

# Extract optimized parameters
optimized_params = result.x
print("Optimized Parameters:", optimized_params)