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
    c1, c2, c3, c4, k31, k41, k51 = 10**c1, 10**c2, 10**c3, 10**c4, 10**k31, 10**k41, 10**k51

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

    return cost / 10000

def gradient_descent(params, learning_rate, iterations, x, P980s):
    costs = []

    for epoch in tqdm(range(iterations)):
        grad = np.zeros_like(params)
        epsilon = 1e-2 # something to change
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            # param_tmp = params.copy()
            params_plus[i] += epsilon
            params_minus[i] -= epsilon
            # print('hi', cost_function(params_plus, x, P980s))
            # print('hi', cost_function(params_minus, x, P980s))
            # param_tmp[0] = 2.25551330653112
            # param_tmp[2] = 0.4984769347191149
            # true_param =  [2.25551330653112, -1.3056691656198742, 0.4984769347191149, 2.31942225639846, -0.631828119307082, 2.7201431777574223, -0.005981397391665376]
            # print('hi', cost_function(param_tmp, x, P980s))
            grad[i] = (cost_function(params_plus, x, P980s) -
                       cost_function(params_minus, x, P980s)) / (2 * epsilon)
        
        max_learning_rate = 5e-6
        if np.linalg.norm(grad) < 1e4:
            learning_rate = min(learning_rate * 1.5, max_learning_rate)  
        params -= learning_rate * grad
        print(epoch, grad, learning_rate * grad, params, learning_rate)

        current_cost = cost_function(params, x, P980s)
        costs.append(current_cost)
    
    return params, costs

def gradient_descent_adam(params, learning_rate, iterations, x, P980s):
    beta1 = 0.9  # Exponential decay rates for the moment estimates
    beta2 = 0.999
    epsilon = 1e-8  # Small constant to prevent division by zero

    m = np.zeros_like(params)  # Initialize 1st moment vector
    v = np.zeros_like(params)  # Initialize 2nd moment vector
    t = 0  # Initialize timestep

    costs = []

    for epoch in tqdm(range(iterations)):
        grad = np.zeros_like(params)
        epsilon = 1e-2  # Something to change
        # print('hello', v)
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += epsilon
            params_minus[i] -= epsilon
            grad[i] = (cost_function(params_plus, x, P980s) -
                       cost_function(params_minus, x, P980s)) / (2 * epsilon)

        t += 1
        # first moment estimate is estimate of gradient
        # second moment estimate is estimate of variance
        # Use moving average to update v and m
        # bias-correction because we initialize from 0
        # epsilon to prevent division by 0
        m = beta1 * m + (1 - beta1) * grad  # Update biased first moment estimate
        # print(v, beta2 * v, (1 - beta2), (grad ** 2))
        v = beta2 * v + (1 - beta2) * (grad ** 2)  # Update biased second moment estimate

        m_hat = m / (1 - beta1 ** t)  # Compute bias-corrected first moment estimate
        v_hat = v / (1 - beta2 ** t)  # Compute bias-corrected second moment estimate

        params -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)  # Update parameters

        print(epoch, grad, v_hat, params, learning_rate)
        current_cost = cost_function(params, x, P980s)
        costs.append(current_cost)

    return params, costs


conc = 15 ## Something to change
#  2.25551330653112, -1.3056691656198742, 0.4984769347191149, 2.31942225639846, -0.631828119307082, 2.7201431777574223, -0.005981397391665376
initial_params = [2, -1, 0, 2, 0, 2, 0] ## Something to change

learning_rate = 5e-7  ##5e-3## 5e-7 ## Something to change
iterations = 40 ## Something to change
P980s = power[conc]
optimized_params, costs = gradient_descent(initial_params, learning_rate, iterations, conc, P980s)

print(optimized_params)
print(costs)
output = {'optimized_params': list(optimized_params), 'costs': list(costs)}
json_data = json.dumps(output)
with open(f'./GD/results_{conc}.json', 'w') as f:
    f.write(json_data) # TypeError: Object of type ndarray is not JSON serializable

plt.plot(range(iterations), costs)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Cost by Epoch')
plt.show()

print("Optimized Parameters:", optimized_params)
