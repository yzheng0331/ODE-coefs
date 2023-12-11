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

def gradient_descent(params, learning_rate, iterations, t, experimental_data_NIR, experimental_data_blue, x, P980):
    costs = []

    for epoch in tqdm(range(iterations)):
        grad = np.zeros_like(params)
        epsilon = 1e-6
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += epsilon
            params_minus[i] -= epsilon
            grad[i] = (cost_function(params_plus, t, experimental_data_NIR, experimental_data_blue, x, P980) -
                       cost_function(params_minus, t, experimental_data_NIR, experimental_data_blue, x, P980)) / (2 * epsilon)

        params -= learning_rate * grad

        current_cost = cost_function(params, t, experimental_data_NIR, experimental_data_blue, x, P980)
        costs.append(current_cost)

    return params, costs

def gradient_descent_multiple_experiments(experiments, learning_rate, iterations):
    all_optimized_params = []
    all_costs = []

    for experiment in experiments:
        t, experimental_data_NIR, experimental_data_blue, x, P980 = experiment

        # Initial parameters (replace with your initial guesses)
        initial_params = [c1_initial, c2_initial, c3_initial, c4_initial, k31_initial, k41_initial, k51_initial]

        # Run gradient descent
        optimized_params, costs = gradient_descent(initial_params, learning_rate, iterations, t, experimental_data_NIR, experimental_data_blue, x, P980)

        # Store results for this experiment
        all_optimized_params.append(optimized_params)
        all_costs.append(costs)

    return all_optimized_params, all_costs



conc = 10 ## Something to change
initial_params = [] ## Something to change
with open(f'./synthetic_data/syn_{conc}.json', 'r') as f:
    loaded_json_data = f.read()
loaded_data = json.loads(loaded_json_data)
exp_blue_total = loaded_data.get('blue', [])
experimental_data_NIR = loaded_data.get('NIR', [])
experimental_data_blue = loaded_data.get('blue', [])


learning_rate = 0.01 ## Something to change
iterations = 100 ## Something to change

optimized_params, costs = gradient_descent(initial_params, learning_rate, iterations, t, experimental_data_NIR, experimental_data_blue, conc, P980)


#########################################################
# Example experiments (replace with your actual data)
experiment_1 = (t_1, experimental_data_NIR_1, experimental_data_blue_1, x_1, P980_1)
experiment_2 = (t_2, experimental_data_NIR_2, experimental_data_blue_2, x_2, P980_2)
# Add more experiments as needed

experiments = [experiment_1, experiment_2]

# Hyperparameters
learning_rate = 0.01
iterations = 100

# Run gradient descent for multiple experiments
all_optimized_params, all_costs = gradient_descent_multiple_experiments(experiments, learning_rate, iterations)
#########################################################

plt.plot(range(iterations), costs)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Cost by Epoch')
plt.show()

print("Optimized Parameters:", optimized_params)
