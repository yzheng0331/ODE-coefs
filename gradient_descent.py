import numpy as np
import json
from scipy.integrate import odeint
from util_183 import system, compute_blue_for_x, compute_NIR_for_x, power

def cost_function(exponents, conc):
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

def gradient_descent_NIR_and_blue(x, P980, c1, c2, c3, c4, k31, k41, k51, learning_rate, iterations):
    for _ in range(iterations):
        grad = np.zeros(7)
        epsilon = 0.5
        for i in range(7):
            params_plus = np.array([c1, c2, c3, c4, k31, k41, k51])
            params_minus = np.array([c1, c2, c3, c4, k31, k41, k51])
            params_plus[i] += epsilon
            params_minus[i] -= epsilon
            grad[i] = (compute_NIR_for_x(x, P980, *params_plus) - compute_NIR_for_x(x, P980, *params_minus)) / (2 * epsilon)

        # Update parameters for NIR
        c1 -= learning_rate * grad[0]
        c2 -= learning_rate * grad[1]
        c3 -= learning_rate * grad[2]
        c4 -= learning_rate * grad[3]
        k31 -= learning_rate * grad[4]
        k41 -= learning_rate * grad[5]
        k51 -= learning_rate * grad[6]

        # Compute gradients for Blue
        grad_blue = np.zeros(7)
        for i in range(7):
            params_plus = np.array([c1, c2, c3, c4, k31, k41, k51])
            params_minus = np.array([c1, c2, c3, c4, k31, k41, k51])
            params_plus[i] += epsilon
            params_minus[i] -= epsilon
            grad_blue[i] = (compute_blue_for_x(x, P980, *params_plus) - compute_blue_for_x(x, P980, *params_minus)) / (2 * epsilon)

        # Update parameters for Blue
        c1 -= learning_rate * grad_blue[0]
        c2 -= learning_rate * grad_blue[1]
        c3 -= learning_rate * grad_blue[2]
        c4 -= learning_rate * grad_blue[3]
        k31 -= learning_rate * grad_blue[4]
        k41 -= learning_rate * grad_blue[5]
        k51 -= learning_rate * grad_blue[6]

    return c1, c2, c3, c4, k31, k41, k51




# Example data
t = np.linspace(0, 1, 100)
observed_data = np.dot(np.random.rand(100, 7), np.power(t, np.arange(1, 8)))  # Adjust based on your actual data

# Initialize coefficients (thetas) and perform gradient descent
thetas_initial = np.random.rand(7)
learning_rate = 0.1
iterations = 100

thetas_optimized = gradient_descent(thetas_initial, t, observed_data, learning_rate, iterations)

print("Optimized Coefficients (thetas):", thetas_optimized)