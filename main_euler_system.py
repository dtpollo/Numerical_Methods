from euler_system import euler_system
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory if it doesn't exist
os.makedirs("euler_lorenz", exist_ok=True)

error_log_path = "euler_lorenz/euler_graficos_null.txt"


# Define the system of ODEs
def f(t, functions):
    x, y, z = functions
    sigma = 10
    rho = 28
    beta = 8 / 3
    return np.array([sigma * (y - x),
                     x * (rho - z) - y,
                     x * y - beta * z])

# Define the exact solution
"""
def exact_solution(t):
    y1_exact=np.cos(t)
    y2_exact=np.sin(t)
    return y1_exact, y2_exact
"""


# =======+>  General Parameters  <+======
T = 50 # Time Horizon

# Initial conditions
y0 = np.array([[1,1,1], [1,1,1+10e-6], [1,1,1+10e-8]])

# Step size
h = [10e-2, 5e-3, 2e-3, 10e-3]

# =============+>  End  <+==============



def main_euler_system(f, y0, T, h, step, initial_condition):
    # Resolve the system using Euler's method
    t_vals, y_vals = euler_system(f, 0, y0, T, h)
    colors = [
        '#e41a1c',  # 0
        '#377eb8',  # 1
        '#4daf4a',  # 2
        '#984ea3',  # 3
        '#ff7f00',  # 4
        '#ffff33'   # 5
        ]

    # Plotting
    plt.figure(figsize=(10, 6))
    if not np.any(np.isnan(y_vals[:, 0])) and not np.any(np.isinf(y_vals[:, 0])):
        plt.plot(t_vals, y_vals[:, 0], label='X (Euler)', linestyle='-', color=colors[2], linewidth=1)

    if not np.any(np.isnan(y_vals[:, 1])) and not np.any(np.isinf(y_vals[:, 1])):
        plt.plot(t_vals, y_vals[:, 1], label='Y (Euler)', linestyle='-', color=colors[3], linewidth=1)

    if not np.any(np.isnan(y_vals[:, 2])) and not np.any(np.isinf(y_vals[:, 2])):   
        plt.plot(t_vals, y_vals[:, 2], label='Z (Euler)', linestyle='-', color=colors[4], linewidth=1)

    # Exact solutions
    #y1_exact,y2_exact=exact_solution(t_vals)
    #plt.plot(t_vals, y1_exact, label='y1 exact')
    #plt.plot(t_vals, y2_exact, label='y2 exact')


    plt.xlabel('t')
    plt.ylabel('Functions of the system')
    plt.legend()
    plt.title('Euler Method for Lorenz System')
    filename = f"lorenz_h{step:.4f}_init{initial_condition[2]:.8f}".replace('.', 'p')

    if np.any(np.isnan(y_vals)) or np.any(np.isinf(y_vals)):
        with open(error_log_path, "a") as log_file:
            log_file.write(f"{filename}.png - contains NaN or inf\n")
        return



    # Save valid images
    plt.savefig(f"euler_lorenz/{filename}.png")
    plt.close()


for initial_condition in y0:
    for step in h:
        main_euler_system(f, initial_condition, T, step, step, initial_condition)
