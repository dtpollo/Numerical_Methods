from runge_kutta_45_system import runge_kutta_45_system
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory if it doesn't exist
os.makedirs("rkf45_lorenz", exist_ok=True)
error_log_path = "rkf45_lorenz/rkf45_graficos_null.txt"

# Define the system of ODEs
def f(t, functions):
    x, y, z = functions
    sigma = 10
    rho = 28
    beta = 8 / 3
    return np.array([sigma * (y - x),
                     x * (rho - z) - y,
                     x * y - beta * z])

# =======+>  General Parameters  <+======
T = 50 # Time Horizon

# Initial conditions
y0 = np.array([[1,1,1], [1,1,1+10e-6], [1,1,1+10e-8]])

# Step size
h = [10e-2, 5e-3, 2e-3, 10e-3]

# Tolerance
tol = [10e-3, 10e-5, 10e-10]

# =============+>  End  <+==============



def main_rkf45_system(f, y0, T, h, step, initial_condition, tol):
    # Resolve the system using Runge-Kutta Fehlberg method
    t_vals, y_vals = runge_kutta_45_system(f, 0, y0, T, tol, h)
    y_vals = np.array(y_vals)
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
    plt.plot(t_vals, y_vals[:, 0], linestyle='-', label='X (RKF45)', color=colors[0], linewidth=1)
    plt.plot(t_vals, y_vals[:, 1], linestyle='-', label='Y (RKF45)', color=colors[1], linewidth=1)
    plt.plot(t_vals, y_vals[:, 2], linestyle='-', label='Z (RKF45)', color=colors[2], linewidth=1)


    plt.xlabel("t")
    plt.ylabel('Functions of the system')
    plt.legend()
    plt.title("Runge-Kutta Fehlberg Method for Lorenz System")
    filename = f"rkf45_h{step:.4f}_init{initial_condition[2]:.8f}_tol{tol:.8f}".replace('.', 'p')

    # Verificar si hay valores inválidos
    if np.any(np.isnan(y_vals)) or np.any(np.isinf(y_vals)):
        with open(error_log_path, "a") as log_file:
            log_file.write(f"{filename}.png - contiene NaN o inf\n")
        return
    
    # Save valid images
    plt.savefig(f"rkf45_lorenz/{filename}.png")
    plt.close()



# Ejecutar todas las combinaciones
for initial_condition in y0:
    for step in h:
        for tolerance in tol:
            main_rkf45_system(f, initial_condition, T, step, step, initial_condition, tolerance)