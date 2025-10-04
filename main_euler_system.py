from euler_system import euler_system
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory if it doesn't exist
os.makedirs("output_lorenz", exist_ok=True)

error_log_path = "errores_guardado.txt"


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
"""def exact_solution(t):
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

    # Plotting
    plt.plot(t_vals, y_vals[:, 0], label='X (Euler)', linestyle='--')
    plt.plot(t_vals, y_vals[:, 1], label='Y (Euler)', linestyle='--')
    plt.plot(t_vals, y_vals[:, 2], label='Z (Euler)', linestyle='--')

    # Exact solutions
    #y1_exact,y2_exact=exact_solution(t_vals)


    #plt.plot(t_vals, y1_exact, label='y1 exact')
    #plt.plot(t_vals, y2_exact, label='y2 exact')


    plt.xlabel('t')
    plt.ylabel('Variables del sistema')
    plt.legend()
    plt.title('Método de Euler para el sistema de Lorenz')
    filename = f"lorenz_h{step:.4f}_init{initial_condition[2]:.8f}".replace('.', 'p')
    # Verifica si hay valores inválidos
    if np.any(np.isnan(y_vals)) or np.any(np.isinf(y_vals)):
        with open(error_log_path, "a") as log_file:
            log_file.write(f"{filename}.png - contiene NaN o inf\n")
        return  # Evita graficar

    # Guardar imagen si los datos son válidos
    plt.savefig(f"output_lorenz/{filename}.png")
    plt.close()


for initial_condition in y0:
    for step in h:
        main_euler_system(f, initial_condition, T, step, step, initial_condition)
