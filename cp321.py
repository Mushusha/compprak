import numpy as np
import matplotlib.pyplot as plt

def beam_equations(state, x, EJ, q, L):
    w, theta, M, V = state
    
    dw_dx = theta
    dtheta_dx = M / EJ
    dM_dx = V
    dV_dx = -q 
    
    return np.array([dw_dx, dtheta_dx, dM_dx, dV_dx])

def rk4_step(state, x, h, EJ, q, L):
    k1 = h * beam_equations(state, x, EJ, q, L)
    k2 = h * beam_equations(state + 0.5*k1, x + 0.5*h, EJ, q, L)
    k3 = h * beam_equations(state + 0.5*k2, x + 0.5*h, EJ, q, L)
    k4 = h * beam_equations(state + k3, x + h, EJ, q, L)
    
    return state + (k1 + 2*k2 + 2*k3 + k4) / 6

def integrate_beam(theta_guess, EJ, q, L, n_steps=100):
    initial_state = np.array([0.0, theta_guess, 0.0, q*L/2])
    
    x = np.linspace(0, L, n_steps)
    h = L / (n_steps - 1)
    
    trajectory = np.zeros((n_steps, 4))
    trajectory[0] = initial_state
    
    for i in range(1, n_steps):
        trajectory[i] = rk4_step(trajectory[i-1], x[i-1], h, EJ, q, L)
    
    return x, trajectory

def shooting_regula_falsi(EJ, q, L, max_iter=50):
    theta_left = np.radians(-5.0)
    theta_right = np.radians(5.0)
    
    x_left, traj_left = integrate_beam(theta_left, EJ, q, L)
    x_right, traj_right = integrate_beam(theta_right, EJ, q, L)
    
    residual_left = traj_left[-1, 0]
    residual_right = traj_right[-1, 0]
    
    for iteration in range(max_iter):
        if abs(residual_left) < 1e-8:
            return theta_left, traj_left
        if abs(residual_right) < 1e-8:
            return theta_right, traj_right
        
        theta_new = (residual_right * theta_left - residual_left * theta_right) / (residual_right - residual_left)
        
        x_new, traj_new = integrate_beam(theta_new, EJ, q, L)
        residual_new = traj_new[-1, 0]
        
        if residual_new * residual_left > 0:
            theta_left, residual_left = theta_new, residual_new
        else:
            theta_right, residual_right = theta_new, residual_new
        
        if abs(residual_new) < 1e-8:
            return theta_new, traj_new
    
    return theta_new, traj_new

def shooting_newton(EJ, q, L, theta_initial=np.radians(1.0), max_iter=50):
    theta = theta_initial
    h = 1e-6
    
    for iteration in range(max_iter):
        x, traj = integrate_beam(theta, EJ, q, L)
        residual = traj[-1, 0]  # w(L)
        
        if abs(residual) < 1e-8:
            return theta, traj
        
        x_p, traj_p = integrate_beam(theta + h, EJ, q, L)
        residual_p = traj_p[-1, 0]
        
        derivative = (residual_p - residual) / h
        
        theta = theta - residual / derivative
    
    return theta, traj

EJ = 1e5 
q = 1000 
L = 1.0 

theta_rf, traj_rf = shooting_regula_falsi(EJ, q, L)

theta_n, traj_n = shooting_newton(EJ, q, L)

x = np.linspace(0, L, 100)

plt.figure(figsize=(10, 6))

w_rf = traj_rf[:, 0] * 1000 
w_n = traj_n[:, 0] * 1000

plt.plot(x, w_rf, 'b-', linewidth=2, label='Regula falsi')
plt.plot(x, w_n, 'r--', linewidth=2, label='Ньютон')
plt.xlabel('x, м')
plt.ylabel('Прогиб w, мм')
plt.title('Форма изогнутой балки')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()