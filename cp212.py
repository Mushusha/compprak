import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from scipy.integrate import solve_ivp

L1 = 1
L2 = 2
g = 9.81

phi1_0_base = np.radians(60.0)
phi2_0_variations = np.radians(np.linspace(0, 180, 15))
psi1_0 = 0.0
psi2_0 = 0.0

time = 25.0
nsteps = 300
dt = time / nsteps
t = np.linspace(0, time, nsteps + 1)

all_phi1 = []
all_phi2 = []
all_trajectories = []

def equations_of_motion(phi1, phi2, psi1, psi2):
    psi1_dot = -2.0 * g * phi1 / L1 + g * phi2 / L1
    psi2_dot = -2.0 * g * phi2 / L2 + 2.0 * g * phi1 / L2
    return psi1, psi2, psi1_dot, psi2_dot

def rk4_step(phi1, phi2, psi1, psi2, dt):
    k1_phi1, k1_phi2, k1_psi1, k1_psi2 = equations_of_motion(phi1, phi2, psi1, psi2)
    
    phi1_temp = phi1 + k1_phi1 * dt / 2
    phi2_temp = phi2 + k1_phi2 * dt / 2
    psi1_temp = psi1 + k1_psi1 * dt / 2
    psi2_temp = psi2 + k1_psi2 * dt / 2
    k2_phi1, k2_phi2, k2_psi1, k2_psi2 = equations_of_motion(phi1_temp, phi2_temp, psi1_temp, psi2_temp)
    
    phi1_temp = phi1 + k2_phi1 * dt / 2
    phi2_temp = phi2 + k2_phi2 * dt / 2
    psi1_temp = psi1 + k2_psi1 * dt / 2
    psi2_temp = psi2 + k2_psi2 * dt / 2
    k3_phi1, k3_phi2, k3_psi1, k3_psi2 = equations_of_motion(phi1_temp, phi2_temp, psi1_temp, psi2_temp)
    
    phi1_temp = phi1 + k3_phi1 * dt
    phi2_temp = phi2 + k3_phi2 * dt
    psi1_temp = psi1 + k3_psi1 * dt
    psi2_temp = psi2 + k3_psi2 * dt
    k4_phi1, k4_phi2, k4_psi1, k4_psi2 = equations_of_motion(phi1_temp, phi2_temp, psi1_temp, psi2_temp)
    
    phi1_new = phi1 + (k1_phi1 + 2*k2_phi1 + 2*k3_phi1 + k4_phi1) * dt / 6
    phi2_new = phi2 + (k1_phi2 + 2*k2_phi2 + 2*k3_phi2 + k4_phi2) * dt / 6
    psi1_new = psi1 + (k1_psi1 + 2*k2_psi1 + 2*k3_psi1 + k4_psi1) * dt / 6
    psi2_new = psi2 + (k1_psi2 + 2*k2_psi2 + 2*k3_psi2 + k4_psi2) * dt / 6
    
    return phi1_new, phi2_new, psi1_new, psi2_new

def get_cartesian_coordinates(phi1, phi2):
    x1 = L1 * np.sin(phi1)
    y1 = -L1 * np.cos(phi1)
    x2 = x1 + L2 * np.sin(phi2)
    y2 = y1 - L2 * np.cos(phi2)
    return x1, y1, x2, y2

for phi2_0 in phi2_0_variations:
    phi1_arr = np.zeros(nsteps + 1)
    phi2_arr = np.zeros(nsteps + 1)
    trajectories = np.zeros((nsteps + 1, 4))
    
    phi1_arr[0], phi2_arr[0] = phi1_0_base, phi2_0
    trajectories[0] = get_cartesian_coordinates(phi1_0_base, phi2_0)
    
    phi1, phi2, psi1, psi2 = phi1_0_base, phi2_0, psi1_0, psi2_0
    
    for i in range(nsteps):
        phi1, phi2, psi1, psi2 = rk4_step(phi1, phi2, psi1, psi2, dt)
        phi1_arr[i + 1] = phi1
        phi2_arr[i + 1] = phi2
        trajectories[i + 1] = get_cartesian_coordinates(phi1, phi2)
    
    all_phi1.append(phi1_arr)
    all_phi2.append(phi2_arr)
    all_trajectories.append(trajectories)

fig = plt.figure(figsize=(12, 10))
ax = plt.subplot(111)
ax.set_xlim(-(L1 + L2) * 1.3, (L1 + L2) * 1.3)
ax.set_ylim(-(L1 + L2) * 1.3, (L1 + L2) * 1.3)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_title('Двойные маятники с разными начальными условиями', fontsize=14)
ax.axhline(y=0, color='gray', linestyle='-', linewidth=2, alpha=0.5)
ax.plot(0, 0, 'ko', markersize=8)

colors = plt.cm.plasma(np.linspace(0, 1, len(phi2_0_variations)))
lines = []
trails = []
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

legend_text = ax.text(0.02, 0.02, '', transform=ax.transAxes, fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

for i, color in enumerate(colors):
    line, = ax.plot([], [], 'o-', lw=1.5, markersize=4, color=color, alpha=0.8)
    lines.append(line)
    
    trail, = ax.plot([], [], '-', lw=1, color=color, alpha=0.6)
    trails.append(trail)

trails_data = [[] for _ in range(len(phi2_0_variations))]

def init_anim():
    for line, trail in zip(lines, trails):
        line.set_data([], [])
        trail.set_data([], [])
    time_text.set_text('')
    
    legend_info = "Начальные углы φ:\n"
    for i in range(0, len(phi2_0_variations), 3):
        angles = [f"{np.degrees(phi2_0_variations[j]):.0f}°" 
                 for j in range(i, min(i+3, len(phi2_0_variations)))]
        legend_info += "  " + ", ".join(angles) + "\n"
    legend_text.set_text(legend_info)
    
    return lines + trails + [time_text, legend_text]

def animate(i):
    for idx, (phi1_arr, phi2_arr, trajectories) in enumerate(zip(all_phi1, all_phi2, all_trajectories)):
        x1, y1, x2, y2 = trajectories[i]
        
        lines[idx].set_data([0, x1, x2], [0, y1, y2])
        
        if len(trails_data[idx]) < 80:
            trails_data[idx].append((x2, y2))
        else:
            trails_data[idx].pop(0)
            trails_data[idx].append((x2, y2))
        
        if trails_data[idx]:
            trail_x, trail_y = zip(*trails_data[idx])
            trails[idx].set_data(trail_x, trail_y)
        
    return lines + trails + [legend_text]

ani = anim.FuncAnimation(fig, animate, frames=nsteps, init_func=init_anim,
                        interval=dt*800, blit=True, repeat=True, repeat_delay=2000)

plt.tight_layout()
plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

for i, (phi1_arr, phi2_arr) in enumerate(zip(all_phi1, all_phi2)):
    angle_deg = np.degrees(phi2_0_variations[i])
    ax1.plot(t, np.degrees(phi1_arr), color=colors[i], alpha=0.6, linewidth=1)
    ax2.plot(t, np.degrees(phi2_arr), color=colors[i], alpha=0.6, linewidth=1)

ax1.set_xlabel('Время (с)')
ax1.set_ylabel('Угол φ₁ (градусы)')
ax1.set_title('Углы первого маятника для разных начальных условий')
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Время (с)')
ax2.set_ylabel('Угол φ₂ (градусы)')
ax2.set_title('Углы второго маятника для разных начальных условий')
ax2.grid(True, alpha=0.3)

sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, 
                          norm=plt.Normalize(0, 180))
sm.set_array([])

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(12, 6))

reference_idx = 0
for i in range(1, len(all_phi2)):
    divergence = np.sqrt((all_trajectories[i][:, 2] - all_trajectories[reference_idx][:, 2])**2 + 
                        (all_trajectories[i][:, 3] - all_trajectories[reference_idx][:, 3])**2)
    angle_deg = np.degrees(phi2_0_variations[i])
    ax.plot(t, divergence, color=colors[i], alpha=0.7, 
            label=f'φ₂₀ = {angle_deg:.0f}°')

ax.set_xlabel('Время (с)')
ax.set_ylabel('Расстояние между траекториями')
ax.set_title('Расхождение траекторий маятника')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
