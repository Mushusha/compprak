import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as anim

L = 1
g = 9.81
omega0 = g / L
alpha0 = math.radians(5.0)

time = 10

nsteps = 100
dt = time / nsteps

x = np.linspace(0, time, nsteps + 1)
y_an = alpha0 * np.cos(omega0 * x)
y_num = np.zeros(nsteps + 1)

plt.plot(x, y_an)
plt.axis('equal')
plt.show()


# fig, ax = plt.subplots()
# ax.set_xlim(0, flight_range)
# ax.set_ylim(0, max(y_an) * 1.1)

# traject_an = ax.plot(x, y_an, lw=3)[0]
# traject_num = ax.plot(x[:1], y_num[:1], lw=3)[0]

# def init_anim():
#     traject_an.set_data(x, y_an)
#     traject_num.set_data(x[:1], y_num[:1])
#     return traject_an, traject_num

# def loop_anim(i):
#     global vy
#     k1 = vy
#     k2 = vy - 0.5 * g * dt
#     k3 = vy - 0.5 * g * dt
#     k4 = vy - g * dt

#     y_num[i + 1] = y_num[i] + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6.0
#     #y_num[i + 1] = y_num[i] + vy * dt

#     vy = vy - g * dt
#     traject_num.set_data(x[:i + 2], y_num[:i + 2])
#     return traject_an, traject_num

# ani = anim.FuncAnimation(fig=fig, func=loop_anim, 
#                          init_func=init_anim, frames=nsteps, 
#                          interval=10, repeat=False, blit=True)
# plt.show()
