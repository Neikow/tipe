from math import log10, pi
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from matplotlib.widgets import Slider
import random

X1 = [30000.0, 31000.0, 32000.0, 33000.0, 34000.0, 35000.0, 36000.0, 37000.0, 38000.0, 39000.0, 40000.0, 41000.0, 42000.0, 43000.0, 44000.0, 45000.0, 46000.0, 47000.0, 48000.0, 49000.0, 50000.0, 51000.0, 52000.0, 53000.0, 54000.0, 55000.0, 56000.0, 57000.0, 58000.0, 59000.0, 60000.0, 61000.0, 62000.0, 63000.0, 64000.0, 65000.0, 66000.0, 67000.0, 68000.0, 69000.0, 70000.0, 71000.0, 72000.0, 73000.0, 74000.0, 75000.0, 76000.0, 77000.0, 78000.0, 79000.0, 80000.0, 81000.0, 82000.0, 83000.0, 84000.0, 85000.0, 86000.0, 87000.0, 88000.0, 89000.0, 90000.0, 91000.0, 92000.0, 93000.0, 94000.0, 95000.0]
Y1 = [52.736318407960226, 52.1091811414392, 48.41075794621028, 46.48910411622276, 39.99999999999999, 39.71291866028709, 36.70588235294119, 37.264150943396224, 31.090487238979158, 33.41067285382834, 30.733944954128432, 27.126436781609232, 27.126436781609232, 23.18181818181817, 23.18181818181817, 23.18181818181817, 19.955654101995574, 20.444444444444443, 16.888888888888882, 22.421524663677133, 16.814159292035434, 16.814159292035434, 17.294900221729517, 15.077605321507756, 15.077605321507756, 17.294900221729517, 17.294900221729517, 16.814159292035434, 21.524663677130064, 19.999999999999975, 28.438228438228453, 40.19370460048427, 61.29870129870131, 92.79538904899134, 110.57401812688819, 100.58479532163742, 82.73972602739728, 67.89473684210526, 51.758793969849236, 45.96577017114917, 43.583535108958856, 40.66985645933017, 27.126436781609232, 23.18181818181817, 25.747126436781638, 25.454545454545435, 22.727272727272727, 22.727272727272727, 21.818181818181795, 20.316027088036126, 19.730941704035892, 16.814159292035434, 16.814159292035434, 20.39911308203991, 13.973799126637568, 13.973799126637568, 14.442013129102845, 14.442013129102845, 14.442013129102845, 10.50328227571113, 10.50328227571113, 10.50328227571113, 10.940919037199125, 10.940919037199125, 11.37855579868708, 11.37855579868708]

def H(w, Cp, Cs, Rs, Ls):
    CsCp = Cs + Cp
    num = 1j * Rs * Cs * w - Ls * Cs * (w ** 2) + 1
    denum = 1j * w * CsCp * ((1j * Rs * Cp * Cs * w) / CsCp + 1 - (Cs * Cp * Ls * (w ** 2)) / CsCp)

    return abs(num / denum)


Cp = 5.2191e-08
Ls = 0.000406
Rs = 4.293
Cs = 2.3214e-07



# def dB(x):
#   return 20 * log10(abs(x))

X = np.linspace(min(X1), max(X1), 1000)
plt.plot(X1, [1 / y for y in Y1])


popt, pcov = curve_fit(H, X1, [1 / y for y in Y1], [Cp, Cs, Rs, Ls], bounds=[0, [1, 1, 1000, 1]])

Y2 = [H(x, *popt) for x in X]
plt.plot(X, Y2)

plt.xscale('log')
plt.yscale('log')

plt.show()

# Rs = [0, 1, 10, 100, 1000, 10000]
# Rs = [10]
# Rs = 10
# Cs = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]
# Cs = 1e-8
# Cp = [1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16]
# Cp = 1e-13
# Ls = [1e-4, 1e-5, 1e-6, 1e-7]
# Ls = 1e-5

# tot = len(Rs) * len(Cs) * len(Cp) * len(Ls)

Rs, Cs, Cp, Ls = 0.5, 0.5, 0.5, 0.5

X = np.linspace(1, 10000000000, 5000)
W = X * 2 * pi

# for rs in Rs:
#   for cs in Cs:
#     for cp in Cp:
#       for ls in Ls:
#         def H(w):
#           CsCp = cs + cp
#           num = 1j * rs * cs * w - ls * cs * w * w + 1
#           denum = 1j * w * CsCp * ((1j * rs * cp * cs * w) / CsCp + 1 - (cs * cp * ls * w ** 2) / CsCp)

#           return abs(num / denum)

#         Y = [H(w) for w in W]
#         plt.plot(X, Y, label=f'Rs: {rs}, Cs: {cs}, Cp: {cp}, Ls: {ls}')

#         print(f'Il manque {tot - 1} courbes.')

#         tot -= 1

###

fig, ax = plt.subplots()

ax.plot(X1, Y1)

line, = ax.plot(W, H(W, Cp, Cs, Rs, Ls), lw=2)

plt.subplots_adjust(bottom=0.25)


def interp(x, min, max):
    return (max - min) * x + min


ax.set_title(f'Cp: {Cp}\nLs: {Ls}\nRs: {Rs}\nCs: {Cs}')


def update(val):
    global Cp, Cs, Rs, Ls
    Cp = 10 ** interp(slCp.val, -18, -3)
    Ls = 10 ** interp(slLs.val, -11, -1)
    Rs = 10 ** interp(slRs.val, 0, 5)
    Cs = 10 ** interp(slCs.val, -13, -2)
    # print(f'Cp: {Cp}\nLs: {Ls}\nRs: {Rs}\nCs: {Cs}')

    ax.set_title(f'Cp: {Cp}\nLs: {Ls}\nRs: {Rs}\nCs: {Cs}')

    line.set_ydata([H(w, Cp, Cs, Rs, Ls) for w in W])

    ax.relim()
    ax.autoscale_view()

    fig.canvas.draw_idle()


axLs = plt.axes([0.050, 0.05, 0.300, 0.050])
axRs = plt.axes([0.050, 0.10, 0.300, 0.050])
axCs = plt.axes([0.600, 0.05, 0.300, 0.050])
axCp = plt.axes([0.600, 0.10, 0.300, 0.050])

slLs = Slider(ax=axLs, label='Ls', valmin=0, valmax=1, valinit=Ls)
slRs = Slider(ax=axRs, label='Rs', valmin=0, valmax=1, valinit=Rs)
slCs = Slider(ax=axCs, label='Cs', valmin=0, valmax=1, valinit=Cs)
slCp = Slider(ax=axCp, label='Cp', valmin=0, valmax=1, valinit=Cp)

slRs.on_changed(update)
slCs.on_changed(update)
slCp.on_changed(update)
slLs.on_changed(update)

ax.set_xscale('log')
ax.set_yscale('log')

update(None)

plt.show()
