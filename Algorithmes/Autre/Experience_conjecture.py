import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Définir la fonction z=f(x,y)
def f(x, y):
    return abs(np.sqrt(x) + np.sqrt(y))


# Créer une grille de points
x = np.linspace(0, 2, 100)
y = np.linspace(0, 2, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0)
ax.set_title(r"$\sigma=|\sqrt{p} + \sqrt{n}|$")
ax.set_xlabel('p')
ax.set_ylabel('n')
ax.set_zlabel(r'$\sigma$')

plt.show()
