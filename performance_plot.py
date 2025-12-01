import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

groups = [r'$\sigma=1,\ p=20$', r'$\sigma=1,\ p=40$', r'$\sigma=3,\ p=20$', r'$\sigma=3,\ p=40$']

methods = [
    "deepHTL",
    "R-dnn",
    "Rev-Lasso",
    "R-Lasso",
    "Rev-XGBoost",
    "R-XGBoost",
    "Rev-KRR",
    "R-KRR",
]

# lighter blues for deep methods
colors = {
    "deepHTL":     (139/255, 169/255, 219/255),
    "R-dnn":       (103/255, 146/255, 207/255),
    "Rev-Lasso":   (50/255, 160/255,  44/255),
    "R-Lasso":     (153/255, 199/255, 148/255),
    "Rev-XGBoost": (230/255, 159/255, 0/255),
    "R-XGBoost":   (253/255, 191/255, 111/255),
    "Rev-KRR":     (148/255, 103/255, 189/255),
    "R-KRR":       (197/255, 176/255, 213/255),
}

# mean values only
I_1000 = {
    "deepHTL":     [-0.711, -0.402, 0.333, 0.607],
    "R-dnn":       [-0.682, -0.381, 0.355, 0.628],
    "R-Lasso":     [0.252, 0.292, 0.534, 0.824],
    "Rev-Lasso":   [0.251, 0.292, 0.532, 0.822],
    "R-XGBoost":   [0.294, 0.376, 1.019, 1.121],
    "Rev-XGBoost": [0.283, 0.370, 1.014, 1.118],
    "R-KRR":       [0.104, 0.458, 0.677, 0.958],
    "Rev-KRR":     [0.046, 0.448, 0.669, 0.954],
}
I_2000 = {
    "deepHTL":     [-0.919, -0.874, 0.001, 0.113],
    "R-dnn":       [-0.847, -0.769, 0.029, 0.154],
    "R-Lasso":     [0.152, 0.184, 0.319, 0.387],
    "Rev-Lasso":   [0.152, 0.183, 0.319, 0.387],
    "R-XGBoost":   [-0.069, 0.006, 0.694, 0.790],
    "Rev-XGBoost": [-0.088, -0.018, 0.679, 0.776],
    "R-KRR":       [-0.382, 0.183, 0.356, 0.668],
    "Rev-KRR":     [-0.436, 0.150, 0.342, 0.676],
}

II_1000 = {
    "deepHTL":     [-0.068, 0.241, 0.307, 0.337],
    "R-dnn":       [-0.010, 0.353, 0.382, 0.401],
    "R-Lasso":     [0.410, 0.423, 0.459, 0.455],
    "Rev-Lasso":   [0.410, 0.424, 0.463, 0.452],
    "R-XGBoost":   [0.389, 0.468, 0.817, 0.841],
    "Rev-XGBoost": [0.319, 0.406, 0.546, 0.551],
    "R-KRR":       [0.410, 0.356, 0.488, 0.460],
    "Rev-KRR":     [0.272, 0.412, 0.446, 0.434],
}
II_2000 = {
    "deepHTL":     [-0.292, -0.218, 0.265, 0.269],
    "R-dnn":       [-0.269, -0.189, 0.306, 0.309],
    "R-Lasso":     [0.413, 0.391, 0.405, 0.383],
    "Rev-Lasso":   [0.413, 0.392, 0.406, 0.382],
    "R-XGBoost":   [0.193, 0.299, 0.623, 0.550],
    "Rev-XGBoost": [0.156, 0.285, 0.460, 0.409],
    "R-KRR":       [0.081, 0.403, 0.427, 0.405],
    "Rev-KRR":     [-0.027, 0.367, 0.388, 0.377],
}

def panel(ax, data, title, letter=None, ylim=None):
    x = np.arange(len(groups))
    m = len(methods)
    width = 0.80 / m
    for i, method in enumerate(methods):
        vals = np.array(data[method])
        ax.bar(x + (i - m/2 + 0.5)*width, vals, width=width, color=colors[method], alpha=0.95)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=0)
    ax.set_ylabel("log-MSE")
    ax.set_title(title, fontsize=11, pad=4)
    ax.yaxis.grid(False)
    ax.axhline(0, color='0.2', lw=0.8) 

    if ylim is not None:
        ax.set_ylim(*ylim)
    if letter:
        ax.text(-0.03, 1.02, f'({letter})', transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='bottom')

fig, axes = plt.subplots(4, 1, figsize=(8.5, 11), dpi=200, sharex=True)
panel(axes[0], I_1000,  "Scenario I — n = 1000", letter='a', ylim=(-0.8, 1.2))
panel(axes[1], I_2000,  "Scenario I — n = 2000", letter='b', ylim=(-1.0, 0.6))
panel(axes[2], II_1000, "Scenario II — n = 1000", letter='c', ylim=(-0.1, 0.85))
panel(axes[3], II_2000, "Scenario II — n = 2000", letter='d', ylim=(-0.3, 0.65))

plt.xlabel("Scenario $(\\sigma,\\ p)$", fontsize=11)

handles = [plt.Rectangle((0,0),1,1,color=colors[m]) for m in methods]
labels = methods
legend = fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.00, 0.50),
                    frameon=True, ncol=1, title="Method", borderaxespad=0.4, fontsize=8)
legend.get_title().set_fontsize(9)

fig.subplots_adjust(right=0.90, hspace=0.55)
fig.savefig("Simu.png", dpi = 600, bbox_inches='tight')
