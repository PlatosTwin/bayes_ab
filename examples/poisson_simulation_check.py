import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import matplotlib.colors as mcolors

from bayes_ab.experiments import PoissonDataTest


def min_max(x_min, x_max, x_plot, y_plot):
    if x_plot[np.where(y_plot >= 0.0001)[0][0]] < x_min:
        x_min = x_plot[np.where(y_plot >= 0.0001)[0][0]]
    if x_plot[np.where(y_plot >= 0.0001)[0][-1]] > x_max:
        x_max = x_plot[np.where(y_plot >= 0.0001)[0][-1]]

    return x_min, x_max


# generating some random data
rng = np.random.default_rng(52)
# random 1x1500 array of 0/1 data with 5.2% probability for 1:
data_a = rng.poisson(43, size=200)
data_b = rng.poisson(43, size=250)
data_c = rng.poisson(38, size=150)

# initialize a test:
test = PoissonDataTest()

# add variant using raw data (arrays of zeros and ones) and informative priors:
test.add_variant_data("A", data_a, a_prior=10, b_prior=17)
test.add_variant_data("B", data_b, a_prior=5, b_prior=30)
test.add_variant_data("C", data_c, a_prior=10, b_prior=10)

# evaluate test:
test.evaluate(closed_form=False, sim_count=2000000)

# access simulation samples and evaluation metrics
data = test.data

# setup plots
fig, ax = plt.subplots(
    figsize=(10, 8),
)

colors = list(mcolors.TABLEAU_COLORS.values())
xmin = 100
xmax = 0

###

x = np.linspace(0, 100, 10000)
y = stats.gamma.pdf(x, a=10 + len(data_a) * np.mean(data_a), scale=1 / (17 + len(data_a)))
ax.plot(x, y, "k--")
ax.hist(data["A"]["samples"], bins=500, density=True, color=colors[0], label="A")

xmin, xmax = min_max(xmin, xmax, x, y)

###

x = np.linspace(0, 100, 10000)
y = stats.gamma.pdf(x, a=5 + len(data_b) * np.mean(data_b), scale=1 / (30 + len(data_b)))
ax.plot(x, y, "k--")
ax.hist(data["B"]["samples"], bins=500, density=True, color=colors[1], label="B")

xmin, xmax = min_max(xmin, xmax, x, y)

###

x = np.linspace(0, 100, 10000)
y = stats.gamma.pdf(x, a=10 + len(data_c) * np.mean(data_c), scale=1 / (10 + len(data_c)))
ax.plot(x, y, "k--")
ax.hist(data["C"]["samples"], bins=500, density=True, color=colors[2], label="C", alpha=0.65)

xmin, xmax = min_max(xmin, xmax, x, y)

###

ax.set_xlim(xmin * 0.95, xmax * 1.05)
ax.legend()
plt.title("Poisson test: sampled posterior distribution vs. analytical posterior distribution")

fig.tight_layout()

plt.savefig("plots/poisson_simulation_check.png", dpi=300)
plt.show()
