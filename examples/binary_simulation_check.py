import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import scipy.stats as stats
import matplotlib.colors as mcolors

from bayes_ab.experiments import BinaryDataTest


def min_max(x_min, x_max, x_plot, y_plot):
    if x_plot[np.where(y_plot >= 0.0001)[0][0]] < x_min:
        x_min = x_plot[np.where(y_plot >= 0.0001)[0][0]]
    if x_plot[np.where(y_plot >= 0.0001)[0][-1]] > x_max:
        x_max = x_plot[np.where(y_plot >= 0.0001)[0][-1]]

    return x_min, x_max


# generating some random data
rng = np.random.default_rng(52)
# random 1x1500 array of 0/1 data with 5.2% probability for 1:
data_a = rng.binomial(n=1, p=0.052, size=1500)
# random 1x1200 array of 0/1 data with 6.7% probability for 1:
data_b = rng.binomial(n=1, p=0.067, size=1200)

# initialize a test:
test = BinaryDataTest()

# add variant using raw data (arrays of zeros and ones) and informative priors:
test.add_variant_data("A", data_a, a_prior=10, b_prior=17)
test.add_variant_data("B", data_b, a_prior=5, b_prior=30)

# add variant using aggregated data and non-informative priors:
test.add_variant_data_agg("C", total=1000, positives=50)

# evaluate test:
test.evaluate(closed_form=False, sim_count=2000000)

# access simulation samples and evaluation metrics
data = test.data

# setup plots
fig, ax = plt.subplots(
    figsize=(10, 8),
)

colors = list(mcolors.TABLEAU_COLORS.values())
xmin = 1
xmax = 0

###

x = np.linspace(0, 1, 10000)
y = stats.beta.pdf(x, 10 + data_a.sum(), 17 + len(data_a) - data_a.sum())
ax.plot(x * 100, y / 100, "k--")
ax.hist(data["A"]["samples"] * 100, bins=500, density=True, color=colors[0], label="A")

xmin, xmax = min_max(xmin, xmax, x, y)

###

x = np.linspace(0, 1, 10000)
y = stats.beta.pdf(x, 5 + data_b.sum(), 30 + len(data_b) - data_b.sum())
ax.plot(x * 100, y / 100, "k--")
ax.hist(data["B"]["samples"] * 100, bins=500, density=True, color=colors[1], label="B")

xmin, xmax = min_max(xmin, xmax, x, y)

###

x = np.linspace(0, 1, 10000)
y = stats.beta.pdf(x, 1 + 50, 1 + 1000 - 50)
ax.plot(x * 100, y / 100, "k--")
ax.hist(data["C"]["samples"] * 100, bins=500, density=True, color=colors[2], label="C", alpha=0.65)

xmin, xmax = min_max(xmin, xmax, x, y)

###

ax.xaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_xlim(xmin * 80, xmax * 120)
ax.legend()
plt.title("Binary test: sampled posterior distribution vs. analytical posterior distribution")

fig.tight_layout()

plt.savefig("plots/binary_simulation_check.png", dpi=300)
plt.show()
