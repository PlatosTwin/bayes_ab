from numbers import Number
from typing import List, Tuple
from scipy.stats import t, gamma
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import warnings

from bayes_ab.experiments.base import BaseDataTest
from bayes_ab.metrics import eval_normal_agg
from bayes_ab.utilities import get_logger, print_normal_evaluation

logger = get_logger("bayes_ab")


class NormalDataTest(BaseDataTest):
    """
    Class for Bayesian A/B test for normal data.

    After class initialization, use add_variant methods to insert variant data.
    Then to get results of the test, use for instance `evaluate` method.
    """

    def __init__(self) -> None:
        """
        Initialize NormalDataTest class.
        """
        super().__init__()

    @property
    def totals(self):
        return [self.data[k]["total"] for k in self.data]

    @property
    def sum_values(self):
        return [self.data[k]["sum_values"] for k in self.data]

    @property
    def sum_squares(self):
        return [self.data[k]["sum_squares"] for k in self.data]

    @property
    def sum_values_squared(self):
        return [self.data[k]["sum_values_squared"] for k in self.data]

    @property
    def m_priors(self):
        return [self.data[k]["m_prior"] for k in self.data]

    @property
    def v_priors(self):
        return [self.data[k]["v_prior"] for k in self.data]

    @property
    def s_2_priors(self):
        return [self.data[k]["s_2_prior"] for k in self.data]

    @property
    def n_priors(self):
        return [self.data[k]["n_prior"] for k in self.data]

    @property
    def means(self):
        return [self.data[k]["mean"] for k in self.data]

    @property
    def bounds(self):
        return [self.data[k]["bounds"] for k in self.data]

    @property
    def precisions(self):
        return [self.data[k]["precision"] for k in self.data]

    @property
    def stdevs(self):
        return [self.data[k]["stdev"] for k in self.data]

    @property
    def stdev_bounds(self):
        return [self.data[k]["stdev_bounds"] for k in self.data]

    @property
    def chance_to_beat(self):
        try:
            return [self.data[k]["chance_to_beat"] for k in self.data]
        except KeyError:
            msg = "You must run the evaluate method before attempting to access this property."
            raise NotImplementedError(msg)

    @property
    def exp_loss(self):
        try:
            return [self.data[k]["exp_loss"] for k in self.data]
        except KeyError:
            msg = "You must run the evaluate method before attempting to access this property."
            raise NotImplementedError(msg)

    def _eval_simulation(self, sim_count: int = 20000, seed: int = None) -> Tuple[dict, dict]:
        """
        Calculate probabilities of being best and expected loss for a current class state.

        Parameters
        ----------
        sim_count : Number of simulations to be used for probability estimation.
        seed : Random seed.

        Returns
        -------
        res_pbbs : Dictionary with probabilities of being best for all variants in experiment.
        res_loss : Dictionary with expected loss for all variants in experiment.
        """
        pbbs, loss, samples = eval_normal_agg(
            self.totals,
            self.sum_values,
            self.sum_squares,
            sim_count=sim_count,
            m_priors=self.m_priors,
            v_priors=self.v_priors,
            s_2_priors=self.s_2_priors,
            n_priors=self.n_priors,
            seed=seed,
        )
        res_pbbs = dict(zip(self.variant_names, pbbs))
        res_loss = dict(zip(self.variant_names, loss))

        for i, var in enumerate(self.variant_names):
            self.data[var]["samples"] = samples[i]

        return res_pbbs, res_loss

    def evaluate(self, sim_count: int = 20000, verbose: bool = True, seed: int = None) -> List[dict]:
        """
        Evaluation of experiment.

        Parameters
        ----------
        sim_count : Number of simulations to be used for probability estimation.
        verbose : If True, output prints to console.
        seed : Random seed.

        Returns
        -------
        res : List of dictionaries with results per variant.
        """
        keys = [
            "variant",
            "total",
            "mean",
            "prob_being_best",
            "expected_loss",
            "uplift_vs_a",
            "bounds",
            "precision",
            "stdev",
            "stdev_bounds",
        ]
        eval_pbbs, eval_loss = self._eval_simulation(sim_count, seed)
        pbbs = list(eval_pbbs.values())
        loss = list(eval_loss.values())

        uplift = [0]
        for i in self.means[1:]:
            uplift.append(round((i - self.means[0]) / self.means[0], 5))

        for i, var in enumerate(self.variant_names):
            self.data[var]["chance_to_beat"] = pbbs[i]
            self.data[var]["exp_loss"] = loss[i]
            self.data[var]["uplift_vs_a"] = uplift[i]

        data = [
            self.variant_names,
            self.totals,
            self.means,
            pbbs,
            loss,
            uplift,
            self.bounds,
            self.precisions,
            self.stdevs,
            self.stdev_bounds,
        ]
        res = [dict(zip(keys, item)) for item in zip(*data)]

        if verbose:
            print_normal_evaluation(res)

        return res

    def add_variant_data_agg(
        self,
        name: str,
        total: int,
        sum_values: float,
        sum_squares: float,
        sum_values_squared: float,
        m_prior: float = 0,
        v_prior: float = -1,
        s_2_prior: float = 0,
        n_prior: float = 0,
        replace: bool = True,
    ) -> None:
        """
        Add variant data to test class using aggregated normal data.
        This can be convenient as aggregation can be done on database level.

        The goal of default prior setup is to be low information.
        It should be tuned with caution.

        Parameters
        ----------
        name : Variant name.
        total : Total number of experiment observations (e.g. number of sessions).
        sum_values : Sum of values for a given variant.
        sum_squares : Sum of squares for a given variant: sum[(y - y_bar)**2].
        sum_values_squared : Sum of observations squared: sum(y**2).
        m_prior : Estimate for the prior mean.
        v_prior : Estimate for the prior degrees of freedom. This is one input to the alpha parameter for the inverse
            gamma distribution.
        s_2_prior : Estimate for the prior variance. This is one input to the beta parameter for the inverse gamma
            distribution.
        n_prior : Estimate for the prior sample size.
        replace : Replace data if variant already exists.
            If set to False, data of existing variant will be appended to existing data.
        """
        if not isinstance(name, str):
            raise ValueError("Variant name has to be a string.")
        if m_prior < 0 or v_prior < -1 or s_2_prior < 0 or n_prior < 0:
            raise ValueError("All priors of [m, v, s_2, w] have to be non-negative numbers.")
        if total <= 0:
            raise ValueError("Input variable 'total' is expected to be positive integer.")

        if name not in self.variant_names:
            y_bar = sum_values / total
            mu = (total * y_bar + n_prior * m_prior) / (total + n_prior)

            v_n = v_prior + total
            n_n = n_prior + total
            s_n_2 = (1 / v_n) * (sum_squares + s_2_prior * v_prior + (n_prior * total / n_n) * (y_bar - m_prior) ** 2)

            inv_gamma_alpha = (1 / 2) * v_n
            inv_gamma_beta = (1 / 2) * s_n_2 * v_n

            self.data[name] = {
                "total": total,
                "sum_values": round(sum_values, 5),
                "sum_squares": round(sum_squares, 5),
                "sum_values_squared": round(sum_values_squared, 5),
                "m_prior": m_prior,
                "v_prior": v_prior,
                "s_2_prior": s_2_prior,
                "n_prior": n_prior,
                "mean": round(mu, 5),
                "bounds": [
                    round(t.ppf(0.025, v_n, mu, np.sqrt(s_n_2 / n_n)), 5),
                    round(t.ppf(0.975, v_n, mu, np.sqrt(s_n_2 / n_n)), 5),
                ],
                "precision": round(inv_gamma_alpha / inv_gamma_beta, 5),
                "stdev": round(np.sqrt(inv_gamma_beta / inv_gamma_alpha), 5),
                "stdev_bounds": [
                    round(1 / np.sqrt(gamma.ppf(0.975, a=inv_gamma_alpha, scale=1 / inv_gamma_beta)), 5),
                    round(1 / np.sqrt(gamma.ppf(0.025, a=inv_gamma_alpha, scale=1 / inv_gamma_beta)), 5),
                ],
            }
        elif name in self.variant_names and replace:
            msg = (
                f"Variant {name} already exists - new data is replacing it. "
                "If you wish to append instead, use replace=False."
            )
            logger.info(msg)

            y_bar = sum_values / total
            mu = (total * y_bar + n_prior * m_prior) / (total + n_prior)

            v_n = v_prior + total
            n_n = n_prior + total
            s_n_2 = (1 / v_n) * (
                sum_squares + s_2_prior * v_prior + (n_prior * total / (n_prior + total)) * (y_bar - m_prior) ** 2
            )

            inv_gamma_alpha = (1 / 2) * v_n
            inv_gamma_beta = (1 / 2) * s_n_2 * v_n

            self.data[name] = {
                "total": total,
                "sum_values": round(sum_values, 5),
                "sum_squares": round(sum_squares, 5),
                "sum_values_squared": round(sum_values_squared, 5),
                "m_prior": m_prior,
                "v_prior": v_prior,
                "s_2_prior": s_2_prior,
                "n_prior": n_prior,
                "mean": round(mu, 5),
                "bounds": [
                    round(t.ppf(0.025, v_n, mu, np.sqrt(s_n_2 / n_n)), 5),
                    round(t.ppf(0.975, v_n, mu, np.sqrt(s_n_2 / n_n)), 5),
                ],
                "precision": round(inv_gamma_alpha / inv_gamma_beta, 5),
                "stdev": round(np.sqrt(inv_gamma_beta / inv_gamma_alpha), 5),
                "stdev_bounds": [
                    round(1 / np.sqrt(gamma.ppf(0.975, a=inv_gamma_alpha, scale=1 / inv_gamma_beta)), 5),
                    round(1 / np.sqrt(gamma.ppf(0.025, a=inv_gamma_alpha, scale=1 / inv_gamma_beta)), 5),
                ],
            }
        elif name in self.variant_names and not replace:
            msg = (
                f"Variant {name} already exists - new data is appended to variant, "
                "keeping its original prior setup. "
                "If you wish to replace data instead, use replace=True."
            )
            logger.info(msg)
            self.data[name]["total"] += round(total, 5)
            self.data[name]["sum_values"] += round(sum_values, 5)
            self.data[name]["sum_values_squared"] += round(sum_values_squared, 5)
            self.data[name]["sum_squares"] = round(
                self.data[name]["sum_values_squared"]
                + self.data[name]["total"] * (self.data[name]["sum_values"] / self.data[name]["total"]) ** 2
                - 2 * self.data[name]["sum_values"] / self.data[name]["total"] * self.data[name]["sum_values"],
                5,
            )

            y_bar = self.data[name]["sum_values"] / self.data[name]["total"]
            mu = (self.data[name]["total"] * y_bar + n_prior * m_prior) / (self.data[name]["total"] + n_prior)

            self.data[name]["mean"] = round(mu, 5)

            v_n = v_prior + self.data[name]["total"]
            n_n = n_prior + self.data[name]["total"]
            s_n_2 = (1 / v_n) * (
                self.data[name]["sum_squares"]
                + s_2_prior * v_prior
                + (n_prior * self.data[name]["total"] / (n_prior + self.data[name]["total"])) * (y_bar - m_prior) ** 2
            )

            self.data[name]["bounds"] = [
                round(t.ppf(0.025, v_n, mu, np.sqrt(s_n_2 / n_n)), 5),
                round(t.ppf(0.975, v_n, mu, np.sqrt(s_n_2 / n_n)), 5),
            ]

            inv_gamma_alpha = (1 / 2) * v_n
            inv_gamma_beta = (1 / 2) * s_n_2 * v_n
            self.data[name]["precision"] = round(inv_gamma_alpha / inv_gamma_beta, 5)
            self.data[name]["stdev"] = round(np.sqrt(inv_gamma_beta / inv_gamma_alpha), 5)

            self.data[name]["stdev_bounds"] = [
                round(1 / np.sqrt(gamma.ppf(0.975, a=inv_gamma_alpha, scale=1 / inv_gamma_beta)), 5),
                round(1 / np.sqrt(gamma.ppf(0.025, a=inv_gamma_alpha, scale=1 / inv_gamma_beta)), 5),
            ]

    def add_variant_data(
        self,
        name: str,
        data: List[Number],
        m_prior: float = 0,
        v_prior: float = -1,
        s_2_prior: float = 0,
        n_prior: float = 0,
        replace: bool = True,
    ) -> None:
        """
        Add variant data to test class using raw normal data.

        The goal of default prior setup is to be low information. It should be tuned with caution.

        Parameters
        ----------
        name : Variant name.
        data : List of normal data.
        m_prior : Estimate for the prior mean.
        v_prior : Estimate for the prior degrees of freedom. This is one input to the alpha parameter for the inverse
            gamma distribution.
        s_2_prior : Estimate for the prior variance. This is one input to the beta parameter for the inverse gamma
            distribution.
        n_prior : Estimate for the prior sample size.
        replace : Replace data if variant already exists.
            If set to False, data of existing variant will be appended to existing data.
        """
        if len(data) == 0:
            raise ValueError("Data of added variant needs to have some observations.")

        total = len(data)
        sum_values = sum(data)
        sum_values_squared = sum(np.square(data))
        sum_squares = sum((data - np.mean(data)) ** 2)

        self.add_variant_data_agg(
            name,
            total,
            sum_values,
            sum_squares,
            sum_values_squared,
            m_prior,
            v_prior,
            s_2_prior,
            n_prior,
            replace,
        )

    def plot_joint_prior(self, variant: str, fname: str = None, dpi: int = 300) -> plt.figure:
        """
        For each variant, plot its posterior distribution.

        Parameters
        ----------
        variant : The variant to treat as control; this variant will be subtracted from each other variant.
        fname : Filename to which to save the resultant image; if None, the image is not saved.
        dpi : DPI setting for saved image; used only when fname is not None.
        """
        if variant not in self.variant_names:
            raise ValueError(f"variant name must correspond to an added variant: {self.variant_names}.")

        fig = plt.figure(figsize=(10, 8))

        ax = fig.add_subplot(1, 1, 1, projection="3d")
        m_0 = self.data[variant]["m_prior"]
        s_2_0 = self.data[variant]["s_2_prior"]
        n_0 = self.data[variant]["n_prior"]
        v_0 = self.data[variant]["v_prior"]

        if 0 in [m_0, s_2_0, n_0] or v_0 == -1:
            raise ValueError(
                "To plot the joint prior for this variant, initialize it with at least"
                " weakly information prior parameters."
            )

        mesh = 1000
        mu = np.linspace(m_0 * 0.25, m_0 * 1.75, mesh)
        sigma = np.linspace(1e-5, np.sqrt(s_2_0) * 1.75, mesh)
        mu, sigma = np.meshgrid(mu, sigma)
        z = (
            1
            / sigma
            * (sigma**2) ** -(v_0 / 2 + 1)
            * np.exp(-1 / (2 * sigma**2) * (v_0 * s_2_0 + n_0 * (m_0 - mu) ** 2))
        )

        z = z / np.max(z)

        mu_min = mu[0, np.where(np.max(z, axis=0) >= 0.0001)[0][0]]
        mu_max = mu[0, np.where(np.max(z, axis=0) >= 0.0001)[0][-1]]

        sigma_min = sigma[np.where(np.max(z, axis=1) >= 0.0001)[0][0], 0]
        sigma_max = sigma[np.where(np.max(z, axis=1) >= 0.0001)[0][-1], 0]

        ax.plot_surface(mu, sigma, z, edgecolor="royalblue", lw=0.3, rstride=12, cstride=12, alpha=0.3)

        ax.contour(mu, sigma, z, zdir="z", cmap="coolwarm", offset=-0.1, levels=np.arange(0, 1, 0.1))
        ax.contour(
            mu,
            sigma,
            z,
            zdir="x",
            cmap="coolwarm",
            offset=m_0 * 0.25,
            levels=np.arange(mu_min, mu_max, (mu_max - mu_min) / 20),
        )
        ax.contour(
            mu,
            sigma,
            z,
            zdir="y",
            cmap="coolwarm",
            offset=np.sqrt(s_2_0) * 1.75,
            levels=np.arange(sigma_min, sigma_max, (sigma_max - sigma_min) / 20),
        )

        ax.set(xlabel=r"$\mu$", ylabel=r"$\sigma^2$", zlabel=r"$\propto p(\mu, \sigma^2$)")

        plt.suptitle(f"Joint prior distribution for variant {variant}")

        fig.tight_layout()

        if fname:
            plt.savefig(fname, dpi=dpi)

        plt.show()

        return fig

    def plot_distributions(self, control: str, fname: str = None, dpi: int = 300) -> plt.figure:
        """
        For each variant, plot its posterior distribution.

        Parameters
        ----------
        control : The variant to treat as control; this variant will be subtracted from each other variant.
        fname : Filename to which to save the resultant image; if None, the image is not saved.
        dpi : DPI setting for saved image; used only when fname is not None.
        """
        if len(self.variant_names) == 1:
            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                figsize=(10, 8),
            )
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(
                3,
                1,
                figsize=(10, 8),
            )

        ###
        # subplot 1: mean distribution
        ###
        xmin = 1e6
        xmax = 0
        colors = list(mcolors.TABLEAU_COLORS.values())
        dist_names = []
        for var, color in zip(self.data, colors):
            m_prior = self.data[var]["m_prior"]
            s_2_prior = self.data[var]["s_2_prior"]
            n_prior = self.data[var]["n_prior"]
            v_prior = self.data[var]["v_prior"]
            sum_squares = self.data[var]["sum_squares"]
            sum_values = self.data[var]["sum_values"]
            n = self.data[var]["total"]

            y_bar = sum_values / n
            mu = (n * y_bar + n_prior * m_prior) / (n + n_prior)
            v_n = v_prior + n
            n_n = n_prior + n
            s_n_2 = (1 / v_n) * (sum_squares + s_2_prior * v_prior + (n_prior * n / n_n) * (y_bar - m_prior) ** 2)

            label = f"{var}: " + r"$\mu=" + f"{mu:.2f}$"
            dist_names.append(label)
            hdi_buffer = (self.data[var]["bounds"][1] - self.data[var]["bounds"][0]) / 2
            x = np.linspace(
                max(1e-5, self.data[var]["bounds"][0] - hdi_buffer), self.data[var]["bounds"][1] + hdi_buffer, 100000
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y = t.pdf(x, df=v_n, loc=mu, scale=np.sqrt(s_n_2 / n_n))
            ax1.plot(x, y, label=label)

            x_bound = x[
                np.intersect1d(np.where(x > self.data[var]["bounds"][0])[0], np.where(x < self.data[var]["bounds"][1]))
            ]
            y_bound = y[
                np.intersect1d(np.where(x > self.data[var]["bounds"][0])[0], np.where(x < self.data[var]["bounds"][1]))
            ]
            ax1.fill_between(x_bound, y_bound, color=color, alpha=0.55)

            x_bound = x[np.where(x < self.data[var]["bounds"][0])[0]]
            y_bound = y[np.where(x < self.data[var]["bounds"][0])[0]]
            ax1.fill_between(x_bound, y_bound, color=color, alpha=0.10)

            x_bound = x[np.where(x > self.data[var]["bounds"][1])[0]]
            y_bound = y[np.where(x > self.data[var]["bounds"][1])[0]]
            ax1.fill_between(x_bound, y_bound, color=color, alpha=0.10)

            if x[np.where(y >= 0.0001)[0][0]] < xmin:
                xmin = x[np.where(y >= 0.0001)[0][0]]
            if x[np.where(y >= 0.0001)[0][-1]] > xmax:
                xmax = x[np.where(y >= 0.0001)[0][-1]]

        ax1.set_xlabel(r"Posterior marginal distribution for $\mu$")

        handles = [Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(self.data))]
        ax1.legend(handles, dist_names)

        ax1.set_xlim(xmin * 0.99, xmax * 1.01)

        ###
        # subplot 2: 1/sigma**2 distribution
        ###
        xmin = 1e6
        xmax = 0
        colors = list(mcolors.TABLEAU_COLORS.values())
        dist_names = []
        for var, color in zip(self.data, colors):
            m_prior = self.data[var]["m_prior"]
            s_2_prior = self.data[var]["s_2_prior"]
            n_prior = self.data[var]["n_prior"]
            v_prior = self.data[var]["v_prior"]
            sum_squares = self.data[var]["sum_squares"]
            sum_values = self.data[var]["sum_values"]
            n = self.data[var]["total"]

            y_bar = sum_values / n
            v_n = v_prior + n
            n_n = n_prior + n
            s_n_2 = (1 / v_n) * (sum_squares + s_2_prior * v_prior + (n_prior * n / n_n) * (y_bar - m_prior) ** 2)
            inv_gamma_alpha = (1 / 2) * v_n
            inv_gamma_beta = (1 / 2) * s_n_2 * v_n

            label = f"{var}: " + r"$\frac{1}{\sigma^2}" + f"={inv_gamma_alpha / inv_gamma_beta:.4f}$"
            dist_names.append(label)
            hdi_buffer = (1 / self.data[var]["stdev_bounds"][0] ** 2 - 1 / self.data[var]["stdev_bounds"][1] ** 2) / 2
            x = np.linspace(
                max(1e-5, 1 / self.data[var]["stdev_bounds"][1] ** 2 - hdi_buffer),
                1 / self.data[var]["stdev_bounds"][0] ** 2 + hdi_buffer,
                100000,
            )
            y = gamma.pdf(x, a=inv_gamma_alpha, scale=1 / inv_gamma_beta)
            ax2.plot(x, y, label=label)

            x_bound = x[
                np.intersect1d(
                    np.where(x > 1 / self.data[var]["stdev_bounds"][1] ** 2)[0],
                    np.where(x < 1 / self.data[var]["stdev_bounds"][0] ** 2),
                )
            ]
            y_bound = y[
                np.intersect1d(
                    np.where(x > 1 / self.data[var]["stdev_bounds"][1] ** 2)[0],
                    np.where(x < 1 / self.data[var]["stdev_bounds"][0] ** 2),
                )
            ]
            ax2.fill_between(x_bound, y_bound, color=color, alpha=0.55)

            x_bound = x[np.where(x < 1 / self.data[var]["stdev_bounds"][1] ** 2)[0]]
            y_bound = y[np.where(x < 1 / self.data[var]["stdev_bounds"][1] ** 2)[0]]
            ax2.fill_between(x_bound, y_bound, color=color, alpha=0.10)

            x_bound = x[np.where(x > 1 / self.data[var]["stdev_bounds"][0] ** 2)[0]]
            y_bound = y[np.where(x > 1 / self.data[var]["stdev_bounds"][0] ** 2)[0]]
            ax2.fill_between(x_bound, y_bound, color=color, alpha=0.10)

            if x[np.where(y >= 0.0001)[0][0]] < xmin:
                xmin = x[np.where(y >= 0.0001)[0][0]]
            if x[np.where(y >= 0.0001)[0][-1]] > xmax:
                xmax = x[np.where(y >= 0.0001)[0][-1]]

        ax2.set_xlabel(r"Posterior distribution for $\frac{1}{\sigma^2}$")

        handles = [Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(self.data))]
        ax2.legend(handles, dist_names)

        ax2.set_xlim(xmin * 0.9, xmax * 1.1)

        ###
        # subplot 3: distribution of differences for mean
        ###
        if len(self.variant_names) > 1:
            m_prior = self.data[control]["m_prior"]
            s_2_prior = self.data[control]["s_2_prior"]
            n_prior = self.data[control]["n_prior"]
            v_prior = self.data[control]["v_prior"]
            sum_squares = self.data[control]["sum_squares"]
            sum_values = self.data[control]["sum_values"]
            n = self.data[control]["total"]

            y_bar = sum_values / n
            mu = (n * y_bar + n_prior * m_prior) / (n + n_prior)
            v_n = v_prior + n
            n_n = n_prior + n
            s_n_2 = (1 / v_n) * (
                sum_squares + s_2_prior * v_prior + (n_prior * n / (n_prior + n)) * (y_bar - m_prior) ** 2
            )
            control_samples = t.rvs(v_n, mu, np.sqrt(s_n_2 / n_n), 200000, random_state=278)

            num_bins = 300
            hist_names = []
            colors = list(mcolors.TABLEAU_COLORS.values())[1:]
            for color, var in zip(colors, [i for i in self.variant_names if i != control]):
                m_prior = self.data[var]["m_prior"]
                s_2_prior = self.data[var]["s_2_prior"]
                n_prior = self.data[var]["n_prior"]
                v_prior = self.data[var]["v_prior"]
                sum_squares = self.data[var]["sum_squares"]
                sum_values = self.data[var]["sum_values"]
                n = self.data[var]["total"]

                y_bar = sum_values / n
                mu = (n * y_bar + n_prior * m_prior) / (n + n_prior)
                v_n = v_prior + n
                n_n = n_prior + n
                s_n_2 = (1 / v_n) * (
                    sum_squares + s_2_prior * v_prior + (n_prior * n / (n_prior + n)) * (y_bar - m_prior) ** 2
                )
                samples = t.rvs(v_n, mu, np.sqrt(s_n_2 / n_n), 200000, random_state=314)

                temp_sample = (samples - control_samples) / self.data[control]["mean"] * 100
                temp_mu = (self.data[var]["mean"] - self.data[control]["mean"]) / self.data[control]["mean"]

                label = f"{var}: " + r"$\mu=" + f"{temp_mu:.2%}$%"
                hist_names.append(label)
                n, bins, patches = ax3.hist(temp_sample, num_bins, label=label, alpha=0.65)

                for b, p in zip(bins, patches):
                    if b <= 0:
                        p.set_facecolor("r")
                    else:
                        p.set_facecolor(color)

                ax3.xaxis.set_major_formatter(mtick.PercentFormatter())
                ax3.set_xlabel(f"Relative uplift vs. {control}" + r"(using marginal distirbutions for $\mu$)")

            handles = [Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(hist_names))]
            ax3.legend(handles, hist_names)

        ###

        fig.tight_layout()

        if fname:
            plt.savefig(fname, dpi=dpi)

        plt.show()

        return fig
