from numbers import Number
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import numpy as np

from bayes_ab.experiments.base import BaseDataTest
from bayes_ab.metrics import eval_numerical_dirichlet_agg
from bayes_ab.utilities import get_logger, print_dirichlet_evaluation

logger = get_logger("bayes_ab")


class DiscreteDataTest(BaseDataTest):
    """
    Class for Bayesian A/B test for data with finite discrete states (i.e. categorical data
    with numerical categories). As a real world examples we can think of dice rolls,
    1-5 star ratings, 1-10 ratings, etc.

    After class initialization, use add_variant methods to insert variant data.
    Then to get results of the test, use for instance `evaluate` method.
    """

    def __init__(self, states: List[Union[float, int]]) -> None:
        """
        Initialize DiscreteDataTest class.

        Parameters
        ----------
        states : List of all possible states for a given discrete variable.
        """
        super().__init__()
        if not self.check_if_numerical(states):
            raise ValueError("States in the test have to be numbers (int or float).")
        self.states = states

    @property
    def concentrations(self):
        return [self.data[k]["concentration"] for k in self.data]

    @property
    def prior_alphas(self):
        return [self.data[k]["prior"] for k in self.data]

    @staticmethod
    def check_if_numerical(values):
        res = True
        for v in values:
            if not isinstance(v, Number):
                res = False
        return res

    @property
    def means(self):
        return [self.data[k]["mean"] for k in self.data]

    @property
    def rel_probs(self):
        return [self.data[k]["rel_probs"] for k in self.data]

    @property
    def bounds(self):
        try:
            return [self.data[k]["bounds"] for k in self.data]
        except KeyError:
            msg = "You must run the evaluate method before attempting to access this property."
            raise RuntimeError(msg)

    @property
    def rel_bounds(self):
        try:
            return [self.data[k]["rel_bounds"] for k in self.data]
        except KeyError:
            msg = "You must run the evaluate method before attempting to access this property."
            raise RuntimeError(msg)

    @property
    def chance_to_beat(self):
        try:
            return [self.data[k]["chance_to_beat"] for k in self.data]
        except KeyError:
            msg = "You must run the evaluate method before attempting to access this property."
            raise RuntimeError(msg)

    @property
    def exp_loss(self):
        try:
            return [self.data[k]["exp_loss"] for k in self.data]
        except KeyError:
            msg = "You must run the evaluate method before attempting to access this property."
            raise RuntimeError(msg)

    @property
    def uplift_vs_a(self):
        try:
            return [self.data[k]["uplift_vs_a"] for k in self.data]
        except KeyError:
            msg = "You must run the evaluate method before attempting to access this property."
            raise RuntimeError(msg)

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
        pbbs, loss, samples = eval_numerical_dirichlet_agg(
            self.states, self.concentrations, self.prior_alphas, sim_count, seed
        )
        res_pbbs = dict(zip(self.variant_names, pbbs))
        res_loss = dict(zip(self.variant_names, loss))

        for i, var in enumerate(self.variant_names):
            self.data[var]["samples"] = samples[i]

        return res_pbbs, res_loss

    def evaluate(self, verbose: bool = True, sim_count: int = 20000, seed: int = None) -> List[dict]:
        """
        Evaluation of experiment.

        Parameters
        ----------
        verbose : If True, output prints to console.
        sim_count : Number of simulations to be used for probability estimation.
        seed : Random seed.

        Returns
        -------
        res : List of dictionaries with results per variant.
        """
        keys = [
            "variant",
            "concentration",
            "sample_mean",
            "rel_probs",
            "posterior_mean",
            "bounds",
            "rel_bounds",
            "uplift_vs_a",
            "prob_being_best",
            "expected_loss",
            "bounds",
        ]

        eval_pbbs, eval_loss = self._eval_simulation(sim_count, seed)
        pbbs = list(eval_pbbs.values())
        loss = list(eval_loss.values())
        average_values = [np.sum(np.multiply(i, self.states)) / np.sum(i) for i in self.concentrations]

        uplift = [0]
        for i in self.means[1:]:
            uplift.append(round((i - self.means[0]) / self.means[0], 5))

        for i, var in enumerate(self.variant_names):
            self.data[var]["chance_to_beat"] = pbbs[i]
            self.data[var]["exp_loss"] = loss[i]
            self.data[var]["uplift_vs_a"] = uplift[i]

            means = np.sum(np.multiply(self.data[var]["samples"], np.array(self.states)), axis=1)
            bounds = np.quantile(means, (0.025, 0.975))
            self.data[var]["bounds"] = [round(bounds[0], 5), round(bounds[1], 5)]
            rel_bounds = np.quantile(self.data[var]["samples"], (0.025, 0.975), axis=0)
            self.data[var]["rel_bounds"] = [[round(b[0], 5), round(b[1], 5)] for b in rel_bounds.T]

        data = [
            self.variant_names,
            [dict(zip(self.states, i)) for i in self.concentrations],
            average_values,
            self.rel_probs,
            self.means,
            self.bounds,
            self.rel_bounds,
            self.uplift_vs_a,
            pbbs,
            loss,
        ]
        res = [dict(zip(keys, item)) for item in zip(*data)]

        if verbose:
            print_dirichlet_evaluation(res, self.states)

        return res

    def add_variant_data_agg(
        self,
        name: str,
        concentration: List[int],
        prior: List[Union[float, int]] = None,
        replace: bool = True,
    ) -> None:
        """
        Add variant data to test class using already aggregated discrete data.

        The default prior is Dirichlet(1,...,1), a non-informative prior which sets alpha = 1 for each state.

        Parameters
        ----------
        name : Variant name.
        concentration : Total number of experiment observations for each state
            (e.g. number of rolls for each side in a die roll).
        prior : Prior alpha parameters for the Dirichlet distribution, one for each state.
        replace : Replace data if variant already exists.
            If set to False, data of existing variant will be appended to existing data.
        """
        if not isinstance(name, str):
            raise ValueError("Variant name has to be a string.")
        if not len(self.states) == len(concentration):
            msg = (
                f"Concentration list has to have same size as number of states in a test "
                f"{len(concentration)} != {len(self.states)}."
            )
            raise ValueError(msg)
        if not self.check_if_numerical(concentration):
            raise ValueError("Concentration parameter has to be a list of integer values.")

        if not prior:
            prior = [1] * len(self.states)

        if name not in self.variant_names:
            a_0 = sum(prior + concentration)
            self.data[name] = {
                "concentration": concentration,
                "prior": prior,
                "mean": round(sum([x[-1] * sum(x[:-1]) / a_0 for x in zip(prior, concentration, self.states)]), 5),
                "rel_probs": [round(sum(x) / a_0, 5) for x in zip(prior, concentration)],
            }
        elif name in self.variant_names and replace:
            msg = (
                f"Variant {name} already exists - new data is replacing it. "
                "If you wish to append instead, use replace=False."
            )
            logger.info(msg)
            a_0 = sum([sum(x) for x in zip(prior, concentration)])
            self.data[name] = {
                "concentration": concentration,
                "prior": prior,
                "mean": round(sum([x[-1] * sum(x[:-1]) / a_0 for x in zip(prior, concentration, self.states)]), 5),
                "rel_probs": [round(sum(x) / a_0, 5) for x in zip(prior, concentration)],
            }
        elif name in self.variant_names and not replace:
            msg = (
                f"Variant {name} already exists - new data is appended to variant, "
                "keeping its original prior setup. "
                "If you wish to replace data instead, use replace=True."
            )
            logger.info(msg)
            concentration_updated = [sum(x) for x in zip(self.data[name]["concentration"], concentration)]
            self.data[name]["concentration"] = concentration_updated

            a_0 = sum([sum(x) for x in zip(prior, concentration_updated)])
            self.data[name]["mean"] = round(
                sum([x[-1] * sum(x[:-1]) / a_0 for x in zip(prior, concentration_updated, self.states)]), 5
            )
            self.data[name]["rel_probs"] = [round(sum(x) / a_0, 5) for x in zip(prior, concentration_updated)]

    def add_variant_data(
        self,
        name: str,
        data: List[int],
        prior: List[Union[float, int]] = None,
        replace: bool = True,
    ) -> None:
        """
        Add variant data to test class using raw discrete data.

        Default prior setup is Dirichlet(1,...,1) which is low information prior
        (we can interpret it as prior 1 observation of each state).

        Parameters
        ----------
        name : Variant name.
        data : List of numerical data observations from possible states.
        prior : Prior alpha parameters of Dirichlet distribution.
        replace : Replace data if variant already exists.
            If set to False, data of existing variant will be appended to existing data.
        """
        if len(data) == 0:
            raise ValueError("Data of added variant needs to have some observations.")
        if not min([i in self.states for i in data]):
            msg = "Input data needs to be a list of numbers from possible states: " f"{self.states}."
            raise ValueError(msg)

        counter_dict = dict(zip(self.states, np.zeros(len(self.states))))
        for i in data:
            counter_dict[i] += 1
        concentration = [counter_dict[i] for i in self.states]

        self.add_variant_data_agg(name, concentration, prior, replace)

    def plot_distributions(self, fname: str = None, dpi: int = 300) -> plt.figure:
        """
        For each variant, plot the posterior distribution for each state.

        For 10 states and fewer, state colors will be distinct; for more than 10 states, colors will duplicate.

        Parameters
        ----------
        fname : Filename to which to save the resultant image; if None, the image is not saved.
        dpi : DPI setting for saved image; used only when fname is not None.
        """
        fig = plt.figure(figsize=(10, 8))
        colors = list(mcolors.TABLEAU_COLORS.values())
        for i, var in enumerate(self.data):
            dist_names = []
            ax = fig.add_subplot(len(self.variant_names), 1, i + 1)

            for j, (state, color) in enumerate(zip(self.data[var]["samples"].T, colors)):
                mu = state.mean()
                label = f"{var} | {self.states[j]}: " + r"$\mu=" + f"{mu:.2%}$%"
                dist_names.append(label)

                ax.hist(state * 100, label=label, color=color, alpha=0.5, bins=200)

            handles = [Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(self.states))]
            ax.legend(handles, dist_names)
            ax.set_xlabel("Posterior probability")
            ax.set_xlim(0, 100)
            ax.set_title(f'Variant "{var}"')
            ax.xaxis.set_major_formatter(mtick.PercentFormatter())

        fig.suptitle("Posterior probability for each state, by variant", fontsize=15)
        fig.tight_layout()

        if fname:
            plt.savefig(fname, dpi=dpi)

        plt.show()

        return fig
