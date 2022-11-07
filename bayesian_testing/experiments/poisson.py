from numbers import Number
from typing import List, Tuple
import numpy as np
import warnings

from bayesian_testing.experiments.base import BaseDataTest
from bayesian_testing.metrics import eval_poisson_agg, eval_closed_form_poisson_two, eval_closed_form_poisson_three
from bayesian_testing.metrics import print_poisson_evaluation, print_closed_form_comparison
from bayesian_testing.utilities import get_logger

logger = get_logger("bayesian_testing")


class PoissonDataTest(BaseDataTest):
    """
    Class for Bayesian A/B test for count data.

    After class initialization, use add_variant methods to insert variant data.
    Then to get results of the test, use for instance `evaluate` method.
    """

    def __init__(self) -> None:
        """
        Initialize PoissonDataTest class.
        """
        super().__init__()

    @property
    def totals(self):
        return [self.data[k]["totals"] for k in self.data]

    @property
    def mean(self):
        return [self.data[k]["mean"] for k in self.data]

    @property
    def sums(self):
        return [self.data[k]["sum"] for k in self.data]

    @property
    def a_priors(self):
        return [self.data[k]["a_prior"] for k in self.data]

    @property
    def b_priors(self):
        return [self.data[k]["b_prior"] for k in self.data]

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
        pbbs, loss, samples = eval_poisson_agg(
            self.totals,
            self.mean,
            self.a_priors,
            self.b_priors,
            sim_count=sim_count,
            seed=seed
        )
        res_pbbs = dict(zip(self.variant_names, pbbs))
        res_loss = dict(zip(self.variant_names, loss))

        for i, var in enumerate(self.variant_names):
            self.data[var]['samples'] = samples[i]

        return res_pbbs, res_loss

    def _closed_form_poisson(self) -> dict:
        """
        Calculate the probability to beat all via a closed-form solution.
        Implemented for up to three variants only; will generate a warning if run for test with many observations.

        Credit: Closed-form chance-to-beat solutions (for two and three variants) are due to
        Evan Miller (https://www.evanmiller.org/bayesian-ab-testing.html).

        Returns
        -------
        pbbs : Dictionary with probabilities of being best for all variants in experiment.
        """
        if len(self.totals) > 3:
            msg = f"The closed-form solution is not implemented for more than three variants."
            logger.error(msg)
            raise NotImplementedError(msg)

        if sum(self.sums) >= 5000:
            msg = f"The closed-form solution for {sum(self.sums):,} observations it too computationally intensive."
            logger.error(msg)
            raise ValueError(msg)

        if sum(self.sums) >= 3000:
            msg = f"The closed-form solution for {sum(self.sums):,} observations may consume significant resources."
            logger.warn(msg)
            warnings.warn(msg)

        for d in self.data.values():
            if int(d['a_prior']) != d['a_prior'] or int(d['b_prior']) != d['b_prior']:
                msg = f"The closed-form solution requires integer values of a, b for all beta(a, b) priors."
                logger.error(msg)
                raise ValueError(msg)

        pbbs = []
        if len(self.totals) == 2:
            a = self.data[self.variant_names[0]]['sum']
            b = self.data[self.variant_names[1]]['sum']

            pbbs.append(eval_closed_form_poisson_two(a, b))  # chance of A to beat B
            pbbs.append(eval_closed_form_poisson_two(b, a))  # chance of B to beat A

        elif len(self.totals) == 3:
            a = self.data[self.variant_names[0]]['sum']
            b = self.data[self.variant_names[1]]['sum']
            c = self.data[self.variant_names[2]]['sum']

            # A beats all
            b_beats_a = eval_closed_form_poisson_two(b, a)  # chance of B to beat A
            c_beats_a = eval_closed_form_poisson_two(c, a)  # chance of C to beat A
            correction = eval_closed_form_poisson_three(a, b, c)
            pbbs.append(1 - b_beats_a - c_beats_a + correction)  # chance of A to beat all

            # B beats all
            a_beats_b = eval_closed_form_poisson_two(a, b)  # chance of A to beat B
            c_beats_b = eval_closed_form_poisson_two(c, b)  # chance of C to beat B
            correction = eval_closed_form_poisson_three(b, c, a)
            pbbs.append(1 - a_beats_b - c_beats_b + correction)  # chance of B to beat all

            # C beats all
            a_beats_c = eval_closed_form_poisson_two(a, c)  # chance of A to beat C
            b_beats_c = eval_closed_form_poisson_two(b, c)  # chance of B to beat C
            correction = eval_closed_form_poisson_three(c, a, b)
            pbbs.append(1 - a_beats_c - b_beats_c + correction)  # chance of C to beat all

        return dict(zip(self.variant_names, pbbs))

    def evaluate(
            self,
            closed_form: bool = False,
            sim_count: int = 200000,
            seed: int = None,
            verbose: bool = True
    ) -> List[dict]:
        """
        Evaluation of experiment.

        Parameters
        ----------
        closed_form : If True, compare the results of MC simulation to the closed-form result.
        sim_count : Number of simulations to be used for probability estimation.
        seed : Random seed.
        verbose : If True, output prints to console.

        Returns
        -------
        res : List of dictionaries with results per variant.
        """
        keys = [
            "variant",
            "totals",
            "mean",
            "prob_being_best",
            "expected_loss"
        ]

        eval_pbbs, eval_loss = self._eval_simulation(sim_count, seed)
        pbbs = list(eval_pbbs.values())
        loss = list(eval_loss.values())

        if closed_form and verbose:
            cf_pbbs = list(self._closed_form_poisson().values())
            print_closed_form_comparison(self.variant_names, pbbs, cf_pbbs)

        data = [
            self.variant_names,
            self.totals,
            self.mean,
            pbbs,
            loss,
        ]
        res = [dict(zip(keys, item)) for item in zip(*data)]

        if verbose:
            print_poisson_evaluation(res)

        return res

    def add_variant_data_agg(
            self,
            name: str,
            totals: int,
            mean: Number,
            obs_sum: Number,
            a_prior: Number = 1,
            b_prior: Number = 1,
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
        totals : Total number of experiment observations.
        mean : Mean value of observations.
        obs_sum : Sum of counts from all observations.
        a_prior : Prior alpha parameter for Gamma distributions. Default value is 1.
        b_prior : Prior beta parameter for Gamma distributions. Default value is 1.
        replace : Replace data if variant already exists.
            If set to False, data of existing variant will be appended to existing data.
        """
        if not isinstance(name, str):
            raise ValueError("Variant name has to be a string.")
        if a_prior <= 0 or b_prior <= 0:
            raise ValueError("Both [a_prior, b_prior] have to be positive numbers.")
        if totals <= 0:
            raise ValueError("Input variable 'totals' is expected to be positive integer.")

        if name not in self.variant_names:
            self.data[name] = {
                "totals": totals,
                "mean": mean,
                "sum": obs_sum,
                "a_prior": a_prior,
                "b_prior": b_prior
            }
        elif name in self.variant_names and replace:
            msg = (
                f"Variant {name} already exists - new data is replacing it. "
                "If you wish to append instead, use replace=False."
            )
            logger.info(msg)
            self.data[name] = {
                "totals": totals,
                "mean": mean,
                "sum": obs_sum,
                "a_prior": a_prior,
                "b_prior": b_prior
            }
        elif name in self.variant_names and not replace:
            msg = (
                f"Variant {name} already exists - new data is appended to variant,"
                "keeping its original prior setup. "
                "If you wish to replace data instead, use replace=True."
            )
            logger.info(msg)
            self.data[name]["mean"] = (self.data[name]["mean"] * self.data[name]["totals"] + mean * totals) / \
                                      (totals + self.data[name]["totals"])
            self.data[name]["totals"] += totals
            self.data[name]["sum"] += obs_sum

    def add_variant_data(
            self,
            name: str,
            data: List[Number],
            a_prior: Number = 1,
            b_prior: Number = 1,
            replace: bool = True,
    ) -> None:
        """
        Add variant data to test class using raw normal data.

        The goal of default prior setup is to be low information. It should be tuned with caution.

        Parameters
        ----------
        name : Variant name.
        data : List of count data, each data point corresponding to one observation.
        a_prior : Prior alpha parameter for Gamma distributions. Default value is 1.
        b_prior : Prior beta parameter for Gamma distributions. Default value is 1.
        replace : Replace data if variant already exists.
            If set to False, data of existing variant will be appended to existing data.
        """
        if len(data) == 0:
            raise ValueError("Data of added variant needs to have some observations.")

        totals = len(data)
        mean = np.mean(data)
        obs_sum = np.sum(data)

        self.add_variant_data_agg(
            name,
            totals,
            mean,
            obs_sum,
            a_prior,
            b_prior,
            replace,
        )
