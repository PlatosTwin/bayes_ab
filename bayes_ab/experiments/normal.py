from numbers import Number
from typing import List, Tuple

import numpy as np

from bayes_ab.experiments.base import BaseDataTest
from bayes_ab.metrics import eval_normal_agg
from bayes_ab.utilities import get_logger

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
    def sum_values_2(self):
        return [self.data[k]["sum_values_2"] for k in self.data]

    @property
    def m_priors(self):
        return [self.data[k]["m_prior"] for k in self.data]

    @property
    def a_priors_ig(self):
        return [self.data[k]["a_prior_ig"] for k in self.data]

    @property
    def b_priors_ig(self):
        return [self.data[k]["b_prior_ig"] for k in self.data]

    @property
    def w_priors(self):
        return [self.data[k]["w_prior"] for k in self.data]

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
            self.sum_values_2,
            sim_count=sim_count,
            m_priors=self.m_priors,
            a_priors_ig=self.a_priors_ig,
            b_priors_ig=self.b_priors_ig,
            w_priors=self.w_priors,
            seed=seed,
        )
        res_pbbs = dict(zip(self.variant_names, pbbs))
        res_loss = dict(zip(self.variant_names, loss))

        for i, var in enumerate(self.variant_names):
            self.data[var]["samples"] = samples[i]

        return res_pbbs, res_loss

    def evaluate(self, sim_count: int = 20000, seed: int = None) -> List[dict]:
        """
        Evaluation of experiment.

        Parameters
        ----------
        sim_count : Number of simulations to be used for probability estimation.
        seed : Random seed.

        Returns
        -------
        res : List of dictionaries with results per variant.
        """
        keys = [
            "variant",
            "total",
            "sum_values",
            "avg_values",
            "prob_being_best",
            "expected_loss",
        ]
        avg_values = [round(i[0] / i[1], 5) for i in zip(self.sum_values, self.totals)]
        eval_pbbs, eval_loss = self._eval_simulation(sim_count, seed)
        pbbs = list(eval_pbbs.values())
        loss = list(eval_loss.values())
        data = [
            self.variant_names,
            self.totals,
            [round(i, 5) for i in self.sum_values],
            avg_values,
            pbbs,
            loss,
        ]
        res = [dict(zip(keys, item)) for item in zip(*data)]

        return res

    def add_variant_data_agg(
        self,
        name: str,
        total: int,
        sum_values: float,
        sum_values_2: float,
        m_prior: Number = 1,
        a_prior_ig: Number = 0,
        b_prior_ig: Number = 0,
        w_prior: Number = 0.01,
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
        sum_values_2 : Sum of values squared for a given variant.
        m_prior : Prior normal mean.
        a_prior_ig : Prior alpha from inverse gamma dist. for unknown variance.
            In theory a > 0, but as we always have at least one observation, we can start at 0.
        b_prior_ig : Prior beta from inverse gamma dist. for unknown variance.
            In theory b > 0, but as we always have at least one observation, we can start at 0.
        w_prior : Prior effective sample sizes.
        replace : Replace data if variant already exists.
            If set to False, data of existing variant will be appended to existing data.
        """
        if not isinstance(name, str):
            raise ValueError("Variant name has to be a string.")
        if m_prior < 0 or a_prior_ig < 0 or b_prior_ig < 0 or w_prior < 0:
            raise ValueError("All priors of [m, a_ig, b_ig, w] have to be non-negative numbers.")
        if total <= 0:
            raise ValueError("Input variable 'total' is expected to be positive integer.")

        if name not in self.variant_names:
            self.data[name] = {
                "total": total,
                "sum_values": sum_values,
                "sum_values_2": sum_values_2,
                "m_prior": m_prior,
                "a_prior_ig": a_prior_ig,
                "b_prior_ig": b_prior_ig,
                "w_prior": w_prior,
            }
        elif name in self.variant_names and replace:
            msg = (
                f"Variant {name} already exists - new data is replacing it. "
                "If you wish to append instead, use replace=False."
            )
            logger.info(msg)
            self.data[name] = {
                "total": total,
                "sum_values": sum_values,
                "sum_values_2": sum_values_2,
                "m_prior": m_prior,
                "a_prior_ig": a_prior_ig,
                "b_prior_ig": b_prior_ig,
                "w_prior": w_prior,
            }
        elif name in self.variant_names and not replace:
            msg = (
                f"Variant {name} already exists - new data is appended to variant, "
                "keeping its original prior setup. "
                "If you wish to replace data instead, use replace=True."
            )
            logger.info(msg)
            self.data[name]["total"] += total
            self.data[name]["sum_values"] += sum_values
            self.data[name]["sum_values_2"] += sum_values_2

    def add_variant_data(
        self,
        name: str,
        data: List[Number],
        m_prior: Number = 1,
        a_prior_ig: Number = 0,
        b_prior_ig: Number = 0,
        w_prior: Number = 0.01,
        replace: bool = True,
    ) -> None:
        """
        Add variant data to test class using raw normal data.

        The goal of default prior setup is to be low information. It should be tuned with caution.

        Parameters
        ----------
        name : Variant name.
        data : List of normal data.
        m_prior : Prior mean.
        a_prior_ig : Prior alpha from inverse gamma dist. for unknown variance.
            In theory a > 0, but as we always have at least one observation, we can start at 0.
        b_prior_ig : Prior beta from inverse gamma dist. for unknown variance.
            In theory b > 0, but as we always have at least one observation, we can start at 0.
        w_prior : Prior effective sample sizes.
        replace : Replace data if variant already exists.
            If set to False, data of existing variant will be appended to existing data.
        """
        if len(data) == 0:
            raise ValueError("Data of added variant needs to have some observations.")

        total = len(data)
        sum_values = sum(data)
        sum_values_2 = sum(np.square(data))

        self.add_variant_data_agg(
            name,
            total,
            sum_values,
            sum_values_2,
            m_prior,
            a_prior_ig,
            b_prior_ig,
            w_prior,
            replace,
        )
