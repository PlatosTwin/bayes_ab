from numbers import Number
from typing import List, Tuple
import warnings

from bayesian_testing.experiments.base import BaseDataTest
from bayesian_testing.metrics import eval_bernoulli_agg, print_bernoulli_evaluation, print_closed_form_comparison
from bayesian_testing.metrics import eval_closed_form_bernoulli_two, eval_closed_form_bernoulli_three
from bayesian_testing.utilities import get_logger

logger = get_logger("bayesian_testing")


class BinaryDataTest(BaseDataTest):
    """
    Class for Bayesian A/B test for binary-like data (conversions, successes, ...).

    After class initialization, use add_variant methods to insert variant data.
    Then to get results of the test, use for instance `evaluate` method.
    """

    def __init__(self) -> None:
        """
        Initialize BinaryDataTest class.
        """
        super().__init__()

    @property
    def totals(self):
        return [self.data[k]["totals"] for k in self.data]

    @property
    def positives(self):
        return [self.data[k]["positives"] for k in self.data]

    @property
    def a_priors(self):
        return [self.data[k]["a_prior"] for k in self.data]

    @property
    def b_priors(self):
        return [self.data[k]["b_prior"] for k in self.data]

    def _eval_simulation(self, sim_count: int = 200000, seed: int = None) -> Tuple[dict, dict]:
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
        pbbs, loss, samples = eval_bernoulli_agg(
            self.totals, self.positives, self.a_priors, self.b_priors, sim_count, seed
        )
        res_pbbs = dict(zip(self.variant_names, pbbs))
        res_loss = dict(zip(self.variant_names, loss))

        for i, var in enumerate(self.variant_names):
            self.data[var]['samples'] = samples[i]

        return res_pbbs, res_loss

    def _closed_form_bernoulli(self) -> dict:
        """
        Calculate the probability to beat all via a closed-form solution.
        Implemented for up to three variants only; will generate a warning if run for test with many observations.
        For tests with many observations, the user may choose to implement an asymptotic forumla, as described by
        Chris Stucchio here: https://www.chrisstucchio.com/blog/2014/bayesian_asymptotics.html.

        Credit: Closed-form chance-to-beat solutions (for two and three variants) are due to
        Evan Miller (https://www.evanmiller.org/bayesian-ab-testing.html), and closed-form expected loss solution (for
        two variants; not implemented currently, but may be implemented in a future release) is due to
        Chris Stucchio (https://www.chrisstucchio.com/blog/2014/bayesian_ab_decision_rule.html).

        Returns
        -------
        pbbs : Dictionary with probabilities of being best for all variants in experiment.
        """
        if sum(self.totals) >= 45000:
            msg = f"The closed-form solution for {sum(self.totals):,} observations it too computationally intensive."
            logger.error(msg)
            raise ValueError(msg)

        if sum(self.totals) >= 35000:
            msg = f"The closed-form solution for {sum(self.totals):,} observations may consume significant resources."
            logger.warn(msg)
            warnings.warn(msg)

        if len(self.totals) > 3:
            msg = f"The closed-form solution is not implemented for more than three variants."
            logger.error(msg)
            raise NotImplementedError(msg)

        for d in self.data.values():
            if int(d['a_prior']) != d['a_prior'] or int(d['b_prior']) != d['b_prior']:
                msg = f"The closed-form solution requires integer values of a, b for all beta(a, b) priors."
                logger.error(msg)
                raise ValueError(msg)

        pbbs = []
        if len(self.totals) == 2:
            a = self.data[self.variant_names[0]]
            b = self.data[self.variant_names[1]]
            pbbs.append(eval_closed_form_bernoulli_two(b, a))  # chance of A to beat B
            pbbs.append(eval_closed_form_bernoulli_two(a, b))  # chance of B to beat A

        elif len(self.totals) == 3:
            a = self.data[self.variant_names[0]]
            b = self.data[self.variant_names[1]]
            c = self.data[self.variant_names[2]]

            # a beats all
            b_beats_a = eval_closed_form_bernoulli_two(a, b)  # chance of B to beat A
            c_beats_a = eval_closed_form_bernoulli_two(a, c)  # chance of C to beat A
            correction = eval_closed_form_bernoulli_three(b, c, a)
            pbbs.append(1 - b_beats_a - c_beats_a + correction)  # chance of A to beat all

            # b beats all
            a_beats_b = eval_closed_form_bernoulli_two(b, a)  # chance of A to beat B
            c_beats_b = eval_closed_form_bernoulli_two(b, c)  # chance of C to beat B
            correction = eval_closed_form_bernoulli_three(c, a, b)
            pbbs.append(1 - a_beats_b - c_beats_b + correction)  # chance of B to beat all

            # c beats all
            a_beats_c = eval_closed_form_bernoulli_two(c, a)  # chance of A to beat C
            b_beats_c = eval_closed_form_bernoulli_two(c, b)  # chance of B to beat C
            correction = eval_closed_form_bernoulli_three(a, b, c)
            pbbs.append(1 - a_beats_c - b_beats_c + correction)  # chance of A to beat all

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
            "positives",
            "positive_rate",
            "prob_being_best",
            "expected_loss",
            "uplift_vs_a"
        ]

        eval_pbbs, eval_loss = self._eval_simulation(sim_count, seed)
        pbbs = list(eval_pbbs.values())
        loss = list(eval_loss.values())

        if closed_form and verbose:
            cf_pbbs = list(self._closed_form_bernoulli().values())
            print_closed_form_comparison(self.variant_names, pbbs, cf_pbbs)

        positive_rate = [round(i[0] / i[1], 5) for i in zip(self.positives, self.totals)]
        uplift = [0]
        for i in positive_rate[1:]:
            uplift.append(round((i - positive_rate[0]) / positive_rate[0], 5))

        data = [self.variant_names, self.totals, self.positives, positive_rate, pbbs, loss, uplift]
        res = [dict(zip(keys, item)) for item in zip(*data)]

        if verbose:
            print_bernoulli_evaluation(res)

        return res

    def add_variant_data_agg(
            self,
            name: str,
            totals: int,
            positives: int,
            a_prior: Number = 1,
            b_prior: Number = 1,
            replace: bool = True,
    ) -> None:
        """
        Add variant data to test class using aggregated binary data.
        This can be convenient as aggregation can be done on database level.

        Default prior is Beta(1, 1), which is the Bayes-Laplace non-informative prior. Other
        common non-informative priors are the Jeffreys beta(1/2, 1/2) and Haldane beta(0, 0). While the selection
        of an appropriate prior is not always straightforward, the effect of selecting the Bayes-Laplace over
        either the Jeffreys or the Haldane priors will be minimal for any reasonably large number of observations.

        For one comparison of these three priors, the user is advised to consult,
        "Posterior Predictive Arguments in Favor of the Bayes-Laplace Prior
        as the Consensus Prior for Binomial and Multinomial Parameters" (https://doi.org/10.1214/09-BA405).

        Other resources include:
            - "Noninformative Bayesian Priors Interpretation And Problems With Construction And Applications"
              (http://www.stats.org.uk/priors/noninformative/Syversveen1998.pdf)
            - "A Catalogue of Non-informative Priors"
              (http://www.stats.org.uk/priors/noninformative/YangBerger1998.pdf)
            - "The Selection of Prior Distributions by Formal Rules"
              (https://www.stat.cmu.edu/~kass/papers/rules.pdf)

        Parameters
        ----------
        name : Variant name.
        totals : Total number of experiment observations (e.g. number of sessions).
        positives : Total number of ones for a given variant (e.g. number of conversions).
        a_prior : Prior alpha parameter for Beta distributions.
            Default value 1 is based on the Bayes-Laplace non-informative prior Beta(1, 1).
        b_prior : Prior beta parameter for Beta distributions.
            Default value 1 is based on the Bayes-Laplace non-informative prior Beta(1, 1).
        replace : Replace data if variant already exists.
            If set to False, data of existing variant will be appended to existing data.
        """
        if not isinstance(name, str):
            raise ValueError("Variant name has to be a string.")
        if a_prior <= 0 or b_prior <= 0:
            raise ValueError("Both [a_prior, b_prior] have to be positive numbers.")
        if totals <= 0:
            raise ValueError("Input variable 'totals' is expected to be positive integer.")
        if positives < 0:
            raise ValueError("Input variable 'positives' is expected to be non-negative integer.")
        if totals < positives:
            raise ValueError("Not possible to have more positives that totals!")

        if name not in self.variant_names:
            self.data[name] = {
                "totals": totals,
                "positives": positives,
                "a_prior": a_prior,
                "b_prior": b_prior,
            }
        elif name in self.variant_names and replace:
            msg = (
                f"Variant {name} already exists - new data is replacing it. "
                "If you wish to append instead, use replace=False."
            )
            logger.info(msg)
            self.data[name] = {
                "totals": totals,
                "positives": positives,
                "a_prior": a_prior,
                "b_prior": b_prior,
            }
        elif name in self.variant_names and not replace:
            msg = (
                f"Variant {name} already exists - new data is appended to variant, "
                "keeping its original prior setup. "
                "If you wish to replace data instead, use replace=True."
            )
            logger.info(msg)
            self.data[name]["totals"] += totals
            self.data[name]["positives"] += positives

    def add_variant_data(
            self,
            name: str,
            data: List[int],
            a_prior: Number = 1,
            b_prior: Number = 1,
            replace: bool = True,
    ) -> None:
        """
        Add variant data to test class using raw binary data.

        Default prior is Beta(1, 1), which is the Bayes-Laplace non-informative prior. Other
        common non-informative priors are the Jeffreys beta(1/2, 1/2) and Haldane beta(0, 0). While the selection
        of an appropriate prior is not always straightforward, the effect of selecting the Bayes-Laplace over
        either the Jeffreys or the Haldane priors will be minimal for any reasonably large number of observations.

        For one comparison of these three priors, the user is advised to consult,
        "Posterior Predictive Arguments in Favor of the Bayes-Laplace Prior
        as the Consensus Prior for Binomial and Multinomial Parameters" (https://doi.org/10.1214/09-BA405).

        Other resources include:
            - "Noninformative Bayesian Priors Interpretation And Problems With Construction And Applications"
              (http://www.stats.org.uk/priors/noninformative/Syversveen1998.pdf)
            - "A Catalogue of Non-informative Priors"
              (http://www.stats.org.uk/priors/noninformative/YangBerger1998.pdf)
            - "The Selection of Prior Distributions by Formal Rules"
              (https://www.stat.cmu.edu/~kass/papers/rules.pdf)

        Parameters
        ----------
        name : Variant name.
        data : List of binary data containing zeros (non-conversion) and ones (conversions).
        a_prior : Prior alpha parameter for Beta distributions.
            Default value 1 is based on the Bayes-Laplace non-informative prior Beta(1, 1).
        b_prior : Prior beta parameter for Beta distributions.
            Default value 1 is based on the Bayes-Laplace non-informative prior Beta(1, 1).
        replace : Replace data if variant already exists.
            If set to False, data of existing variant will be appended to existing data.
        """
        if len(data) == 0:
            raise ValueError("Data of added variant needs to have some observations.")
        if not min([i in [0, 1] for i in data]):
            raise ValueError("Input data needs to be a list of zeros and ones.")

        totals = len(data)
        positives = sum(data)

        self.add_variant_data_agg(name, totals, positives, a_prior, b_prior, replace)
