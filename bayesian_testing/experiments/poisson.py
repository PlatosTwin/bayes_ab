from typing import List, Tuple, Union, Dict
import numpy as np
import warnings
import matplotlib.pyplot as plt
import scipy.stats as stats

from bayesian_testing.experiments.base import BaseDataTest
from bayesian_testing.metrics import eval_poisson_agg, eval_closed_form_poisson_two, eval_closed_form_poisson_three
from bayesian_testing.utilities import get_logger, print_poisson_evaluation, print_closed_form_comparison

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
        return [self.data[k]["total"] for k in self.data]

    @property
    def obs_means(self):
        return [self.data[k]["obs_mean"] for k in self.data]

    @property
    def sums(self):
        return [self.data[k]["sum"] for k in self.data]

    @property
    def a_priors(self):
        return [self.data[k]["a_prior"] for k in self.data]

    @property
    def b_priors(self):
        return [self.data[k]["b_prior"] for k in self.data]

    @property
    def means(self):
        return [self.data[k]["mean"] for k in self.data]

    @property
    def stdevs(self):
        return [self.data[k]["stdev"] for k in self.data]

    @property
    def bounds(self):
        return [self.data[k]["bounds"] for k in self.data]

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
            self.obs_means,
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
            msg = f"The closed-form solution for {sum(self.sums):,} total counts it too computationally intensive."
            logger.error(msg)
            raise ValueError(msg)

        if sum(self.sums) >= 3000:
            msg = f"The closed-form solution for {sum(self.sums):,} total counts may consume significant resources."
            logger.warn(msg)
            warnings.warn(msg)

        for d in self.data.values():
            if int(d['a_prior']) != d['a_prior'] or int(d['b_prior']) != d['b_prior']:
                msg = (f"The closed-form solution requires integer values of a, b "
                       f"for all gamma(a, b) and beta(a, b) priors.")
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

    def _decision_rule(
            self,
            control: str,
            rope: float,
            precision: float,
            interval: float,
            verbose: bool
    ) -> Union[Dict, None]:
        """
        This method implements a basic experimentation decision rule, based largely on the decision rules
        outlined by Yanir Seroussi (https://yanirseroussi.com/2016/06/19/making-bayesian-ab-testing-more-accessible/),
        themselves based on decision rules outlined by John K. Kruschke
        (http://doingbayesiandataanalysis.blogspot.com/2013/11/optional-stopping-in-data-collection-p.html). The
        motivation for both authors is outlined by David Robinson (http://varianceexplained.org/r/bayesian-ab-testing/).

        If the width of the high-density interval (HDI) is less than <precision>*<rope>, the decision is made with
        high confidence; otherwise, the decision is made with low confidence.

        If the HDI fully excludes the Region of Practical Equivalence (ROPE), the recommendation is to stop and
        select the better variant. If the HDI partially contains the ROPE, the recommendation is to continue
        gathering data. If the HDI fully contains the ROPE, the decision is to select either variant.

        Parameters
        ----------
        control : Denotes the variant to treat as the control.
        rope : Region of Practical Equivalence. Should be passed in absolute terms: 0.1% = 0.001.
        precision : Controls experiment stopping. HDI is compared to (rope * precision). Defaults to 0.8.
        interval : The percentage width of the HDI. Defaults to 95%. Defaults to 95%. Must be in (0, 1).

        Returns
        -------
        confidence : Whether the recommendation is made with high or low confidence, based on width of bound.
        decision : The recommendation of what to do given the test data.
        lower_bound : The lower bound of the HDI given by <interval>.
        upper_bound : The upper bound of the HDI given by <interval>.
        """
        if not control and not rope:
            return None

        if (control or rope) and len(self.totals) != 2:
            msg = f"Decision assessments are implemented for two-variant models only."
            logger.error(msg)
            raise NotImplementedError(msg)

        if (control and not rope) or (rope and not control):
            msg = f"In order to return an assessment, you need to specify both <control> and <rope>."
            logger.error(msg)
            raise ValueError(msg)

        if len(self.totals) == 2:
            var_names = self.variant_names.copy()
            var_names.remove(control)
            diff_distribution = self.data[var_names[0]]['samples'] - self.data[control]['samples']
            lower_bound = np.percentile(diff_distribution, 100*(1 - interval) / 2)
            upper_bound = np.percentile(diff_distribution, 100*(1 - interval) / 2 + 100*interval)

            if upper_bound - lower_bound < rope * precision:
                confidence = 'High'
            else:
                confidence = 'Low'

            if rope < lower_bound or -rope > upper_bound:
                decision = 'Stop and select better variant.'
            elif -rope > lower_bound and rope < upper_bound:
                decision = 'Stop and implement either variant.'
            else:
                decision = 'Continue collecting data.'

            if verbose:
                print(f'Decision: {decision} Confidence: {confidence}. '
                      f'Bounds: [{lower_bound:.1f}, {upper_bound:.1f}].', '\n')

            assessment = {'decision': decision, 'confidence': confidence,
                          'lower_bound': lower_bound, 'upper_bound': upper_bound}

            return assessment

    def evaluate(
            self,
            closed_form: bool = False,
            sim_count: int = 200000,
            seed: int = None,
            verbose: bool = True,
            control: str = None,
            rope: float = None,
            precision: float = 0.8,
            interval: float = 0.95
    ) -> Tuple[List[dict], Union[Dict, None]]:
        """
        Evaluation of experiment.

        Parameters
        ----------
        closed_form : If True, compare the results of MC simulation to the closed-form result.
        sim_count : Number of simulations to be used for probability estimation.
        seed : Random seed.
        verbose : If True, output prints to console.
        control : Denotes the variant to treat as the control. If not None, used in generating a stopping decision.
        rope : Region of Practical Equivalence. Should be passed in absolute terms: 0.1% = 0.001. Defaults to 0.001.
        precision : Controls experiment stopping. HDI is compared to (rope * precision). Defaults to 0.8.
        interval : The percentage width of the HDI. Defaults to 95%. Must be in (0, 1).

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
            "bounds"
        ]

        eval_pbbs, eval_loss = self._eval_simulation(sim_count, seed)
        pbbs = list(eval_pbbs.values())
        loss = list(eval_loss.values())

        if closed_form and verbose:
            cf_pbbs = list(self._closed_form_poisson().values())
            print_closed_form_comparison(self.variant_names, pbbs, cf_pbbs)

        data = [self.variant_names, self.totals, self.means, pbbs, loss, self.bounds]
        res = [dict(zip(keys, item)) for item in zip(*data)]

        if verbose:
            print_poisson_evaluation(res)

        assessment = self._decision_rule(control, rope, precision, interval, verbose)

        return res, assessment

    def add_variant_data_agg(
            self,
            name: str,
            total: int,
            obs_mean: float,
            obs_sum: int,
            a_prior: float = 1,
            b_prior: float = 1,
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
        total : Total number of experiment observations.
        obs_mean : Mean value of observations.
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
        if total <= 0:
            raise ValueError("Input variable 'total' is expected to be positive integer.")

        if name not in self.variant_names:
            self.data[name] = {
                "total": total,
                "obs_mean": obs_mean,
                "sum": obs_sum,
                "a_prior": a_prior,
                "b_prior": b_prior,
                "mean": round((a_prior + total * obs_mean) / (b_prior + total), 5),
                "stdev": round(np.sqrt((a_prior + total * obs_mean)) / (b_prior + total), 5),
                "bounds": [round(stats.gamma.ppf(1 - 0.95, a=a_prior + total * obs_mean,
                                                 scale=1 / (b_prior + total)), 5),
                           round(stats.gamma.ppf(0.95, a=a_prior + total * obs_mean,
                                                 scale=1 / (b_prior + total)), 5)]
            }
        elif name in self.variant_names and replace:
            msg = (
                f"Variant {name} already exists - new data is replacing it. "
                "If you wish to append instead, use replace=False."
            )
            logger.info(msg)
            self.data[name] = {
                "total": total,
                "obs_mean": obs_mean,
                "sum": obs_sum,
                "a_prior": a_prior,
                "b_prior": b_prior,
                "mean": round((a_prior + total * obs_mean) / (b_prior + total), 5),
                "stdev": round(np.sqrt((a_prior + total * obs_mean)) / (b_prior + total), 5),
                "bounds": [round(stats.gamma.ppf(1 - 0.95, a=a_prior + total * obs_mean,
                                                 scale=1 / (b_prior + total)), 5),
                           round(stats.gamma.ppf(0.95, a=a_prior + total * obs_mean,
                                                 scale=1 / (b_prior + total)), 5)]
            }
        elif name in self.variant_names and not replace:
            msg = (
                f"Variant {name} already exists - new data is appended to variant,"
                "keeping its original prior setup. "
                "If you wish to replace data instead, use replace=True."
            )
            logger.info(msg)
            self.data[name]["obs_mean"] = (self.data[name]["obs_mean"] * self.data[name]["total"] +
                                           obs_mean * total) / \
                                          (total + self.data[name]["total"])
            self.data[name]["total"] += total
            self.data[name]["sum"] += obs_sum

            obs_mean = self.data[name]["obs_mean"]
            total = self.data[name]["total"]
            self.data[name]["mean"] = round((a_prior + total * obs_mean) / (b_prior + total), 5),
            self.data[name]["stdev"] = round(np.sqrt((a_prior + total * obs_mean)) / (b_prior + total), 5)
            self.data[name]["bounds"] = [round(stats.gamma.ppf(1 - 0.95, a=a_prior + total * obs_mean,
                                                               scale=1 / (b_prior + total)), 5),
                                         round(stats.gamma.ppf(0.95, a=a_prior + total * obs_mean,
                                                               scale=1 / (b_prior + total)), 5)]

    def add_variant_data(
            self,
            name: str,
            data: List[int],
            a_prior: float = 1,
            b_prior: float = 1,
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

        total = len(data)
        obs_mean = np.mean(data)
        obs_sum = np.sum(data)

        self.add_variant_data_agg(
            name,
            total,
            obs_mean,
            obs_sum,
            a_prior,
            b_prior,
            replace,
        )

    def plot_posteriors(self, fname: str = None, dpi: int = 300) -> None:
        """
        For each variant, plot its posterior distribution.

        Parameters
        ----------
        fname : Filename to which to save the resultant image; if None, the image is not saved.
        dpi : DPI setting for saved image; used only when fname is not None.
        """
        fig, ax = plt.subplots(figsize=(10, 8), )

        xmin = max(self.means) * 5
        xmax = 0
        for var in self.data:
            a = self.data[var]['a_prior']
            totals = self.data[var]["total"]
            obs_mean = self.data[var]['obs_mean']
            b = self.data[var]['b_prior']
            mu = (a + totals * obs_mean) / (b + totals)

            x = np.linspace(0, max(self.means) * 5, 10000)
            y = stats.gamma.pdf(x, a=a + totals * obs_mean, scale=1 / (b + totals))
            ax.plot(x, y, label=f'{var}: $\mu={mu:.2f}$')
            ax.fill_between(x, y, alpha=0.35)

            if x[np.where(y >= 0.0001)[0][0]] < xmin:
                xmin = x[np.where(y >= 0.0001)[0][0]]
            if x[np.where(y >= 0.0001)[0][-1]] > xmax:
                xmax = x[np.where(y >= 0.0001)[0][-1]]

        ax.set_ylabel('Probability density')
        ax.set_xlabel('Count value')
        ax.legend()

        plt.xlim([xmin * 0.9, xmax * 1.10])

        fig.tight_layout()

        if fname:
            plt.savefig(fname, dpi=dpi)

        plt.show()

    def plot_differences(self, control: str, fname: str = None, dpi: int = 300) -> None:
        """
        For each variant, plot the difference between its posterior and the posterior for <control>.

        Parameters
        ----------
        control : The variant to treat as control; this variant will be subtracted from each other variant.
        fname : Filename to which to save the resultant image; if None, the image is not saved.
        dpi : DPI setting for saved image; used only when fname is not None.
        """
        num_bins = 250
        fig, ax = plt.subplots(figsize=(10, 8), )

        for var in [i for i in self.variant_names if i != control]:
            temp_sample = self.data[var]['samples'] - self.data[control]['samples']
            temp_mu = self.data[var]['mean'] - self.data[control]['mean']

            ax.hist(temp_sample, num_bins, label=f'{var}: $\mu={temp_mu:.2f}$', alpha=0.65)
            ax.set_xlabel('Value')
            ax.set_ylabel('Unnormalized probability density')

        ax.legend()

        plt.title(f'Difference from {control}')
        fig.tight_layout()

        if fname:
            plt.savefig(fname, dpi=dpi)

        plt.show()
