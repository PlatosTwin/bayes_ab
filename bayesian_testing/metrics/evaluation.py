from numbers import Number
from typing import List, Tuple, Union, Dict
import warnings
from prettytable import PrettyTable
import numpy as np
import scipy

from bayesian_testing.metrics.posteriors import (
    beta_posteriors_all,
    lognormal_posteriors,
    normal_posteriors,
    dirichlet_posteriors,
)
from bayesian_testing.utilities import get_logger

logger = get_logger("bayesian_testing")


def estimate_chance_to_beat(data: Union[List[List[Number]], np.ndarray]) -> List[float]:
    """
    For each variant, estimate probability of beating all other variants,
    considering simulated data from respective posteriors.

    Parameters
    ----------
    data : List of simulated data for each variant.

    Returns
    -------
    res : List of probabilities of being best for each variant.
    """
    max_values = np.argmax(data, axis=0)
    unique, counts = np.unique(max_values, return_counts=True)
    occurrences = dict(zip(unique, counts))
    sim_count = len(data[0])
    res = []
    for i in range(len(data)):
        res.append(round(occurrences.get(i, 0) / sim_count, 7))
    return res


def estimate_expected_loss(data: Union[List[List[Number]], np.ndarray]) -> List[float]:
    """
    For each variant, estimate expected loss of selecting , considering simulated data from respective posteriors.

    Parameters
    ----------
    data : List of simulated data for each variant.

    Returns
    -------
    res : List of expected loss for each variant.
    """
    max_values = np.max(data, axis=0)
    res = list(np.mean(max_values - data, axis=1).round(7))

    return res


def print_bernoulli_evaluation(res: list) -> None:
    """
    Pretty-print output of running standard binary test.
    """
    tab = PrettyTable()
    tab.field_names = ['Variant', 'Totals', 'Positives', 'Positive rate',
                       'Chance to beat all', 'Expected loss', 'Uplift vs. "A"']
    for r in res:
        temp_row = r.copy()
        for i in ['positive_rate', 'prob_being_best', 'expected_loss', 'uplift_vs_a']:
            temp_row[i] = f'{temp_row[i]:.2%}'
        temp_row = list(temp_row.values())

        tab.add_row(temp_row)

    tab.reversesort = True
    tab.sortby = 'Chance to beat all'

    print(tab)


def print_closed_form_comparison(variants: list,
                                 pbbs: list,
                                 cf_pbbs: list) -> None:
    """
    Pretty-print output comparing the estimate to the exact chance to beat all.
    """
    tab = PrettyTable()
    tab.field_names = ['Variant', 'Est. chance to beat all', 'Exact chance to beat all', 'Delta']
    for var, est, cf in zip(variants, pbbs, cf_pbbs):
        tab.add_row([var, f'{est:.2%}', f'{cf:.2%}',  f'{(est - cf)/cf:.2%}'])

    tab.reversesort = True
    tab.sortby = 'Est. chance to beat all'

    print(tab)


def validate_bernoulli_input(totals: List[int], positives: List[int]) -> None:
    """
    Simple validation for pbb_bernoulli_agg inputs.
    """
    if len(totals) != len(positives):
        msg = f"Totals ({totals}) and positives ({positives}) needs to have same length!"
        logger.error(msg)
        raise ValueError(msg)


def expected_loss_accuracy_bernoulli(data: Union[List[List[Number]], np.ndarray]) -> None:
    """
    Validate that the estimated expected loss is within <epsilon> of the true expected loss with probability <tau>.
    """
    epsilon = 0.0001
    s_2 = len(data) * np.var(data[0])
    for i in data[1:]:
        s_2 += np.var(i)

    n = data.shape[1]
    tau = scipy.stats.norm.cdf(np.sqrt(n) * epsilon / np.sqrt(s_2))*2 - 1

    if tau < 0.99:
        msg = f"There is at least a 1% probability that the estimated expected loss is not within {epsilon} tolerance."
        logger.warn(msg)
        warnings.warn(msg)


def eval_closed_form_bernoulli_two(a: Dict,
                                   b: Dict) -> float:
    """
    Given two variants A and B, calculate the probability that B will beat A.

    Parameters
    ----------
    a : Dictionary containing summary statistics for variant A; must contain total observations, positives.
    b : Dictionary containing summary statistics for variant B; must contain total observations, positives.

    Returns
    -------
    total : Probability that B will beat A.
    """
    total = 0
    for k in range(b['positives'] + 1):
        total += np.exp(scipy.special.betaln(a['positives'] + 1 + k,
                                             2 + b['totals'] - b['positives'] + a['totals'] - a['positives']) -
                        np.log(b['totals'] - b['positives'] + 1 + k) -
                        scipy.special.betaln(1 + k, b['totals'] - b['positives'] + 1) -
                        scipy.special.betaln(a['positives'] + 1, 1 + a['totals'] - a['positives']))

    return total


def eval_closed_form_bernoulli_three(a: Dict,
                                     b: Dict,
                                     c: Dict) -> float:
    """
    Given three variants A, B, and C, calculate the probability that C will beat both B and A.

    Parameters
    ----------
    a : Dictionary containing summary statistics for variant A; must contain total observations, positives.
    b : Dictionary containing summary statistics for variant B; must contain total observations, positives.
    C : Dictionary containing summary statistics for variant C; must contain total observations, positives.

    Returns
    -------
    total : Probability that C will beat both B and A.
    """
    total = 0.0
    for i in range(a['positives'] + 1):
        for j in range(b['positives'] + 1):
            beta_A = a['totals'] - a['positives'] + 1
            beta_B = b['totals'] - b['positives'] + 1
            beta_C = c['totals'] - c['positives'] + 1

            total += np.exp(scipy.special.betaln(c['positives'] + 1 + i + j, beta_A + beta_B + beta_C)
                            - np.log(beta_A + i) - np.log(beta_B + j)
                            - scipy.special.betaln(1 + i, beta_A)
                            - scipy.special.betaln(1 + j, beta_B)
                            - scipy.special.betaln(c['positives'] + 1, beta_C))

    return total


def eval_bernoulli_agg(
        totals: List[int],
        positives: List[int],
        a_priors_beta: List[Number] = None,
        b_priors_beta: List[Number] = None,
        sim_count: int = 200000,
        seed: int = None,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Method estimating probabilities of being best and expected loss for beta-bernoulli
    aggregated data per variant.

    Parameters
    ----------
    totals : List of numbers of experiment observations (e.g. number of sessions) for each variant.
    positives : List of numbers of ones (e.g. number of conversions) for each variant.
    sim_count : Number of simulations to be used for probability estimation.
    a_priors_beta : List of prior alpha parameters for Beta distributions for each variant.
    b_priors_beta : List of prior beta parameters for Beta distributions for each variant.
    seed : Random seed.

    Returns
    -------
    res_pbbs : List of probabilities of being best for each variant.
    res_loss : List of expected loss for each variant.
    """
    validate_bernoulli_input(totals, positives)

    if len(totals) == 0:
        return [], []

    # Default prior for all variants is Beta(0.5, 0.5) which is non-information prior.
    if not a_priors_beta:
        a_priors_beta = [0.5] * len(totals)
    if not b_priors_beta:
        b_priors_beta = [0.5] * len(totals)

    beta_samples = beta_posteriors_all(
        totals, positives, sim_count, a_priors_beta, b_priors_beta, seed
    )

    res_pbbs = estimate_chance_to_beat(beta_samples)
    res_loss = estimate_expected_loss(beta_samples)

    expected_loss_accuracy_bernoulli(beta_samples)

    return res_pbbs, res_loss, beta_samples


def eval_normal_agg(
        totals: List[int],
        sums: List[float],
        sums_2: List[float],
        sim_count: int = 20000,
        m_priors: List[Number] = None,
        a_priors_ig: List[Number] = None,
        b_priors_ig: List[Number] = None,
        w_priors: List[Number] = None,
        seed: int = None,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Method estimating probabilities of being best and expected loss for normal
    aggregated data per variant.

    Parameters
    ----------
    totals : List of numbers of experiment observations for each variant.
    sums : List of sum of original data for each variant.
    sums_2 : List of sum of squares of original data for each variant.
    sim_count : Number of simulations.
    m_priors : List of prior means for each variant.
    a_priors_ig : List of prior alphas from inverse gamma dist approximating variance.
    b_priors_ig : List of prior betas from inverse gamma dist approximating variance.
    w_priors : List of prior effective sample sizes for each variant.
    seed : Random seed.

    Returns
    -------
    res_pbbs : List of probabilities of being best for each variant.
    res_loss : List of expected loss for each variant.
    """
    if len(totals) == 0:
        return [], []
    # Same default priors for all variants if they are not provided.
    if not m_priors:
        m_priors = [1] * len(totals)
    if not a_priors_ig:
        a_priors_ig = [0] * len(totals)
    if not b_priors_ig:
        b_priors_ig = [0] * len(totals)
    if not w_priors:
        w_priors = [0.01] * len(totals)

    # we will need different generators for each call of normal_posteriors
    # (so they are not perfectly correlated)
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(len(totals))

    normal_samples = np.array(
        [
            normal_posteriors(
                totals[i],
                sums[i],
                sums_2[i],
                sim_count,
                m_priors[i],
                a_priors_ig[i],
                b_priors_ig[i],
                w_priors[i],
                child_seeds[i],
            )[0]
            for i in range(len(totals))
        ]
    )

    res_pbbs = estimate_chance_to_beat(normal_samples)
    res_loss = estimate_expected_loss(normal_samples)

    return res_pbbs, res_loss, normal_samples


def eval_delta_lognormal_agg(
        totals: List[int],
        non_zeros: List[int],
        sum_logs: List[float],
        sum_logs_2: List[float],
        sim_count: int = 20000,
        a_priors_beta: List[Number] = None,
        b_priors_beta: List[Number] = None,
        m_priors: List[Number] = None,
        a_priors_ig: List[Number] = None,
        b_priors_ig: List[Number] = None,
        w_priors: List[Number] = None,
        seed: int = None,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Method estimating probabilities of being best and expected loss for delta-lognormal
    aggregated data per variant. For that reason, the method works with both totals and non_zeros.

    Parameters
    ----------
    totals : List of numbers of experiment observations (e.g. number of sessions) for each variant.
    non_zeros : List of numbers of non-zeros (e.g. number of conversions) for each variant.
    sum_logs : List of sum of logarithms of original data for each variant.
    sum_logs_2 : List of sum of logarithms squared of original data for each variant.
    sim_count : Number of simulations.
    a_priors_beta : List of prior alpha parameters for Beta distributions for each variant.
    b_priors_beta : List of prior beta parameters for Beta distributions for each variant.
    m_priors : List of prior means for logarithms of non-zero data for each variant.
    a_priors_ig : List of prior alphas from inverse gamma dist approximating variance of logarithms.
    b_priors_ig : List of prior betas from inverse gamma dist approximating variance of logarithms.
    w_priors : List of prior effective sample sizes for each variant.
    seed : Random seed.

    Returns
    -------
    res_pbbs : List of probabilities of being best for each variant.
    res_loss : List of expected loss for each variant.
    """
    if len(totals) == 0:
        return [], []
    # Same default priors for all variants if they are not provided.
    if not a_priors_beta:
        a_priors_beta = [0.5] * len(totals)
    if not b_priors_beta:
        b_priors_beta = [0.5] * len(totals)
    if not m_priors:
        m_priors = [1] * len(totals)
    if not a_priors_ig:
        a_priors_ig = [0] * len(totals)
    if not b_priors_ig:
        b_priors_ig = [0] * len(totals)
    if not w_priors:
        w_priors = [0.01] * len(totals)

    if max(non_zeros) <= 0:
        # if only zeros in all variants
        res_pbbs = list(np.full(len(totals), round(1 / len(totals), 7)))
        res_loss = [np.nan] * len(totals)
        return res_pbbs, res_loss
    else:
        # we will need different generators for each call of lognormal_posteriors
        ss = np.random.SeedSequence(seed)
        child_seeds = ss.spawn(len(totals) + 1)

        beta_samples = beta_posteriors_all(
            totals, non_zeros, sim_count, a_priors_beta, b_priors_beta, child_seeds[0]
        )

        lognormal_samples = np.array(
            [
                lognormal_posteriors(
                    non_zeros[i],
                    sum_logs[i],
                    sum_logs_2[i],
                    sim_count,
                    m_priors[i],
                    a_priors_ig[i],
                    b_priors_ig[i],
                    w_priors[i],
                    child_seeds[1 + i],
                )
                for i in range(len(totals))
            ]
        )

        combined_samples = beta_samples * lognormal_samples

        res_pbbs = estimate_chance_to_beat(combined_samples)
        res_loss = estimate_expected_loss(combined_samples)

        return res_pbbs, res_loss, combined_samples


def eval_numerical_dirichlet_agg(
        states: List[Union[float, int]],
        concentrations: List[List[int]],
        prior_alphas: List[List[Union[float, int]]] = None,
        sim_count: int = 20000,
        seed: int = None,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Method estimating probabilities of being best and expected loss for dirichlet-multinomial
    aggregated data per variant. States in this case are expected to be a numerical values
    (e.g. dice numbers, number of stars in a rating, etc.).

    Parameters
    ----------
    states : All possible outcomes in given multinomial distribution.
    concentrations : Concentration of observations for each state for all variants.
    prior_alphas : Prior alpha values for each state for all variants.
    sim_count : Number of simulations.
    seed : Random seed.

    Returns
    -------
    res_pbbs : List of probabilities of being best for each variant.
    res_loss : List of expected loss for each variant.
    """
    if len(concentrations) == 0:
        return [], []

    # default prior will be expecting 1 observation in all states for all variants
    if not prior_alphas:
        prior_alphas = [[1] * len(states) for i in range(len(concentrations))]

    # we will need different generators for each call of dirichlet_posteriors
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(len(concentrations))

    means_samples = []
    for i in range(len(concentrations)):
        dir_post = dirichlet_posteriors(
            concentrations[i], prior_alphas[i], sim_count, child_seeds[i]
        )
        means = np.sum(np.multiply(dir_post, np.array(states)), axis=1)
        means_samples.append(list(means))

    res_pbbs = estimate_chance_to_beat(means_samples)
    res_loss = estimate_expected_loss(means_samples)

    return res_pbbs, res_loss, means_samples