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
    gamma_posteriors
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
    For each variant, estimate expected loss of selecting, considering simulated data from respective posteriors.

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


def print_closed_form_comparison(variants: list, pbbs: list, cf_pbbs: list) -> None:
    """
    Pretty-print output comparing the estimate to the exact chance to beat all.
    """
    tab = PrettyTable()
    tab.field_names = ['Variant', 'Est. chance to beat all', 'Exact chance to beat all', 'Delta']
    for var, est, cf in zip(variants, pbbs, cf_pbbs):
        tab.add_row([var, f'{est:.2%}', f'{cf:.2%}', f'{(est - cf) / cf:.2%}'])

    tab.reversesort = True
    tab.sortby = 'Est. chance to beat all'

    print(tab, '\n')


def print_bernoulli_evaluation(res: list) -> None:
    """
    Pretty-print output of running standard binary test.
    """
    tab = PrettyTable()
    tab.field_names = ['Variant', 'Totals', 'Positives', 'Positive rate',
                       'Chance to beat all', 'Expected loss', 'Uplift vs. "A"', '95% HDI']
    for r in res:
        temp_row = r.copy()
        for i in ['positive_rate', 'prob_being_best', 'expected_loss', 'uplift_vs_a']:
            temp_row[i] = f'{temp_row[i]:.2%}'
        temp_row = [temp_row['variant'],
                    temp_row['total'],
                    temp_row['positives'],
                    temp_row['positive_rate'],
                    temp_row['prob_being_best'],
                    temp_row['expected_loss'],
                    temp_row['uplift_vs_a'],
                    f'[{temp_row["bounds"][0]:.2%}, {temp_row["bounds"][1]:.2%}]']

        tab.add_row(temp_row)

    tab.reversesort = True
    tab.sortby = 'Chance to beat all'

    print(tab, '\n')


def print_poisson_evaluation(res: list) -> None:
    """
    Pretty-print output of running Poisson test.
    """
    tab = PrettyTable()
    tab.field_names = ['Variant', 'Observations', 'Mean', 'Chance to beat all', 'Expected loss', '95% HDI']
    for r in res:
        temp_row = r.copy()
        temp_row['prob_being_best'] = f"{temp_row['prob_being_best']:.2%}"
        temp_row['expected_loss'] = round(temp_row['expected_loss'], 2)
        temp_row['mean'] = round(temp_row['mean'], 1)
        temp_row = [temp_row['variant'],
                    temp_row["total"],
                    temp_row['mean'],
                    temp_row['prob_being_best'],
                    temp_row['expected_loss'],
                    f'[{temp_row["bounds"][0]:.1f}, {temp_row["bounds"][1]:.1f}]']

        tab.add_row(temp_row)

    tab.reversesort = True
    tab.sortby = 'Chance to beat all'

    print(tab, '\n')


def eval_closed_form_poisson_loss() -> None:
    """
    This is not currently implemented but may be implemented in a future release.
    """
    raise NotImplementedError


def eval_closed_form_poisson_two(alpha_a: List[Number],
                                 alpha_b: List[Number],
                                 beta_a: List[Number] = 1,
                                 beta_b: List[Number] = 1
                                 ) -> float:
    """
    Given two variants A and B, calculate the probability that B will beat A (closed-form).

    Parameters
    ----------
    alpha_a : List containing observation counts for variant A.
    alpha_b : List containing observation counts for variant B.
    beta_a : Exposure value for variant A.
    beta_b : Exposure value for variant B.

    Returns
    -------
    total : Probability that A will beat B.
    """
    total = 0
    for k in range(alpha_a + 1):
        total += np.exp(k * np.log(beta_a) + alpha_b * np.log(beta_b) - (k + alpha_b) * np.log(beta_a + beta_b)
                        - np.log(k + alpha_b) - scipy.special.betaln(k + 1, alpha_b))

    return total


def eval_closed_form_poisson_three(alpha_a: List[Number],
                                   alpha_b: List[Number],
                                   alpha_c: List[Number],
                                   beta_a: List[Number] = 1,
                                   beta_b: List[Number] = 1,
                                   beta_c: List[Number] = 1
                                   ) -> float:
    """
    Given three variants A, B, and C, calculate the probability that A will beat both B and C (closed-form).

    Parameters
    ----------
    alpha_a : List containing observation counts for variant A.
    alpha_b : List containing observation counts for variant B.
    alpha_c : List containing observation counts for variant C.
    beta_a : Exposure value for variant A.
    beta_b : Exposure value for variant B.
    beta_c : Exposure value for variant C.

    Returns
    -------
    total : Probability that A will beat B and C.
    """
    total = 0
    for k in range(alpha_b + 1):
        for j in range(alpha_c + 1):
            total += np.exp(alpha_a * np.log(beta_a) +
                            k * np.log(beta_b) +
                            j * np.log(beta_c) -
                            (k + j + alpha_a) * np.log(beta_a + beta_b + beta_c) +
                            scipy.special.gammaln(k + j + alpha_a) -
                            scipy.special.gammaln(k + 1) -
                            scipy.special.gammaln(j + 1) - scipy.special.gammaln(alpha_a))

    return total


def validate_bernoulli_input(totals: List[int], positives: List[int]) -> None:
    """
    Simple validation for pbb_bernoulli_agg inputs.
    """
    if len(totals) != len(positives):
        msg = f"Totals ({totals}) and positives ({positives}) needs to have same length!"
        logger.error(msg)
        raise ValueError(msg)


def eval_closed_form_bernoulli_loss() -> None:
    """
    This is not currently implemented but may be implemented in a future release.
    """
    raise NotImplementedError


def eval_closed_form_bernoulli_two(a: Dict, b: Dict) -> float:
    """
    Given two variants A and B, calculate the probability that B will beat A (closed-form).

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
                                             2 + b["total"] - b['positives'] + a["total"] - a['positives']) -
                        np.log(b["total"] - b['positives'] + 1 + k) -
                        scipy.special.betaln(1 + k, b["total"] - b['positives'] + 1) -
                        scipy.special.betaln(a['positives'] + 1, 1 + a["total"] - a['positives']))

    return total


def eval_closed_form_bernoulli_three(a: Dict, b: Dict, c: Dict) -> float:
    """
    Given three variants A, B, and C, calculate the probability that C will beat both B and A (closed-form).

    Parameters
    ----------
    a : Dictionary containing summary statistics for variant A; must contain total observations, positives.
    b : Dictionary containing summary statistics for variant B; must contain total observations, positives.
    c : Dictionary containing summary statistics for variant C; must contain total observations, positives.

    Returns
    -------
    total : Probability that C will beat both B and A.
    """
    total = 0.0
    for i in range(a['positives'] + 1):
        for j in range(b['positives'] + 1):
            beta_A = a["total"] - a['positives'] + 1
            beta_B = b["total"] - b['positives'] + 1
            beta_C = c["total"] - c['positives'] + 1

            total += np.exp(scipy.special.betaln(c['positives'] + 1 + i + j, beta_A + beta_B + beta_C)
                            - np.log(beta_A + i) - np.log(beta_B + j)
                            - scipy.special.betaln(1 + i, beta_A)
                            - scipy.special.betaln(1 + j, beta_B)
                            - scipy.special.betaln(c['positives'] + 1, beta_C))

    return total


def expected_loss_accuracy_bernoulli(data: Union[List[List[Number]], np.ndarray]) -> None:
    """
    Validate that the estimated expected loss is within <epsilon> of the true expected loss with probability <tau>.
    """
    epsilon = 0.0001
    s_2 = len(data) * np.var(data[0])
    for i in data[1:]:
        s_2 += np.var(i)

    n = data.shape[1]
    tau = scipy.stats.norm.cdf(np.sqrt(n) * epsilon / np.sqrt(s_2)) * 2 - 1

    if tau < 0.99:
        msg = f"There is at least a 1% probability that the estimated expected loss is not within {epsilon} tolerance."
        logger.warn(msg)
        warnings.warn(msg)


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
        a_priors_beta = [1] * len(totals)
    if not b_priors_beta:
        b_priors_beta = [1] * len(totals)

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
        a_priors_beta = [1] * len(totals)
    if not b_priors_beta:
        b_priors_beta = [1] * len(totals)
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


def eval_poisson_agg(
        totals: List[int],
        mean: List[Number],
        a_priors_gamma: List[Number] = None,
        b_priors_gamma: List[Number] = None,
        sim_count: int = 200000,
        seed: int = None,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Method estimating probabilities of being best and expected loss for beta-bernoulli
    aggregated data per variant.

    Parameters
    ----------
    totals : List of numbers of experiment observations (e.g. number of sessions) for each variant.
    mean : Mean of the observations for each variant.
    a_priors_gamma : List of prior alpha parameters for Gamma distributions for each variant.
    b_priors_gamma : List of prior beta parameters for Gamma distributions for each variant.
    sim_count : Number of simulations to be used for probability estimation.
    seed : Random seed.

    Returns
    -------
    res_pbbs : List of probabilities of being best for each variant.
    res_loss : List of expected loss for each variant.
    """
    # validate_poisson_input(totals, positives)

    if len(totals) == 0:
        return [], []

    # Default prior for all variants is Beta(0.5, 0.5) which is non-information prior.
    if not a_priors_gamma:
        a_priors_gamma = [1] * len(totals)
    if not b_priors_gamma:
        b_priors_gamma = [1] * len(totals)

    gamma_samples = gamma_posteriors(
        totals, mean, a_priors_gamma, b_priors_gamma, sim_count, seed
    )

    res_pbbs = estimate_chance_to_beat(gamma_samples)
    res_loss = estimate_expected_loss(gamma_samples)

    # expected_loss_accuracy_poisson(gamma_samples)

    return res_pbbs, res_loss, gamma_samples
