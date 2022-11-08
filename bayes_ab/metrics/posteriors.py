from typing import List, Tuple, Union

import numpy as np


def beta_posteriors_all(
    totals: List[int],
    positives: List[int],
    sim_count: int,
    a_priors_beta: List[Union[float, int]],
    b_priors_beta: List[Union[float, int]],
    seed: Union[int, np.random.bit_generator.SeedSequence] = None,
) -> np.ndarray:
    """
    Draw from beta posterior distributions for all variants at once.

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
    beta_samples : List of lists of beta distribution samples for all variants.
    """
    rng = np.random.default_rng(seed)

    beta_samples = np.array(
        [
            rng.beta(
                positives[i] + a_priors_beta[i],
                totals[i] - positives[i] + b_priors_beta[i],
                sim_count,
            )
            for i in range(len(totals))
        ]
    )
    return beta_samples


def normal_posteriors(
    total: int,
    sums: float,
    sums_2: float,
    sim_count: int = 200000,
    prior_m: Union[float, int] = 1,
    prior_a: Union[float, int] = 0,
    prior_b: Union[float, int] = 0,
    prior_w: Union[float, int] = 0.01,
    seed: Union[int, np.random.bit_generator.SeedSequence] = None,
) -> Tuple[List[Union[float, int]], List[Union[float, int]]]:
    """
    Drawing mus and sigmas from posterior normal distribution considering given aggregated data.

    Parameters
    ----------
    total : Number of data observations from normal data.
    sums : Sum of original data.
    sums_2 : Sum of squares of original data.
    sim_count : Number of simulations.
    prior_m : Prior mean.
    prior_a : Prior alpha from inverse gamma dist. for unknown variance of original data.
        In theory a > 0, but as we always have at least one observation, we can start at 0.
    prior_b : Prior beta from inverse gamma dist. for unknown variance of original data.
        In theory b > 0, but as we always have at least one observation, we can start at 0.
    prior_w : Prior effective sample size.
    seed : Random seed.

    Returns
    -------
    mu_post : List of size sim_count with mus drawn from normal distribution.
    sig_2_post : List of size sim_count with mus drawn from normal distribution.
    """
    rng = np.random.default_rng(seed)

    x_bar = sums / total
    a_post = prior_a + (total / 2)
    b_post = (
        prior_b
        + (1 / 2) * (sums_2 - 2 * sums * x_bar + total * (x_bar**2))
        + ((total * prior_w) / (2 * (total + prior_w))) * ((x_bar - prior_m) ** 2)
    )

    # here it has to be 1/b as it is a scale, and not a rate
    sig_2_post = 1 / rng.gamma(a_post, 1 / b_post, sim_count)

    m_post = (total * x_bar + prior_w * prior_m) / (total + prior_w)

    mu_post = rng.normal(m_post, np.sqrt(sig_2_post / (total + prior_w)))

    return mu_post, sig_2_post


def lognormal_posteriors(
    total: int,
    sum_logs: float,
    sum_logs_2: float,
    sim_count: int = 200000,
    prior_m: Union[float, int] = 1,
    prior_a: Union[float, int] = 0,
    prior_b: Union[float, int] = 0,
    prior_w: Union[float, int] = 0.01,
    seed: Union[int, np.random.bit_generator.SeedSequence] = None,
) -> List[float]:
    """
    Drawing from posterior lognormal distribution using logarithms of original (lognormal) data
    (logarithms of lognormal data are normal). Input data is in aggregated form.

    Parameters
    ----------
    total : Number of lognormal data observations.
        Could be number of conversions in session data.
    sum_logs : Sum of logarithms of original data.
    sum_logs_2 : Sum of logarithms squared of original data.
    sim_count : Number of simulations.
    prior_m : Prior mean of logarithms of original data.
    prior_a : Prior alpha from inverse gamma dist. for unknown variance of logarithms
        of original data. In theory a > 0, but as we always have at least one observation,
        we can start at 0.
    prior_b : Prior beta from inverse gamma dist. for unknown variance of logarithms
        of original data. In theory b > 0, but as we always have at least one observation,
        we can start at 0.
    prior_w : Prior effective sample size.
    seed : Random seed.

    Returns
    -------
    res : List of sim_count numbers drawn from lognormal distribution.
    """
    if total <= 0:
        return list(np.zeros(sim_count))

    # normal posterior for aggregated data of logarithms of original data
    normal_mu_post, normal_sig_2_post = normal_posteriors(
        total, sum_logs, sum_logs_2, sim_count, prior_m, prior_a, prior_b, prior_w, seed
    )

    # final simulated lognormal means using simulated normal means and sigmas
    res = np.exp(normal_mu_post + (normal_sig_2_post / 2))

    return res


def dirichlet_posteriors(
    concentration: List[int],
    prior: List[Union[float, int]],
    sim_count: int = 200000,
    seed: Union[int, np.random.bit_generator.SeedSequence] = None,
) -> np.ndarray:
    """
    Drawing from dirichlet posterior for a single variant.

    Parameters
    ----------
    concentration : List of numbers of observation for each possible category.
        In dice example it would be numbers of observations for each possible face.
    prior : List of prior values for each category in dirichlet distribution.
    sim_count : Number of simulations.
    seed : Random seed.

    Returns
    -------
    res : List of lists of dirichlet samples.
    """
    rng = np.random.default_rng(seed)

    posterior_concentration = [sum(x) for x in zip(prior, concentration)]
    res = rng.dirichlet(posterior_concentration, sim_count)

    return res


def gamma_posteriors(
    totals: List[int],
    mean: List[float],
    a_priors_gamma: List[Union[float, int]],
    b_priors_gamma: List[Union[float, int]],
    sim_count: int,
    seed: Union[int, np.random.bit_generator.SeedSequence] = None,
) -> np.ndarray:
    """
    Drawing from Poisson posterior for a single variant.

    Parameters
    ----------
    totals : List of numbers of experiment observations (e.g. number of sessions) for each variant.
    mean : Mean of the observations for each variant.
    a_priors_gamma : List of prior alpha parameters for Gamma distributions for each variant.
    b_priors_gamma : List of prior beta parameters for Gamma distributions for each variant.
    sim_count : Number of simulations.
    seed : Random seed.

    Returns
    -------
    res : List of lists of dirichlet samples.
    """
    rng = np.random.default_rng(seed)

    gamma_samples = np.array(
        [
            rng.gamma(
                a_priors_gamma[i] + totals[i] * mean[i],
                1 / (b_priors_gamma[i] + totals[i]),
                sim_count,
            )
            for i in range(len(totals))
        ]
    )

    return gamma_samples
