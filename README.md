[![Tests](https://github.com/PlatosTwin/bayes_ab/workflows/Tests/badge.svg)](https://github.com/PlatosTwin/bayes_ab/actions?workflow=Tests)
[![Codecov](https://codecov.io/gh/PlatosTwin/bayes_ab/branch/main/graph/badge.svg)](https://codecov.io/gh/PlatosTwin/bayes_ab)
[![PyPI](https://img.shields.io/pypi/v/bayes_ab.svg)](https://pypi.org/project/bayes_ab/)

# Bayesian A/B testing

`bayes_ab` is a small package for running Bayesian A/B(/C/D/...) tests.

### Implemented tests

- [BinaryDataTest](https://github.com/PlatosTwin/bayes_ab/blob/main/bayes_ab/experiments/binary.py)
    - **_Input data_** — binary (`[0, 1, 0, ...]`)
    - Designed for binary data, such as conversions
- [PoissonDataTest](https://github.com/PlatosTwin/bayes_ab/blob/main/bayes_ab/experiments/poisson.py)
    - **_Input data_** — integer counts
    - Designed for count data (e.g., number of sales per salesman, deaths per zip code)
- [NormalDataTest](https://github.com/PlatosTwin/bayes_ab/blob/main/bayes_ab/experiments/normal.py)
    - **_Input data_** — normal data with unknown variance
    - Designed for normal data
- [DeltaLognormalDataTest](https://github.com/PlatosTwin/bayes_ab/blob/main/bayes_ab/experiments/delta_lognormal.py)
    - **_Input data_** — lognormal data with zeros
    - Designed for lognormal data, such as revenue per conversions
- [DiscreteDataTest](https://github.com/PlatosTwin/bayes_ab/blob/main/bayes_ab/experiments/discrete.py)
    - **_Input data_** — categorical data with numerical categories
    - Designed for discrete data (e.g. dice rolls, star ratings, 1-10 ratings)

### Implemented evaluation metrics

- `Chance to beat all`
    - Probability of beating all other variants
- `Expected Loss`
    - Risk associated with choosing a given variant over other variants
    - Measured in the same units as the tested measure (e.g. positive rate or average value)
- `Uplift vs. 'A'`
    - Relative uplift of a given variant compared to the first variant added
- `95% HDI`
    - The central interval containing 95% of the probability. The Bayesian approach allows us to say that, 95% of the
      time, the 95% HDI will contain the true value

Evaluation metrics are calculated using Monte Carlo simulations from posterior distributions

### Decision rules for test continuation

For tests between two variants, `bayes_ab` can additionally provide a continuation recommendation—that is, a
recommendation as to the variant to select, or to continue testing. See the docstrings and examples for usage
guidelines.

The decision method makes use of the following concepts:

- **Region of Practical Equivalence (ROPE)** — a region `[-t, t]` of the distribution of differences `B - A` which is
  practically equivalent to no uplift. E.g., you may be indifferent between an uplift of +/- 0.1% and no change, in
  which case the ROPE would be `[-0.1, 0.1`.
- **95% HDI** — the central interval containing 95% of the probability for the distribution of differences
  `B - A`.

The recommendation output has three elements:

1. **Decision**
    - _Stop and select either variant_ if the ROPE is fully contained within the 95% HDI.
    - _Continue testing_ if the ROPE partially overlaps the 95% HDI.
    - _Stop testing and select the better variant_ if the ROPE and the 95% HDI do not overlap.
2. **Confidence**
    - _High_ if the width of the 95% HDI is less than or equal to `0.8*rope`.
    - _Low_ if the width of the 95% HDI is greater than `0.8*rope`.
3. **Bounds**
    - The 95% HDI.

### Closed form solutions

For smaller Binary and Poisson samples, metrics calculated from Monte Carlo simulation can be checked against the
closed-form solutions by passing `closed_form=True` to the `evaluate()` method. Larger samples generate warnings;
samples that are larger than a predetermined threshold will raise an error. The larger the sample, however, the closer
the simulated value will be to the true value, so closed-form comparisons are recommended to validate metrics for
smaller samples only.

### Error tolerance

Binary tests with small sample sizes will raise a warning when the error for the expected loss estimate surpasses a set
tolerance. To reduce error, increase the simulation count. For more detail, see the docstring
for `expected_loss_accuracy_bernoulli`
in [`evaluation.py`](https://github.com/PlatosTwin/bayes_ab/blob/main/bayes_ab/metrics/evaluation.py)

## Installation

`bayes_ab` can be installed using pip:

```console
pip install bayes_ab
```

Alternatively, you can clone the repository and use `poetry` manually:

```console
cd bayes_ab
pip install poetry
poetry install
poetry shell
```

## Basic Usage

There are five primary classes:

- `BinaryDataTest`
- `PoissonDataTest`
- `NormalDataTest`
- `DeltaLognormalDataTest`
- `DiscreteDataTest`

For each class, there are two methods for inserting data:

- `add_variant_data` - add raw data for a variant as a list of observations (or numpy 1-D array)
- `add_variant_data_agg` - add aggregated variant data (this can be practical for a larger data set, as the aggregation
  can be done outside the package)

Both methods for adding data allow the user to specify a prior distribution (see details in respective docstrings). The
default priors are non-informative priors and should be sufficient for most use cases, and in particular when the number
of samples or observations is large.

To get the results of the test, simply call method `evaluate`; to access evaluation metrics as well as the simulated
random samples, call the `data` instance variable.

Chance to beat all and expected loss are approximated using Monte Carlo simulation, so `evaluate` may return slightly
different values for different runs. To decrease variation, you can set the `sim_count` parameter of `evaluate`
to a higher value (the default is 200K); to fix values, set the `seed` parameter.

More examples are available in the [examples directory](https://github.com/PlatosTwin/bayes_ab/blob/main/examples/),
though many examples in this directory are still in the process of being updated to reflect the functionality of the
updated package.

### BinaryDataTest

Class for Bayesian A/B test for binary-like data (e.g. conversions, successes, etc.).

**Example:**

```python
import numpy as np
from bayes_ab.experiments import BinaryDataTest

# generating some random data
rng = np.random.default_rng(52)
# random 1x1500 array of 0/1 data with 5.2% probability for 1:
data_a = rng.binomial(n=1, p=0.052, size=1500)
# random 1x1200 array of 0/1 data with 6.7% probability for 1:
data_b = rng.binomial(n=1, p=0.067, size=1200)

# initialize a test:
test = BinaryDataTest()

# add variant using raw data (arrays of zeros and ones) and specifying priors:
test.add_variant_data("A", data_a, a_prior=10, b_prior=17)
test.add_variant_data("B", data_b, a_prior=5, b_prior=30)
# the default priors are a=b=1
# test.add_variant_data("C", data_c)

# add variant using aggregated data:
test.add_variant_data_agg("C", total=1000, positives=50)

# evaluate test:
test.evaluate(seed=314)

# access simulation samples and evaluation metrics
data = test.data

# generate plots
test.plot_distributions(control='A', fname='binary_distributions_example.png')
```

    +---------+--------+-----------+---------------+--------------------+---------------+----------------+----------------+
    | Variant | Totals | Positives | Positive rate | Chance to beat all | Expected loss | Uplift vs. "A" |    95% HDI     |
    +---------+--------+-----------+---------------+--------------------+---------------+----------------+----------------+
    |    B    |  1200  |     80    |     6.88%     |       83.82%       |     0.08%     |     16.78%     | [5.74%, 8.11%] |
    |    C    |  1000  |     50    |     5.09%     |       2.54%        |     1.87%     |    -13.64%     | [4.00%, 6.28%] |
    |    A    |  1500  |     80    |     5.89%     |       13.64%       |     1.07%     |     0.00%      | [4.94%, 6.92%] |
    +---------+--------+-----------+---------------+--------------------+---------------+----------------+----------------+

For smaller samples, such as the above, it is also possible to check the modeled chance to beat all against the
closed-form equivalent by passing `closed_form=True`.

```python
test.evaluate(closed_form=True, seed=314)
```

    +---------+-------------------------+--------------------------+--------+
    | Variant | Est. chance to beat all | Exact chance to beat all | Delta  |
    +---------+-------------------------+--------------------------+--------+
    |    B    |          83.82%         |          83.58%          | 0.28%  |
    |    C    |          2.54%          |          2.56%           | -0.66% |
    |    A    |          13.64%         |          13.86%          | -1.59% |
    +---------+-------------------------+--------------------------+--------+

Removing variant 'C', as this feature is implemented for two variants only currently, and passing a value to `control`
additionally returns a test-continuation recommendation:

```python
test.delete_variant("C")
test.evaluate(control='A', seed=314)
```

    Decision: Stop and implement either variant. Confidence: Low. Bounds: [-0.84%, 2.85%].

Finally, we can plot the prior and posterior distributions, as well as the distribution of differences.

![](https://raw.githubusercontent.com/PlatosTwin/bayes_ab/main/examples/plots/binary_distributions_example.png)

### PoissonDataTest

Class for Bayesian A/B test for count data. This can be used to compare, e.g., the number of sales per day from
different salesmen, or the number of deaths from a given disease per zip code.

**Example:**

```python
# generating some random data
import numpy as np
from bayes_ab.experiments import PoissonDataTest

# generating some random data
rng = np.random.default_rng(21)
data_a = rng.poisson(43, size=20)
data_b = rng.poisson(39, size=25)
data_c = rng.poisson(37, size=15)

# initialize a test:
test = PoissonDataTest()

# add variant using raw data (arrays of zeros and ones) and specifying priors:
test.add_variant_data("A", data_a, a_prior=30, b_prior=7)
test.add_variant_data("B", data_b, a_prior=5, b_prior=5)
# test.add_variant_data("C", data_c)

# add variant using aggregated data:
test.add_variant_data_agg("C", total=len(data_c), obs_mean=np.mean(data_c), obs_sum=sum(data_c))

# evaluate test:
test.evaluate(seed=314)

# access simulation samples and evaluation metrics
data = test.data

# generate plots
test.plot_distributions(control='A', fname='poisson_distributions_example.png')
```

    +---------+--------------+------+--------------------+---------------+----------------+--------------+
    | Variant | Observations | Mean | Chance to beat all | Expected loss | Uplift vs. "A" |   95% HDI    |
    +---------+--------------+------+--------------------+---------------+----------------+--------------+
    |    C    |      15      | 36.2 |       74.06%       |      0.28     |     4.01%      | [33.8, 38.8] |
    |    B    |      25      | 33.9 |       5.09%        |      2.66     |     -2.83%     | [32.1, 35.6] |
    |    A    |      20      | 34.9 |       20.85%       |      1.68     |     0.00%      | [33.0, 36.7] |
    +---------+--------------+------+--------------------+---------------+----------------+--------------+

For samples smaller than the above, it is also possible to check the modeled chance to beat all against the closed-form
equivalent by passing `closed_form=True`:

```python
test.evaluate(closed_form=True, seed=314)
```

    +---------+-------------------------+--------------------------+--------+
    | Variant | Est. chance to beat all | Exact chance to beat all | Delta  |
    +---------+-------------------------+--------------------------+--------+
    |    C    |          74.06%         |          73.91%          | 0.20%  |
    |    B    |          5.09%          |          5.24%           | -2.84% |
    |    A    |          20.85%         |          20.85%          | -0.01% |
    +---------+-------------------------+--------------------------+--------+

Removing variant 'C', as this feature is implemented for two variants only currently, and passing `control` and `rope`
additionally returns a test-continuation recommendation:

```python
test.delete_variant("C")
test.evaluate(control='A', rope=0.5, seed=314)
```

    Decision: Stop and implement either variant. Confidence: Low. Bounds: [-4.0, 2.1].

Finally, we can plot the posterior distributions as well as the distribution of differences (returning now to the
original number of observations rather than the smaller sample used to show the closed-form validation).

![](https://raw.githubusercontent.com/PlatosTwin/bayes_ab/main/examples/plots/poisson_distributions_example.png)

### NormalDataTest

Class for Bayesian A/B test for normal data.

**Example:**

```python
import numpy as np
from bayes_ab.experiments import NormalDataTest

# generating some random data
rng = np.random.default_rng(314)
data_a = rng.normal(6.9, 2, 500)
data_b = rng.normal(6.89, 2, 800)
data_c = rng.normal(7.0, 4, 500)

# initialize a test:
test = NormalDataTest()

# add variant using raw data:
test.add_variant_data("A", data_a)
test.add_variant_data("B", data_b, m_prior=5, n_prior=11, v_prior=10, s_2_prior=4)
# test.add_variant_data("C", data_c)

# add variant using aggregated data:
test.add_variant_data_agg("C", len(data_c), sum(data_c), sum((data_c - np.mean(data_c)) ** 2), sum(np.square(data_c)))

# evaluate test:
test.evaluate(sim_count=200000, seed=314)

# access simulation samples and evaluation metrics
data = test.data

# generate plots
test.plot_joint_prior(variant='B', fname='normal_prior_distribution_B_example.png')
test.plot_distributions(control='A', fname='normal_distributions_example.png')
```

    +---------+--------------+------+-----------+-----------+--------------------+---------------+----------------+----------------+-----------------+
    | Variant | Observations | Mean | Precision | Std. dev. | Chance to beat all | Expected loss | Uplift vs. "A" | 95% HDI (mean) | 95% HDI (stdev) |
    +---------+--------------+------+-----------+-----------+--------------------+---------------+----------------+----------------+-----------------+
    |    A    |     500      | 6.89 |   0.257   |    1.97   |       90.93%       |      0.0      |     0.00%      |  [6.88, 6.91]  |   [1.86, 2.10]  |
    |    B    |     800      | 6.89 |   0.258   |    1.97   |       9.07%        |      0.01     |     -0.09%     |  [6.88, 6.90]  |   [1.88, 2.07]  |
    |    C    |     500      | 6.75 |   0.066   |    3.9    |       0.00%        |      0.14     |     -2.01%     |  [6.69, 6.81]  |   [3.68, 4.16]  |
    +---------+--------------+------+-----------+-----------+--------------------+---------------+----------------+----------------+-----------------+

We can also plot the joint prior distribution for $\mu$ and $\sigma^2$, the posterior distributions for $\mu$ and
$\frac{1}{\sigma^2}$, and the distribution of differences from a given control.

![](https://raw.githubusercontent.com/PlatosTwin/bayes_ab/main/examples/plots/normal_prior_distribution_B_example.png)
![](https://raw.githubusercontent.com/PlatosTwin/bayes_ab/main/examples/plots/normal_distributions_example.png)

### DeltaLognormalDataTest

Class for Bayesian A/B test for delta-lognormal data (log-normal with zeros). Delta-lognormal data is typical case of
revenue per session data where many sessions have 0 revenue but non-zero values are positive numbers with possible
log-normal distribution. To handle this data, the calculation is combining binary Bayes model for zero vs non-zero
"conversions" and log-normal model for non-zero values.

**Example:**

```python
import numpy as np
from bayes_ab.experiments import DeltaLognormalDataTest

test = DeltaLognormalDataTest()

data_a = [7.1, 0.3, 5.9, 0, 1.3, 0.3, 0, 1.2, 0, 3.6, 0, 1.5, 2.2, 0, 4.9, 0, 0, 1.1, 0, 0, 7.1, 0, 6.9, 0]
data_b = [4.0, 0, 3.3, 19.3, 18.5, 0, 0, 0, 12.9, 0, 0, 0, 10.2, 0, 0, 23.1, 0, 3.7, 0, 0, 11.3, 10.0, 0, 18.3, 12.1]

# adding variant using raw data:
test.add_variant_data("A", data_a)
# test.add_variant_data("B", data_b)

# alternatively, variant can be also added using aggregated data:
# (looks more complicated but for large data it can be quite handy to move around only these sums)
test.add_variant_data_agg(
    name="B",
    total=len(data_b),
    positives=sum(x > 0 for x in data_b),
    sum_values=sum(data_b),
    sum_logs=sum([np.log(x) for x in data_b if x > 0]),
    sum_logs_2=sum([np.square(np.log(x)) for x in data_b if x > 0])
)

# evaluate test:
test.evaluate(seed=21)

# access simulation samples and evaluation metrics
data = test.data
```

    [{'variant': 'A',
      'totals': 24,
      'positives': 13,
      'sum_values': 43.4,
      'avg_values': 1.80833,
      'avg_positive_values': 3.33846,
      'prob_being_best': 0.04815,
      'expected_loss': 4.0941101},
     {'variant': 'B',
      'totals': 25,
      'positives': 12,
      'sum_values': 146.7,
      'avg_values': 5.868,
      'avg_positive_values': 12.225,
      'prob_being_best': 0.95185,
      'expected_loss': 0.1588627}]

### DiscreteDataTest

Class for Bayesian A/B test for discrete data with finite number of numerical categories (states), representing some
value. This test can be used for instance for dice rolls data (when looking for the "best" of multiple dice) or rating
data
(e.g. 1-5 stars or 1-10 scale).

**Example:**

```python
from bayes_ab.experiments import DiscreteDataTest

# dice rolls data for 3 dice - A, B, C
data_a = [2, 5, 1, 4, 6, 2, 2, 6, 3, 2, 6, 3, 4, 6, 3, 1, 6, 3, 5, 6]
data_b = [1, 2, 2, 2, 2, 3, 2, 3, 4, 2]
data_c = [1, 3, 6, 5, 4]

# initialize a test with all possible states (i.e. numerical categories):
test = DiscreteDataTest(states=[1, 2, 3, 4, 5, 6])

# add variant using raw data:
test.add_variant_data("A", data_a)
test.add_variant_data("B", data_b)
test.add_variant_data("C", data_c)

# add variant using aggregated data:
# test.add_variant_data_agg("C", [1, 0, 1, 1, 1, 1]) # equivalent to rolls in data_c

# evaluate test:
test.evaluate(sim_count=20000, seed=52)

# access simulation samples and evaluation metrics
data = test.data
```

    [{'variant': 'A',
      'concentration': {1: 2.0, 2: 4.0, 3: 4.0, 4: 2.0, 5: 2.0, 6: 6.0},
      'average_value': 3.8,
      'prob_being_best': 0.54685,
      'expected_loss': 0.199953},
     {'variant': 'B',
      'concentration': {1: 1.0, 2: 6.0, 3: 2.0, 4: 1.0, 5: 0.0, 6: 0.0},
      'average_value': 2.3,
      'prob_being_best': 0.008,
      'expected_loss': 1.1826766},
     {'variant': 'C',
      'concentration': {1: 1.0, 2: 0.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0},
      'average_value': 3.8,
      'prob_being_best': 0.44515,
      'expected_loss': 0.2870247}]

## Development

To set up a development environment, use [Poetry](https://python-poetry.org/) and [pre-commit](https://pre-commit.com):

```console
pip install poetry
poetry install
poetry run pre-commit install
```

## Roadmap

Improvements in the pipeline:

- Implement sample size/reverse posterior calculation
- Update Jupyter examples folder
- Validate `DeltaLognormalDataTest` and `DiscreteDataTest`
- Improve `DeltaLognormalDataTest` and `DiscreteDataTest`
    - Add test continuation assessment (decision, confidence, bounds)
    - Create formatted output
    - Add plotting for posteriors and differences from control
- Add test continuation to `NormalDataTest`
- Refine decision rule (test continuation assessment) to include more nuance
- Implement Markov Chain Monte Carlo in place of Monte Carlo
- Create a method to easily plot evolutions of posteriors and evaluation metrics with time

## References and related work

The development of this package has relied on the resources outlined below. Where a function or method draws directly on
a particular derivation, the docstring contains the exact reference.

- [Bayesian A/B Testing at VWO](https://vwo.com/downloads/VWO_SmartStats_technical_whitepaper.pdf)
  (Chris Stucchio, 2015)
- [Optional stopping in data collection: p values, Bayes factors, credible intervals, precision](
  http://doingbayesiandataanalysis.blogspot.com/2013/11/optional-stopping-in-data-collection-p.html)
  (John Kruschke, 2013)
- [Is Bayesian A/B Testing Immune to Peeking? Not Exactly](http://varianceexplained.org/r/bayesian-ab-testing/)
  (David Robinson, 2015)
- [Formulas for Bayesian A/B Testing](https://www.evanmiller.org/bayesian-ab-testing.html) (Evan Miller, 2015)
- [Easy Evaluation of Decision Rules in Bayesian A/B testing](
  https://www.chrisstucchio.com/blog/2014/bayesian_ab_decision_rule.html) (Chris Stucchio, 2014)

There is a wealth of material on Bayesian statistics available freely online. A small and somewhat random selection is
catalogued below.

- _[Bayesian Data Analysis, Third Edition](http://www.stat.columbia.edu/~gelman/book/BDA3.pdf)_ (Gelman et al., 2021)
- [Bayesian Inference 2019](https://vioshyvo.github.io/Bayesian_inference/index.html) (Hyvönen & Tolonen, 2019)
- [Probabalistic programming and Bayesian methods for hackers](https://nbviewer.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/tree/master/) (
  Cameron Davidson-Pilon, 2022)
- [Think Bayes 2](https://allendowney.github.io/ThinkBayes2/index.html) (Downey, 2021)
- [Continuous Monitoring of A/B Tests without Pain: Optional Stopping in Bayesian Testing](https://arxiv.org/pdf/1602.05549.pdf)
  (Deng, Lu, & Chen, 2016)

This project was inspired by Aubrey Clayton's (2022) _[Bernoulli's Fallacy:
Statistical Illogic and the Crisis of Modern Science](http://cup.columbia.edu/book/bernoullis-fallacy/9780231199940)_.

## Select online calculators

- [Yanir Seroussi's calculator](https://yanirs.github.io/tools/split-test-calculator/) |
  [project description](https://yanirseroussi.com/2016/06/19/making-bayesian-ab-testing-more-accessible/)
- [Lyst's calculator](https://making.lyst.com/bayesian-calculator/)
  | [project descrption](https://making.lyst.com/2014/05/10/bayesian-ab-testing/)
- [Dynamic Yield's calculator](https://marketing.dynamicyield.com/bayesian-calculator/)

## A note on forking

This package was forked from Matus Baniar's [`bayesian-testing`](https://github.com/Matt52/bayesian-testing). Upon
deciding to take package development in a different direction, I detached the fork from the original repository. The
original author's contributions are large, however, with his central contributions being to the development of the core
infrastructure of the project. This being the first package I have worked on, the original author's work to prepare this
code for packaging has also been instrumental to package publication, not to mention educative.
