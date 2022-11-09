[![Tests](https://github.com/PlatosTwin/bayes_ab/workflows/Tests/badge.svg)](https://github.com/PlatosTwin/bayes_ab/actions?workflow=Tests)
[![Codecov](https://codecov.io/gh/PlatosTwin/bayes_ab/branch/main/graph/badge.svg)](https://codecov.io/gh/PlatosTwin/bayes_ab)
[![PyPI](https://img.shields.io/pypi/v/bayes_ab.svg)](https://pypi.org/project/bayes_ab/)

# Bayesian A/B testing

`bayes_ab` is a small package for running Bayesian A/B(/C/D/...) tests.

**Implemented tests:**

- [BinaryDataTest](bayes_ab/experiments/binary.py)
    - **_Input data_** - binary (`[0, 1, 0, ...]`)
    - Designed for binary data, such as conversions
- [PoissonDataTest](bayes_ab/experiments/poisson.py)
    - **_Input data_** - integer counts
    - Designed for count data (e.g., number of sales per salesman, deaths per zip code)
- [NormalDataTest](bayes_ab/experiments/normal.py)
    - **_Input data_** - normal data with unknown variance
    - Designed for normal data
- [DeltaLognormalDataTest](bayes_ab/experiments/delta_lognormal.py)
    - **_Input data_** - lognormal data with zeros
    - Designed for lognormal data, such as revenue per conversions
- [DiscreteDataTest](bayes_ab/experiments/discrete.py)
    - **_Input data_** - categorical data with numerical categories
    - Designed for discrete data (e.g. dice rolls, star ratings, 1-10 ratings)

**Implemented evaluation metrics:**

- `Chance to beat all`
    - Probability of beating all other variants
- `Expected Loss`
    - Risk associated with choosing a given variant over other variants
    - Measured in the same units as the tested measure (e.g. positive rate or average value)
- `Uplift vs. 'A'`
    - Uplift of a given variant compared to the first variant added
- `95% HDI`
    - 95% confidence interval. The Bayesian approach allows us to say that, 95% of the time, the 95% HDI will contain the true value

Evaluation metrics are calculated using Monte Carlo simulations from posterior distributions

**Closed form solutions:**

For smaller Binary and Poisson samples, metrics calculated from Monte Carlo simulation can be checked against the
closed-form solutions by passing `closed_form=True` to the `evaluate()` method. Larger samples generate warnings;
samples that are too large will raise an error.

**Error tolerance:**

Binary tests with small sample sizes will raise a warning when the error for the expected loss estimate surpasses a set
tolerance. To reduce error, increase the simulation count.

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

The primary features are classes:

- `BinaryDataTest`
- `PoissonDataTest`
- `NormalDataTest`
- `DeltaLognormalDataTest`
- `DiscreteDataTest`

In all cases, there are two methods for inserting data:

- `add_variant_data` - add raw data for a variant as a list of observations (or numpy 1-D array)
- `add_variant_data_agg` - add aggregated variant data (this can be practical for a larger data set, as the aggregation
  can be done outside of the package)

Both methods for adding data allow the user to specify a prior distribution (see details in respective docstrings).
The default priors are non-informative priors and should be sufficient for most use cases, and in particular when
the number of samples or observations is large.

To get the results of the test, simply call method `evaluate`.

Chance to beat all and expected loss are approximated using Monte Carlo simulation, so `evaluate` may return slightly
different values for different runs. To decrease variation, you can set the `sim_count` parameter of `evaluate`
to a higher value (default value is 200K); to fix values, set the `seed` parameter.

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

# add variant using raw data (arrays of zeros and ones):
test.add_variant_data("A", data_a)
test.add_variant_data("B", data_b)
# priors can be specified like this (default for this test is a=b=1/2):
# test.add_variant_data("B", data_b, a_prior=1, b_prior=20)

# add variant using aggregated data (same as raw data with 950 zeros and 50 ones):
test.add_variant_data_agg("C", total=1000, positives=50)

# evaluate test:
test.evaluate()

# generate plots
test.plot_posteriors(fname='binary_posteriors_example.png')
test.plot_differences(control='A', fname='binary_differences_example.png')
```

    +---------+--------+-----------+---------------+--------------------+---------------+----------------+----------------+
    | Variant | Totals | Positives | Positive rate | Chance to beat all | Expected loss | Uplift vs. "A" |    95% HDI     |
    +---------+--------+-----------+---------------+--------------------+---------------+----------------+----------------+
    |    B    |  1200  |     80    |     6.74%     |       89.27%       |     0.05%     |     24.96%     | [5.59%, 7.97%] |
    |    A    |  1500  |     80    |     5.39%     |       6.44%        |     1.40%     |     0.00%      | [4.47%, 6.38%] |
    |    C    |  1000  |     50    |     5.09%     |       4.29%        |     1.69%     |     -5.62%     | [4.00%, 6.28%] |
    +---------+--------+-----------+---------------+--------------------+---------------+----------------+----------------+

Removing variant 'C' and passing `control='A'` and `rope=0.5` additionally returns a test-continuation recommendation:

```python
test.delete_variant("C")
```
    Decision: Stop and implement either variant. Confidence: Low. Bounds: [-0.46%, 3.18%].

Finally, we can plot the posterior distributions as well as the distribution of differences.

![](https://raw.githubusercontent.com/PlatosTwin/bayes-ab/main/examples/plots/binary_posteriors_example.png)

![](https://raw.githubusercontent.com/PlatosTwin/bayes-ab/main/examples/plots/binary_differences_example.png)

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
data_a = rng.poisson(42, size=20)
data_b = rng.poisson(40, size=25)
data_c = rng.poisson(43, size=15)

# initialize a test:
test = PoissonDataTest()

# add variant using raw data:
test.add_variant_data("A", data_a)
test.add_variant_data("B", data_b)
# test.add_variant_data("C", data_c)

# add variant using aggregated data:
test.add_variant_data_agg("C", total=len(data_c), obs_mean=np.mean(data_c), obs_sum=sum(data_c))

# evaluate test:
test.evaluate(sim_count=20000, seed=52)

# generate plots
test.plot_posteriors(fname='poisson_posteriors_example.png')
test.plot_differences(control='A', fname='poisson_differences_example.png')
```
    +---------+--------------+------+--------------------+---------------+--------------+
    | Variant | Observations | Mean | Chance to beat all | Expected loss |   95% HDI    |
    +---------+--------------+------+--------------------+---------------+--------------+
    |    C    |      15      | 42.5 |       56.38%       |      0.69     | [39.9, 45.2] |
    |    A    |      20      | 42.1 |       42.57%       |      1.08     | [39.8, 44.5] |
    |    B    |      25      | 39.2 |       1.05%        |      3.97     | [37.2, 41.3] |
    +---------+--------------+------+--------------------+---------------+--------------+

Removing variant 'C' and passing `control='A'` and `rope=0.5` additionally returns a test-continuation recommendation:

```python
test.delete_variant("C")
```

    Decision: Stop and implement either variant. Confidence: Low. Bounds: [-6.6, 0.7].

Finally, we can plot the posterior distributions as well as the distribution of differences.

![](https://raw.githubusercontent.com/PlatosTwin/bayes-ab/main/examples/plots/poisson_posteriors_example.png)

![](https://raw.githubusercontent.com/PlatosTwin/bayes-ab/main/examples/plots/poisson_differences_example.png)

### NormalDataTest

Class for Bayesian A/B test for normal data.

**Example:**

```python
import numpy as np
from bayes_ab.experiments import NormalDataTest

# generating some random data
rng = np.random.default_rng(21)
data_a = rng.normal(7.2, 2, 1000)
data_b = rng.normal(7.1, 2, 800)
data_c = rng.normal(7.0, 4, 500)

# initialize a test:
test = NormalDataTest()

# add variant using raw data:
test.add_variant_data("A", data_a)
test.add_variant_data("B", data_b)
# test.add_variant_data("C", data_c)

# add variant using aggregated data:
test.add_variant_data_agg("C", len(data_c), sum(data_c), sum(np.square(data_c)))

# evaluate test:
test.evaluate(sim_count=20000, seed=52)
```

    [{'variant': 'A',
      'totals': 1000,
      'sum_values': 7294.67901,
      'avg_values': 7.29468,
      'prob_being_best': 0.1707,
      'expected_loss': 0.1968735},
     {'variant': 'B',
      'totals': 800,
      'sum_values': 5685.86168,
      'avg_values': 7.10733,
      'prob_being_best': 0.00125,
      'expected_loss': 0.385112},
     {'variant': 'C',
      'totals': 500,
      'sum_values': 3736.91581,
      'avg_values': 7.47383,
      'prob_being_best': 0.82805,
      'expected_loss': 0.0169998}]

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
import numpy as np
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

Test classes to add:

- `ExponentialDataTest`

Other improvements:

- Updated Jupyter examples folder
- Validate `NormalDataTest`, `DeltaLognormalDataTest`, and `DiscreteDataTest`
- Updates to `NormalDataTest`, `DeltaLognormalDataTest`, and `DiscreteDataTest`
  - Add test continuation assessment
  - Created formatted output
  - Add plotting for posteriors and differences from control
- Plot evolutions of posteriors with time

## References

This package leans heavily on the resources outlined below:
- [Bayesian A/B Testing at VWO](https://vwo.com/downloads/VWO_SmartStats_technical_whitepaper.pdf)
(Chris Stucchio, 2015)
- [Optional stopping in data collection: p values, Bayes factors, credible intervals, precision](
http://doingbayesiandataanalysis.blogspot.com/2013/11/optional-stopping-in-data-collection-p.html) (John Kruschke, 2013)
- [Is Bayesian A/B Testing Immune to Peeking? Not Exactly](http://varianceexplained.org/r/bayesian-ab-testing/)
(David Robinson, 2015)
- [Formulas for Bayesian A/B Testing](https://www.evanmiller.org/bayesian-ab-testing.html) (Evan Miller, 2015)
- [Easy Evaluation of Decision Rules in Bayesian A/B testing](
https://www.chrisstucchio.com/blog/2014/bayesian_ab_decision_rule.html) (Chris Stucchio, 2014)
- [Bayesian Inference 2019](https://vioshyvo.github.io/Bayesian_inference/index.html) (Hyvönen & Tolonen, 2019)
- [Continuous Monitoring of A/B Tests without Pain: Optional
Stopping in Bayesian Testing](https://arxiv.org/pdf/1602.05549.pdf) (Deng, Lu, & Chen, 2016)
- [Bayesian Data Analysis, Third Edition](http://www.stat.columbia.edu/~gelman/book/BDA3.pdf) (Gelman et al., 2021)

## Online calculators
- [Yanir Seroussi's calculator](https://yanirs.github.io/tools/split-test-calculator/) |
[project description](https://yanirseroussi.com/2016/06/19/making-bayesian-ab-testing-more-accessible/)
- [Lyst's calculator](https://making.lyst.com/bayesian-calculator/) | [project descrption](https://making.lyst.com/2014/05/10/bayesian-ab-testing/)
- [Dynamic Yield's calculator](https://marketing.dynamicyield.com/bayesian-calculator/)
