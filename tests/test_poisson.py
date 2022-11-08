import pytest

from bayes_ab.experiments import PoissonDataTest


@pytest.fixture
def conv_test():
    cv = PoissonDataTest()
    cv.add_variant_data("A", [4, 3, 4, 3, 5, 2, 7, 2, 4, 4])
    cv.add_variant_data("B", [3, 8, 4, 5, 4, 2, 2, 3, 4, 6])
    cv.add_variant_data_agg("C", 11, 4, 39, a_prior=2, b_prior=2)
    cv.add_variant_data_agg("D", 10, 4.5, 42)
    cv.add_variant_data_agg("D", 10, 4.5, 42, replace=False)
    cv.add_variant_data_agg("D", 10, 4.5, 42, replace=True)
    cv.delete_variant("D")
    return cv


def test_variants(conv_test):
    assert conv_test.variant_names == ["A", "B", "C"]


def test_totals(conv_test):
    assert conv_test.totals == [10, 10, 11]


def test_sums(conv_test):
    assert conv_test.sums == [38, 41, 39]


def test_obs_means(conv_test):
    assert conv_test.obs_means == [3.8, 4.1, 4]


def test_means(conv_test):
    assert conv_test.means == [3.54545, 3.81818, 3.53846]


def test_stdevs(conv_test):
    assert conv_test.stdevs == [0.56773, 0.58916, 0.52172]


def test_bounds(conv_test):
    assert conv_test.bounds == [[2.66609, 4.52804], [2.90347, 4.83613], [2.72621, 4.43807]]


def test_a_priors(conv_test):
    assert conv_test.a_priors == [1, 1, 2]


def test_b_priors(conv_test):
    assert conv_test.b_priors == [1, 1, 2]


def test_probabs_of_being_best(conv_test):
    pbbs = conv_test._probabs_of_being_best(sim_count=2000000, seed=314)
    assert pbbs == {"A": 0.266836, "B": 0.480775, "C": 0.252389}


def test_expected_loss(conv_test):
    loss = conv_test._expected_loss(sim_count=2000000, seed=314)
    assert loss == {"A": 0.5896207, "B": 0.3169076, "C": 0.5965555}


def test_evaluate(conv_test):
    eval_report, _ = conv_test.evaluate(sim_count=2000000, seed=314)
    assert eval_report == [
        {'variant': 'A', 'total': 10, 'mean': 3.54545, 'prob_being_best': 0.266836, 'expected_loss': 0.5896207,
         'bounds': [2.66609, 4.52804]},
        {'variant': 'B', 'total': 10, 'mean': 3.81818, 'prob_being_best': 0.480775, 'expected_loss': 0.3169076,
         'bounds': [2.90347, 4.83613]},
        {'variant': 'C', 'total': 11, 'mean': 3.53846, 'prob_being_best': 0.252389, 'expected_loss': 0.5965555,
         'bounds': [2.72621, 4.43807]}]
