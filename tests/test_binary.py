import pytest

from bayes_ab.experiments import BinaryDataTest


@pytest.fixture
def conv_test():
    cv = BinaryDataTest()
    cv.add_variant_data("A", [0, 1, 0, 1, 0, 0, 0, 0, 0, 1])
    cv.add_variant_data("B", [0, 0, 0, 1, 0, 0, 0, 0, 0, 1])
    cv.add_variant_data_agg("C", 11, 2, a_prior=1, b_prior=2)
    cv.add_variant_data_agg("D", 10, 10)
    cv.add_variant_data_agg("D", 20, 20, replace=False)
    cv.add_variant_data_agg("D", 20, 20, replace=True)
    cv.delete_variant("D")
    return cv


def test_variants(conv_test):
    assert conv_test.variant_names == ["A", "B", "C"]


def test_totals(conv_test):
    assert conv_test.totals == [10, 10, 11]


def test_positives(conv_test):
    assert conv_test.positives == [3, 2, 2]


def test_a_priors(conv_test):
    assert conv_test.a_priors == [1, 1, 1]


def test_b_priors(conv_test):
    assert conv_test.b_priors == [1, 1, 2]


def test_means(conv_test):
    assert conv_test.means == [0.33333, 0.25, 0.21429]


def test_stdevs(conv_test):
    assert conv_test.stdevs == [0.13074, 0.1201, 0.10595]


def test_bounds(conv_test):
    assert conv_test.bounds == [[0.13508, 0.56437], [0.07882, 0.47009], [0.06605, 0.4101]]


def test_probabs_of_being_best(conv_test):
    pbbs = conv_test._probabs_of_being_best(sim_count=2000000, seed=314)
    assert pbbs == {"A": 0.58086, "B": 0.259051, "C": 0.160089}


def test_expected_loss(conv_test):
    loss = conv_test._expected_loss(sim_count=2000000, seed=314)
    assert loss == {"A": 0.0509394, "B": 0.1340909, "C": 0.1701915}


def test_evaluate(conv_test):
    eval_report, _ = conv_test.evaluate(sim_count=2000000, seed=314)
    assert eval_report == [
        {'variant': 'A', 'total': 10, 'positives': 3, 'positive_rate': 0.33333, 'prob_being_best': 0.58086,
         'expected_loss': 0.0509394, 'uplift_vs_a': 0, 'bounds': [0.13508, 0.56437]},
        {'variant': 'B', 'total': 10, 'positives': 2, 'positive_rate': 0.25, 'prob_being_best': 0.259051,
         'expected_loss': 0.1340909, 'uplift_vs_a': -0.24999, 'bounds': [0.07882, 0.47009]},
        {'variant': 'C', 'total': 11, 'positives': 2, 'positive_rate': 0.21429, 'prob_being_best': 0.160089,
         'expected_loss': 0.1701915, 'uplift_vs_a': -0.35712, 'bounds': [0.06605, 0.4101]}]
