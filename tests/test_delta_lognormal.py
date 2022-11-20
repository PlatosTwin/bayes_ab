import pytest

from bayes_ab.experiments import DeltaLognormalDataTest


@pytest.fixture
def rev_test():
    rev = DeltaLognormalDataTest()
    rev.add_variant_data_agg("A", 31500, 1580, 30830.02561, 3831.806394737816, 11029.923165846496, a_prior_beta=1)
    rev.add_variant_data_agg(
        "B", 32000, 1700, 35203.21689, 4211.72986767986, 12259.51868396913, m_prior=2, w_prior=0.02
    )
    rev.add_variant_data_agg(
        "C",
        31000,
        1550,
        37259.56336,
        4055.965234848171,
        12357.911862914,
        a_prior_ig=1,
        b_prior_ig=2,
    )
    rev.add_variant_data("D", [0, 10.7, 0, 8, 0, 0, 0, 0, 0, 11.22])
    rev.add_variant_data("D", [0, 10.7, 0, 8, 0, 0, 0, 0, 0, 11.22], replace=False)
    rev.add_variant_data("D", [0, 10.7, 0, 8, 0, 0, 0, 0, 0, 11.22], replace=True)
    rev.delete_variant("D")
    return rev


def test_variants(rev_test):
    assert rev_test.variant_names == ["A", "B", "C"]


def test_totals(rev_test):
    assert rev_test.totals == [31500, 32000, 31000]


def test_positives(rev_test):
    assert rev_test.positives == [1580, 1700, 1550]


def test_sum_values(rev_test):
    assert rev_test.sum_values == [30830.02561, 35203.21689, 37259.56336]


def test_sum_logs(rev_test):
    assert [round(i, 5) for i in rev_test.sum_logs] == [3831.80639, 4211.72987, 4055.96523]


def test_sum_logs_2(rev_test):
    assert [round(i, 5) for i in rev_test.sum_logs_2] == [11029.92317, 12259.51868, 12357.91186]


def test_a_priors_beta(rev_test):
    assert rev_test.a_priors_beta == [1, 1, 1]


def test_b_priors_beta(rev_test):
    assert rev_test.b_priors_beta == [1, 1, 1]


def test_m_priors(rev_test):
    assert rev_test.m_priors == [1, 2, 1]


def test_a_priors_ig(rev_test):
    assert rev_test.a_priors_ig == [0, 0, 1]


def test_b_priors_ig(rev_test):
    assert rev_test.b_priors_ig == [0, 0, 2]


def test_w_priors(rev_test):
    assert rev_test.w_priors == [0.01, 0.02, 0.01]


def test_probabs_of_being_best(rev_test):
    pbbs = rev_test._probabs_of_being_best(sim_count=2000000, seed=314)
    assert pbbs == {"A": 0.000167, "B": 0.008403, "C": 0.99143}


def test_expected_loss(rev_test):
    loss = rev_test._expected_loss(sim_count=2000000, seed=314)
    assert loss == {"A": 18.5688331, "B": 13.8235806, "C": 0.0160803}


def test_evaluate(rev_test):
    eval_report = rev_test.evaluate(sim_count=2000000, seed=314)
    assert eval_report == [
        {
            "variant": "A",
            "totals": 31500,
            "positives": 1580,
            "sum_values": 30830.02561,
            "avg_values": 0.97873,
            "avg_positive_values": 19.51267,
            "prob_being_best": 0.000167,
            "expected_loss": 18.5688331,
        },
        {
            "variant": "B",
            "totals": 32000,
            "positives": 1700,
            "sum_values": 35203.21689,
            "avg_values": 1.1001,
            "avg_positive_values": 20.70777,
            "prob_being_best": 0.008403,
            "expected_loss": 13.8235806,
        },
        {
            "variant": "C",
            "totals": 31000,
            "positives": 1550,
            "sum_values": 37259.56336,
            "avg_values": 1.20192,
            "avg_positive_values": 24.03843,
            "prob_being_best": 0.99143,
            "expected_loss": 0.0160803,
        },
    ]
