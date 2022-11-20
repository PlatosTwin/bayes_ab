import pytest

from bayes_ab.experiments import NormalDataTest


@pytest.fixture
def norm_test():
    norm = NormalDataTest()
    norm.add_variant_data(
        "A",
        [
            11.8,
            12.2,
            12.4,
            9.5,
            2.2,
            3.3,
            16.2,
            4.9,
            12.4,
            6.8,
            8.7,
            9.8,
            5.4,
            9.0,
            15.0,
            12.3,
            9.6,
            12.5,
            9.1,
            10.2,
        ],
        m_prior=9,
    )
    norm.add_variant_data(
        "B",
        [
            10.6,
            5.1,
            9.4,
            11.2,
            2.0,
            13.4,
            14.1,
            15.4,
            16.3,
            11.7,
            7.3,
            6.8,
            8.2,
            16.2,
            10.8,
            7.1,
            12.2,
            11.2,
        ],
        n_prior=0.03,
    )
    norm.add_variant_data(
        "C",
        [
            25.3,
            10.3,
            24.7,
            -8.1,
            8.4,
            10.3,
            14.8,
            13.4,
            11.5,
            -4.7,
            5.3,
            7.4,
            17.2,
            15.4,
            13.0,
            12.9,
            19.2,
            11.6,
            0.4,
            5.7,
            23.5,
            15.2,
        ],
        s_2_prior=2,
    )
    norm.add_variant_data_agg("A", 20, 193.3, 259.4655, 2127.71, replace=False)
    norm.add_variant_data("D", [0, 10.7, 0, 8, 0, 0, 0, 0, 0, 11.22])
    norm.add_variant_data("D", [0, 10.7, 0, 8, 0, 0, 0, 0, 0, 11.22], replace=False)
    norm.add_variant_data("D", [0, 10.7, 0, 8, 0, 0, 0, 0, 0, 11.22], replace=True)
    norm.delete_variant("D")
    return norm


def test_variants(norm_test):
    assert norm_test.variant_names == ["A", "B", "C"]


def test_totals(norm_test):
    assert norm_test.totals == [40, 18, 22]


def test_sum_values(norm_test):
    assert norm_test.sum_values == [386.6, 189.0, 252.7]


def test_sum_values_squared(norm_test):
    assert norm_test.sum_values_squared == [4255.42, 2244.82, 4421.87]


def test_sum_squares(norm_test):
    assert norm_test.sum_squares == [518.931, 260.32, 1519.26591]


def test_m_priors(norm_test):
    assert norm_test.m_priors == [9, 1, 1]


def test_v_priors(norm_test):
    assert norm_test.v_priors == [0, 0, 0]


def test_s_2_priors(norm_test):
    assert norm_test.s_2_priors == [0, 0, 2]


def test_n_priors(norm_test):
    assert norm_test.n_priors == [0.01, 0.03, 0.01]


def test_means(norm_test):
    assert norm_test.means == [9.66283, 10.48419, 11.4816]


def test_bounds(norm_test):
    assert norm_test.bounds == [[9.00655, 10.31912], [8.7815, 12.18688], [4.97001, 17.99319]]


def test_precisions(norm_test):
    assert norm_test.precisions == [0.07697, 0.06844, 0.01447]


def test_stdevs(norm_test):
    assert norm_test.stdevs == [3.60445, 3.82262, 8.31309]


def test_stdev_bounds(norm_test):
    assert norm_test.stdev_bounds == [[2.9593, 4.6119], [2.88842, 5.65298], [6.4293, 11.76595]]


def test_probabs_of_being_best(norm_test):
    pbbs = norm_test._probabs_of_being_best(sim_count=20000, seed=52)
    assert pbbs == {"A": 0.0, "B": 0.01485, "C": 0.98515}


def test_expected_loss(norm_test):
    loss = norm_test._expected_loss(sim_count=20000, seed=52)
    assert loss == {"A": 1.8212161, "B": 1.0033285, "C": 0.0028303}


@pytest.mark.mpl_image_compare
def test_normal_plot_distributions(norm_test):
    norm_test.evaluate(sim_count=20000, seed=52)
    fig = norm_test.plot_distributions(control="A")
    return fig


@pytest.mark.mpl_image_compare
def test_normal_plot_joint_prior(norm_test):
    norm_test.evaluate(sim_count=20000, seed=52)
    fig = norm_test.plot_joint_prior(variant="A")
    return fig


def test_evaluate(norm_test):
    eval_report = norm_test.evaluate(sim_count=20000, seed=52)
    assert eval_report == [
        {
            "variant": "A",
            "total": 40,
            "mean": 9.66283,
            "prob_being_best": 0.0,
            "expected_loss": 1.8212161,
            "uplift_vs_a": 0,
            "bounds": [9.00655, 10.31912],
            "precision": 0.07697,
            "stdev": 3.60445,
            "stdev_bounds": [2.9593, 4.6119],
        },
        {
            "variant": "B",
            "total": 18,
            "mean": 10.48419,
            "prob_being_best": 0.01485,
            "expected_loss": 1.0033285,
            "uplift_vs_a": 0.085,
            "bounds": [8.7815, 12.18688],
            "precision": 0.06844,
            "stdev": 3.82262,
            "stdev_bounds": [2.88842, 5.65298],
        },
        {
            "variant": "C",
            "total": 22,
            "mean": 11.4816,
            "prob_being_best": 0.98515,
            "expected_loss": 0.0028303,
            "uplift_vs_a": 0.18822,
            "bounds": [4.97001, 17.99319],
            "precision": 0.01447,
            "stdev": 8.31309,
            "stdev_bounds": [6.4293, 11.76595],
        },
    ]
