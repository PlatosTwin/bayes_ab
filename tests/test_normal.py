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


@pytest.fixture
def norm_test_plotting():
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
        n_prior=10,
        s_2_prior=4,
        v_prior=9,
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
        n_prior=5,
        m_prior=5,
        s_2_prior=10,
        v_prior=4,
    )
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
    assert norm_test.m_priors == [9, 0, 0]


def test_v_priors(norm_test):
    assert norm_test.v_priors == [-1, -1, -1]


def test_s_2_priors(norm_test):
    assert norm_test.s_2_priors == [0, 0, 2]


def test_n_priors(norm_test):
    assert norm_test.n_priors == [0, 0.03, 0]


def test_means(norm_test):
    assert norm_test.means == [9.665, 10.48253, 11.48636]


def test_bounds(norm_test):
    assert norm_test.bounds == [[8.99216, 10.33784], [8.66793, 12.29713], [4.65665, 18.31608]]


def test_precisions(norm_test):
    assert norm_test.precisions == [0.07515, 0.06449, 0.01384]


def test_stdevs(norm_test):
    assert norm_test.stdevs == [3.64773, 3.93792, 8.50004]


def test_stdev_bounds(norm_test):
    assert norm_test.stdev_bounds == [[2.98808, 4.68381], [2.95496, 5.9035], [6.53952, 12.14711]]


def test_probabs_of_being_best(norm_test):
    pbbs = norm_test._probabs_of_being_best(sim_count=20000, seed=52)
    assert pbbs == {"A": 0.0, "B": 0.01535, "C": 0.98465}


def test_expected_loss(norm_test):
    loss = norm_test._expected_loss(sim_count=20000, seed=52)
    assert loss == {"A": 1.8258233, "B": 1.0101269, "C": 0.0031627}


@pytest.mark.mpl_image_compare
def test_normal_plot_distributions(norm_test_plotting):
    norm_test_plotting.evaluate(sim_count=20000, seed=52)
    fig = norm_test_plotting.plot_distributions(control="A")
    return fig


@pytest.mark.mpl_image_compare
def test_normal_plot_joint_prior(norm_test_plotting):
    norm_test_plotting.evaluate(sim_count=20000, seed=52)
    fig = norm_test_plotting.plot_joint_prior(variant="A")
    return fig


def test_evaluate(norm_test):
    eval_report = norm_test.evaluate(sim_count=20000, seed=52)
    assert eval_report == [
        {
            "variant": "A",
            "total": 40,
            "mean": 9.665,
            "prob_being_best": 0.0,
            "expected_loss": 1.8258233,
            "uplift_vs_a": 0,
            "bounds": [8.99216, 10.33784],
            "precision": 0.07515,
            "stdev": 3.64773,
            "stdev_bounds": [2.98808, 4.68381],
        },
        {
            "variant": "B",
            "total": 18,
            "mean": 10.48253,
            "prob_being_best": 0.01535,
            "expected_loss": 1.0101269,
            "uplift_vs_a": 0.08459,
            "bounds": [8.66793, 12.29713],
            "precision": 0.06449,
            "stdev": 3.93792,
            "stdev_bounds": [2.95496, 5.9035],
        },
        {
            "variant": "C",
            "total": 22,
            "mean": 11.48636,
            "prob_being_best": 0.98465,
            "expected_loss": 0.0031627,
            "uplift_vs_a": 0.18845,
            "bounds": [4.65665, 18.31608],
            "precision": 0.01384,
            "stdev": 8.50004,
            "stdev_bounds": [6.53952, 12.14711],
        },
    ]
