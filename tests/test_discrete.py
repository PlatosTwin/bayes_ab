import pytest

from bayes_ab.experiments import DiscreteDataTest


@pytest.fixture
def discrete_test():
    disc = DiscreteDataTest(states=[1, 2, 3, 4, 5, 6])
    disc.add_variant_data("A", [6, 5, 4, 4, 4, 2, 5, 4, 2, 1, 2, 5, 4, 6, 2, 3, 6, 2, 3, 6])
    disc.add_variant_data("B", [4, 6, 3, 6, 4, 6, 6, 1, 4, 1])
    disc.add_variant_data_agg("C", [10, 10, 10, 10, 10, 10], prior=[100, 100, 100, 100, 100, 100])
    disc.add_variant_data_agg("D", [1, 2, 3, 8, 10, 7])
    disc.add_variant_data_agg("D", [1, 2, 3, 8, 10, 6], replace=False)
    disc.add_variant_data_agg("D", [1, 2, 3, 8, 10, 6], replace=True)
    disc.delete_variant("D")
    return disc


def test_variants(discrete_test):
    assert discrete_test.variant_names == ["A", "B", "C"]


def test_states(discrete_test):
    assert discrete_test.states == [1, 2, 3, 4, 5, 6]


def test_concentrations(discrete_test):
    assert discrete_test.concentrations == [
        [1, 5, 2, 5, 3, 4],
        [2, 0, 1, 3, 0, 4],
        [10, 10, 10, 10, 10, 10],
    ]


def test_means(discrete_test):
    assert discrete_test.means == [3.73077, 3.875, 3.5]


def test_rel_probs(discrete_test):
    assert discrete_test.rel_probs == [
        [0.07692, 0.23077, 0.11538, 0.23077, 0.15385, 0.19231],
        [0.1875, 0.0625, 0.125, 0.25, 0.0625, 0.3125],
        [0.16667, 0.16667, 0.16667, 0.16667, 0.16667, 0.16667],
    ]


def test_bounds(discrete_test):
    with pytest.raises(RuntimeError):
        discrete_test.bounds


def test_rel_bounds(discrete_test):
    with pytest.raises(RuntimeError):
        discrete_test.rel_bounds


def test_probabs_of_being_best(discrete_test):
    pbbs = discrete_test._probabs_of_being_best(sim_count=20000, seed=52)
    assert pbbs == {"A": 0.35595, "B": 0.59325, "C": 0.0508}


def test_expected_loss(discrete_test):
    loss = discrete_test._expected_loss(sim_count=20000, seed=52)
    assert loss == {"A": 0.3053921, "B": 0.1560257, "C": 0.5328904}


def test_evaluate(discrete_test):
    eval_report = discrete_test.evaluate(sim_count=20000, seed=52)
    assert eval_report == [
        {
            "bounds": [3.1227, 4.32656],
            "concentration": {1: 1.0, 2: 5.0, 3: 2.0, 4: 5.0, 5: 3.0, 6: 4.0},
            "expected_loss": 0.3053921,
            "posterior_mean": 3.73077,
            "prob_being_best": 0.35595,
            "rel_bounds": [
                [0.00965, 0.19964],
                [0.09375, 0.40892],
                [0.02483, 0.25741],
                [0.09588, 0.40786],
                [0.04539, 0.31101],
                [0.06775, 0.36172],
            ],
            "rel_probs": ["7.69%", "23.08%", "11.54%", "23.08%", "15.38%", "19.23%"],
            "sample_mean": 3.8,
            "uplift_vs_a": 0,
            "variant": "A",
        },
        {
            "bounds": [2.98262, 4.70942],
            "concentration": {1: 2.0, 2: 0.0, 3: 1.0, 4: 3.0, 5: 0.0, 6: 4.0},
            "expected_loss": 0.1560257,
            "posterior_mean": 3.875,
            "prob_being_best": 0.59325,
            "rel_bounds": [
                [0.04363, 0.40149],
                [0.00154, 0.21871],
                [0.01624, 0.3191],
                [0.07854, 0.47686],
                [0.00172, 0.21535],
                [0.11836, 0.55104],
            ],
            "rel_probs": ["18.75%", "6.25%", "12.50%", "25.00%", "6.25%", "31.25%"],
            "sample_mean": 4.1,
            "uplift_vs_a": 0.03866,
            "variant": "B",
        },
        {
            "bounds": [3.3681, 3.63023],
            "concentration": {1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10},
            "expected_loss": 0.5328904,
            "posterior_mean": 3.5,
            "prob_being_best": 0.0508,
            "rel_bounds": [
                [0.13894, 0.19546],
                [0.13944, 0.19631],
                [0.13937, 0.19642],
                [0.13959, 0.19565],
                [0.13918, 0.1965],
                [0.1394, 0.19595],
            ],
            "rel_probs": ["16.67%", "16.67%", "16.67%", "16.67%", "16.67%", "16.67%"],
            "sample_mean": 3.5,
            "uplift_vs_a": -0.06186,
            "variant": "C",
        },
    ]


def test_non_numerical_states_error():
    with pytest.raises(ValueError):
        DiscreteDataTest(states=[1, 2.0, "3"])


def test_non_string_variant_error(discrete_test):
    with pytest.raises(ValueError):
        discrete_test.add_variant_data_agg(1, [1, 2, 3, 8, 10, 7])


def test_length_mismatch_input_error(discrete_test):
    with pytest.raises(ValueError):
        discrete_test.add_variant_data_agg("D", [1, 2, 3, 8, 10])


def test_empty_data_error(discrete_test):
    with pytest.raises(ValueError):
        discrete_test.add_variant_data("D", [])


def test_non_existing_state_error(discrete_test):
    with pytest.raises(ValueError):
        discrete_test.add_variant_data("D", [1, 2, 3, 5, 21])


@pytest.mark.mpl_image_compare
def test_discrete_plot_distributions(discrete_test):
    discrete_test.evaluate(sim_count=20000, seed=52)
    fig = discrete_test.plot_distributions()
    return fig
