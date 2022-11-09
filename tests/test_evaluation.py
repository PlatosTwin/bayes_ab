import numpy as np
import pytest

from bayes_ab.metrics import (
    eval_bernoulli_agg,
    eval_normal_agg,
    eval_delta_lognormal_agg,
    eval_numerical_dirichlet_agg,
    eval_poisson_agg,
)

PBB_BERNOULLI_AGG_INPUTS = [
    {
        "input": {
            "totals": [31500, 32000, 31000],
            "successes": [1580, 1700, 1550],
            "sim_count": 20000,
            "seed": 52,
        },
        "expected_output": ([0.04195, 0.92215, 0.0359], [0.0030135, 6.07e-05, 0.0031644]),
    },
    {
        "input": {
            "totals": [100, 200],
            "successes": [80, 160],
            "sim_count": 10000,
            "seed": 52,
        },
        "expected_output": ([0.4775, 0.5225], [0.0212003, 0.0175862]),
    },
    {
        "input": {
            "totals": [100, 100],
            "successes": [0, 0],
            "sim_count": 20000,
            "seed": 52,
        },
        "expected_output": ([0.4935, 0.5065], [0.0049983, 0.0048362]),
    },
    {
        "input": {
            "totals": [100],
            "successes": [77],
            "sim_count": 20000,
            "seed": 52,
        },
        "expected_output": ([1.0], [0.0]),
    },
    {
        "input": {
            "totals": [],
            "successes": [],
            "sim_count": 20000,
            "seed": 52,
        },
        "expected_output": ([], []),
    },
]

PBB_NORMAL_AGG_INPUTS = [
    {
        "input": {
            "totals": [31000, 30000, 32000],
            "sums": [33669.629254438274, 32451.58924937506, 34745.69678322253],
            "sums_2": [659657.6891070933, 95284.82070196551, 260327.13931832163],
            "sim_count": 20000,
            "seed": 52,
        },
        "expected_output": ([0.43605, 0.19685, 0.3671], [0.0133512, 0.0179947, 0.0137618]),
    },
    {
        "input": {
            "totals": [10000, 10000],
            "sums": [11446.345516947431, 10708.892428298526],
            "sums_2": [214614.35949718487, 31368.55305547222],
            "sim_count": 20000,
            "seed": 52,
        },
        "expected_output": ([0.94445, 0.05555], [0.0011338, 0.0753121]),
    },
    {
        "input": {
            "totals": [10, 20, 30, 40],
            "sums": [0, 0, 0, 0],
            "sums_2": [0, 0, 0, 0],
            "sim_count": 20000,
            "seed": 52,
        },
        "expected_output": (
            [0.40785, 0.25105, 0.1928, 0.1483],
            [0.0058965, 0.0065083, 0.0066249, 0.0067183],
        ),
    },
    {
        "input": {
            "totals": [100],
            "sums": [0],
            "sums_2": [0],
            "sim_count": 10000,
            "seed": 52,
        },
        "expected_output": ([1], [0]),
    },
    {
        "input": {
            "totals": [10000, 10000],
            "sums": [11446.35, 11446.35],
            "sums_2": [214614.36, 214614.36],
            "sim_count": 20000,
            "seed": 52,
        },
        "expected_output": ([0.5024, 0.4976], [0.0250157, 0.0256253]),
    },
    {
        "input": {
            "totals": [],
            "sums": [],
            "sums_2": [],
            "sim_count": 10000,
            "seed": 52,
        },
        "expected_output": ([], []),
    },
]

PBB_DELTA_LOGNORMAL_AGG_INPUTS = [
    {
        "input": {
            "totals": [31500, 32000, 31000],
            "successes": [1580, 1700, 1550],
            "sum_logs": [3831.806394737816, 4211.72986767986, 4055.965234848171],
            "sum_logs_2": [11029.923165846496, 12259.51868396913, 12357.911862914],
            "sim_count": 20000,
            "seed": 52,
        },
        "expected_output": ([0.00015, 0.03345, 0.9664], [0.2210276, 0.1206176, 0.0008447]),
    },
    {
        "input": {
            "totals": [31000, 31000],
            "successes": [1550, 1550],
            "sum_logs": [4055.965234848171, 4055.965234848171],
            "sum_logs_2": [12357.911862914, 12357.911862914],
            "sim_count": 10000,
            "seed": 52,
        },
        "expected_output": ([0.5014, 0.4986], [0.0281956, 0.0287299]),
    },
    {
        "input": {
            "totals": [10, 20, 30, 40],
            "successes": [0, 0, 0, 0],
            "sum_logs": [0, 0, 0, 0],
            "sum_logs_2": [0, 0, 0, 0],
            "sim_count": 10000,
            "seed": 52,
        },
        "expected_output": ([0.25, 0.25, 0.25, 0.25], [np.nan, np.nan, np.nan, np.nan]),
    },
    {
        "input": {
            "totals": [100],
            "successes": [10],
            "sum_logs": [0],
            "sum_logs_2": [0],
            "sim_count": 10000,
            "seed": 52,
        },
        "expected_output": ([1], [0]),
    },
    {
        "input": {
            "totals": [],
            "successes": [],
            "sum_logs": [],
            "sum_logs_2": [],
            "sim_count": 10000,
            "seed": 52,
        },
        "expected_output": ([], []),
    },
]

PBB_NUMERICAL_DIRICHLET_AGG_INPUTS = [
    {
        "input": {
            "states": [1, 2, 3, 4, 5, 6],
            "concentrations": [
                [10, 10, 10, 10, 20, 10],
                [10, 10, 10, 10, 10, 20],
                [10, 10, 10, 20, 10, 10],
            ],
            "sim_count": 20000,
            "seed": 52,
        },
        "expected_output": ([0.28205, 0.62335, 0.0946], [0.1999528, 0.0698306, 0.334045]),
    },
    {
        "input": {
            "states": [1, 2, 3],
            "concentrations": [[100, 100, 100]],
            "sim_count": 20000,
            "seed": 52,
        },
        "expected_output": ([1], [0]),
    },
    {
        "input": {
            "states": [],
            "concentrations": [],
            "sim_count": 20000,
            "seed": 52,
        },
        "expected_output": ([], []),
    },
]

PBB_POISSON_AGG_INPUTS = [
    {
        "input": {
            "totals": [1400, 1000, 2000],
            "mean": [41.99857142857143, 40.034, 42.9405],
            "obs_sum": [58798, 40034, 85881],
            "a_prior": [1, 1, 1],
            "b_prior": [1, 1, 1],
            "sim_count": 20000,
            "seed": 314,
        },
        "expected_output": ([0.0, 0.0, 1.0], [0.9492442, 2.9237474, 0.0]),
    },
    {
        "input": {
            "totals": [200, 350],
            "mean": [98.26, 99.70285714285714],
            "obs_sum": [19652, 34896],
            "a_prior": [1, 1],
            "b_prior": [1, 1],
            "sim_count": 20000,
            "seed": 314,
        },
        "expected_output": ([0.02855, 0.97145], [1.6558778, 0.0095961]),
    },
    {
        "input": {
            "totals": [250, 270, 100],
            "mean": [0, 0, 0],
            "obs_sum": [0, 0, 0],
            "a_prior": [1, 1, 1],
            "b_prior": [1, 1, 1],
            "sim_count": 20000,
            "seed": 314,
        },
        "expected_output": ([0.2132, 0.1919, 0.5949], [0.0077535, 0.0080318, 0.0018684]),
    },
    {
        "input": {
            "totals": [25],
            "mean": [4.64],
            "obs_sum": [116],
            "a_prior": [1],
            "b_prior": [1],
            "sim_count": 20000,
            "seed": 314,
        },
        "expected_output": ([1.0], [0.0]),
    },
    {
        "input": {
            "totals": [],
            "mean": [],
            "obs_sum": [],
            "a_prior": [],
            "b_prior": [],
            "sim_count": 20000,
            "seed": 314,
        },
        "expected_output": ([], []),
    },
]


@pytest.mark.parametrize("inp", PBB_BERNOULLI_AGG_INPUTS)
def test_eval_bernoulli_agg(inp):
    i = inp["input"]
    res_a, res_b, _ = eval_bernoulli_agg(i["totals"], i["successes"], sim_count=i["sim_count"], seed=i["seed"])
    assert res_a == inp["expected_output"][0] and res_b == inp["expected_output"][1]


@pytest.mark.parametrize("inp", PBB_NORMAL_AGG_INPUTS)
def test_eval_normal_agg(inp):
    i = inp["input"]
    res_a, res_b, _ = eval_normal_agg(
        i["totals"],
        i["sums"],
        i["sums_2"],
        sim_count=i["sim_count"],
        seed=i["seed"],
    )
    assert res_a == inp["expected_output"][0] and res_b == inp["expected_output"][1]


def test_eval_normal_agg_different_runs():
    # two different runs of same input without seed should be different
    run1 = eval_normal_agg([100, 100], [10, 10], [20, 20])
    run2 = eval_normal_agg([100, 100], [10, 10], [20, 20])
    assert run1 != run2


@pytest.mark.parametrize("inp", PBB_DELTA_LOGNORMAL_AGG_INPUTS)
def test_eval_delta_lognormal_agg(inp):
    i = inp["input"]
    res_a, res_b, _ = eval_delta_lognormal_agg(
        i["totals"],
        i["successes"],
        i["sum_logs"],
        i["sum_logs_2"],
        sim_count=i["sim_count"],
        seed=i["seed"],
    )
    assert res_a == inp["expected_output"][0] and res_b == inp["expected_output"][1]


def test_eval_delta_lognormal_agg_different_runs():
    # two different runs of same input without seed should be different
    run1 = eval_delta_lognormal_agg([1000, 1000], [100, 100], [10, 10], [20, 20], sim_count=100000)
    run2 = eval_delta_lognormal_agg([1000, 1000], [100, 100], [10, 10], [20, 20], sim_count=100000)
    assert run1 != run2


@pytest.mark.parametrize("inp", PBB_NUMERICAL_DIRICHLET_AGG_INPUTS)
def test_eval_numerical_dirichlet_agg(inp):
    i = inp["input"]
    res_a, res_b, _ = eval_numerical_dirichlet_agg(
        i["states"], i["concentrations"], sim_count=i["sim_count"], seed=i["seed"]
    )
    assert res_a == inp["expected_output"][0] and res_b == inp["expected_output"][1]


def test_eval_numerical_dirichlet_agg_different_runs():
    # two different runs of same input without seed should be different
    run1 = eval_numerical_dirichlet_agg([1, 20], [[10, 10], [20, 20]])
    run2 = eval_numerical_dirichlet_agg([1, 20], [[10, 10], [20, 20]])
    assert run1 != run2


@pytest.mark.parametrize("inp", PBB_POISSON_AGG_INPUTS)
def test_eval_poisson_agg(inp):
    i = inp["input"]
    res_a, res_b, _ = eval_poisson_agg(
        i["totals"], i["mean"], i["a_prior"], i["b_prior"], sim_count=i["sim_count"], seed=i["seed"]
    )
    assert res_a == inp["expected_output"][0] and res_b == inp["expected_output"][1]


def test_eval_poisson_agg_different_runs():
    # two different runs of same input without seed should be different
    run1 = eval_poisson_agg([100, 100], [10, 10], [20, 20])
    run2 = eval_poisson_agg([100, 100], [10, 10], [20, 20])
    assert run1 != run2
