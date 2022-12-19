from prettytable import PrettyTable
import numpy as np


def print_dirichlet_evaluation(res: list, states: list) -> None:
    """
    Pretty-print output of running standard Dirichlet test.
    """

    tab = PrettyTable()
    tab.field_names = [
        "Variant",
        "Concentrations",
        "Sample mean",
        "Posterior mean",
        "Relative prob.",
        "95% HDI (relative prob.)",
        "Chance to beat all",
        "Expected loss",
        'Uplift vs. "A"',
        "95% HDI (mean)",
    ]
    for r in res:
        temp_row = r.copy()
        for i in ["prob_being_best", "expected_loss", "uplift_vs_a"]:
            temp_row[i] = f"{temp_row[i]:.2%}"
        for i, rp in enumerate(r["rel_probs"]):
            temp_row["rel_probs"][i] = f"{rp:.2%}"

        temp_row["rel_probs"] = dict(zip(states, temp_row["rel_probs"]))
        relative_prob_str = ""
        for key, value in temp_row["rel_probs"].items():
            relative_prob_str += f"{key}: {value}, "
        relative_prob_str.strip(",")

        concentration_str = ""
        for i, (key, value) in enumerate(temp_row["concentration"].items()):
            concentration_str += f"{key}: {int(value)}, "

        temp_row["rel_bounds"] = dict(zip(states, np.array(temp_row["rel_bounds"]).T))
        rel_bounds_str = ""
        for i, (key, value) in enumerate(temp_row["rel_bounds"].items()):
            rel_bounds_str += f"{key}: [{value[0]:.2%}, {value[1]:.2%}], "

        temp_row = [
            temp_row["variant"],
            concentration_str.strip(", "),
            round(temp_row["sample_mean"], 2),
            round(temp_row["posterior_mean"], 2),
            relative_prob_str.strip(", "),
            rel_bounds_str.strip(", "),
            temp_row["prob_being_best"],
            temp_row["expected_loss"],
            temp_row["uplift_vs_a"],
            f'[{temp_row["bounds"][0]:.2f}, {temp_row["bounds"][1]:.2f}]',
        ]

        tab.add_row(temp_row)

    tab.reversesort = True
    tab.sortby = "Chance to beat all"

    print(tab, "\n")


def print_normal_evaluation(res: list) -> None:
    """
    Pretty-print output of running normal test.
    """
    tab = PrettyTable()
    tab.field_names = [
        "Variant",
        "Observations",
        "Sample mean",
        "Posterior mean",
        "Precision",
        "Std. dev.",
        "Chance to beat all",
        "Expected loss",
        'Uplift vs. "A"',
        "95% HDI (mean)",
        "95% HDI (stdev)",
    ]
    for r in res:
        temp_row = r.copy()
        temp_row["prob_being_best"] = f"{temp_row['prob_being_best']:.2%}"
        temp_row["uplift_vs_a"] = f"{temp_row['uplift_vs_a']:.2%}"
        temp_row = [
            temp_row["variant"],
            temp_row["total"],
            round(temp_row["obs_mean"], 2),
            round(temp_row["mean"], 2),
            round(temp_row["precision"], 3),
            round(temp_row["stdev"], 2),
            temp_row["prob_being_best"],
            round(temp_row["expected_loss"], 2),
            temp_row["uplift_vs_a"],
            f'[{temp_row["bounds"][0]:.2f}, {temp_row["bounds"][1]:.2f}]',
            f'[{temp_row["stdev_bounds"][0]:.2f}, {temp_row["stdev_bounds"][1]:.2f}]',
        ]

        tab.add_row(temp_row)

    tab.reversesort = True
    tab.sortby = "Chance to beat all"

    print(tab, "\n")


def print_poisson_evaluation(res: list) -> None:
    """
    Pretty-print output of running Poisson test.
    """
    tab = PrettyTable()
    tab.field_names = [
        "Variant",
        "Observations",
        "Sample mean",
        "Posterior mean",
        "Chance to beat all",
        "Expected loss",
        'Uplift vs. "A"',
        "95% HDI",
    ]
    for r in res:
        temp_row = r.copy()
        temp_row["prob_being_best"] = f"{temp_row['prob_being_best']:.2%}"
        temp_row["uplift_vs_a"] = f"{temp_row['uplift_vs_a']:.2%}"
        temp_row["expected_loss"] = round(temp_row["expected_loss"], 2)
        temp_row["obs_mean"] = round(temp_row["obs_mean"], 1)
        temp_row["mean"] = round(temp_row["mean"], 1)
        temp_row = [
            temp_row["variant"],
            temp_row["total"],
            temp_row["obs_mean"],
            temp_row["mean"],
            temp_row["prob_being_best"],
            temp_row["expected_loss"],
            temp_row["uplift_vs_a"],
            f'[{temp_row["bounds"][0]:.1f}, {temp_row["bounds"][1]:.1f}]',
        ]

        tab.add_row(temp_row)

    tab.reversesort = True
    tab.sortby = "Chance to beat all"

    print(tab, "\n")


def print_bernoulli_evaluation(res: list) -> None:
    """
    Pretty-print output of running standard binary test.
    """
    tab = PrettyTable()
    tab.field_names = [
        "Variant",
        "Totals",
        "Positives",
        "Sample rate",
        "Posterior rate",
        "Chance to beat all",
        "Expected loss",
        'Uplift vs. "A"',
        "95% HDI",
    ]
    for r in res:
        temp_row = r.copy()
        for i in ["sample_positive_rate", "positive_rate", "prob_being_best", "expected_loss", "uplift_vs_a"]:
            temp_row[i] = f"{temp_row[i]:.2%}"
        temp_row = [
            temp_row["variant"],
            temp_row["total"],
            temp_row["positives"],
            temp_row["sample_positive_rate"],
            temp_row["positive_rate"],
            temp_row["prob_being_best"],
            temp_row["expected_loss"],
            temp_row["uplift_vs_a"],
            f'[{temp_row["bounds"][0]:.2%}, {temp_row["bounds"][1]:.2%}]',
        ]

        tab.add_row(temp_row)

    tab.reversesort = True
    tab.sortby = "Chance to beat all"

    print(tab, "\n")


def print_closed_form_comparison(variants: list, pbbs: list, cf_pbbs: list) -> None:
    """
    Pretty-print output comparing the estimate to the exact chance to beat all.
    """
    tab = PrettyTable()
    tab.field_names = ["Variant", "Est. chance to beat all", "Exact chance to beat all", "Delta"]
    for var, est, cf in zip(variants, pbbs, cf_pbbs):
        tab.add_row([var, f"{est:.2%}", f"{cf:.2%}", f"{(est - cf) / cf:.2%}"])

    tab.reversesort = True
    tab.sortby = "Est. chance to beat all"

    print(tab, "\n")
