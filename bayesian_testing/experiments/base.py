from typing import Tuple
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

class BaseDataTest:
    """
    Base class for Bayesian A/B test.
    """

    def __init__(self) -> None:
        """
        Initialize BaseDataTest class.
        """
        self.data = {}
        self.samples = [[]]

    @property
    def variant_names(self):
        return [k for k in self.data]

    def eval_simulation(self, sim_count: int = 20000, seed: int = None) -> Tuple[dict, dict]:
        """
        Should be implemented in each individual experiment.
        """
        raise NotImplementedError

    def probabs_of_being_best(self, sim_count: int = 20000, seed: int = None) -> dict:
        """
        Calculate probabilities of being best for a current class state.

        Parameters
        ----------
        sim_count : Number of simulations to be used for probability estimation.
        seed : Random seed.

        Returns
        -------
        pbbs : Dictionary with probabilities of being best for all variants in experiment.
        """
        pbbs, loss = self.eval_simulation(sim_count, seed)

        return pbbs

    def expected_loss(self, sim_count: int = 20000, seed: int = None) -> dict:
        """
        Calculate expected loss for a current class state.

        Parameters
        ----------
        sim_count : Number of simulations to be used for probability estimation.
        seed : Random seed.

        Returns
        -------
        loss : Dictionary with expected loss for all variants in experiment.
        """
        pbbs, loss = self.eval_simulation(sim_count, seed)

        return loss

    def delete_variant(self, name: str) -> None:
        """
        Delete variant and all its data from experiment.

        Parameters
        ----------
        name : Variant name.
        """
        if not isinstance(name, str):
            raise ValueError("Variant name has to be a string.")
        if name not in self.variant_names:
            warnings.warn(f"Nothing to be deleted. Variant {name} is not in experiment.")
        else:
            del self.data[name]

    def plot_distributions(self, fname=None, dpi=300):
        num_bins = 750

        fig, ax = plt.subplots(figsize=(10, 8),)

        for s, v in zip(self.samples, self.variant_names):
            n, bins = np.histogram(s, num_bins)
            sigma = np.var(s)**0.5
            mu = np.mean(s)
            y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
                 np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))

            ax.plot(bins*100, y, label=f'{v}: $\mu={mu:.2%}$%')
            ax.fill_between(bins*100, y, alpha=0.35)

        ax.xaxis.set_major_formatter(mtick.PercentFormatter())

        ax.set_xlabel('Probability')
        ax.set_ylabel('Probability density')
        ax.legend()

        fig.tight_layout()

        if fname:
            plt.savefig(fname, dpi=dpi)

        plt.show()