"""Experiment defaults from the original research notebooks."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """Store runtime sampling and figure-export options.

    Attributes:
        sample_size: Total sample count. Defaults to 100.
        n_trials: Monte Carlo trial count. Defaults to 10,000.
        random_state: Initial random seed. Defaults to 42.
        save_trees: Whether to export tree PNGs. Defaults to True.
        save_plots: Whether to export result plots. Defaults to True.

    Raises:
        ValueError: If sampling counts are nonpositive or trial seeds are out of range.
    """

    sample_size: int = 100
    n_trials: int = 10_000
    random_state: int = 42
    save_trees: bool = True
    save_plots: bool = True

    def __post_init__(self):
        """Validate positive sampling settings and the per-trial seed range.

        Returns:
            None: The frozen configuration remains unchanged.

        Raises:
            ValueError: If sample_size or n_trials is nonpositive, or trial seeds exceed the
                RandomState range.
        """
        if self.sample_size < 1 or self.n_trials < 1:
            raise ValueError("sample_size and n_trials must be positive integers")
        if not 0 <= self.random_state <= 2**32 - self.n_trials:
            raise ValueError("The seed and trial count must fit the RandomState seed range")
