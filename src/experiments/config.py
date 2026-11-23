"""Experiment defaults from the original research notebooks."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """Only runtime options vary; model settings remain in the experiment modules."""

    sample_size: int = 100
    n_trials: int = 10_000
    random_state: int = 42
    save_trees: bool = True
    save_plots: bool = True

    def __post_init__(self):
        if self.sample_size < 1 or self.n_trials < 1:
            raise ValueError("sample_size and n_trials must be positive integers")
        if not 0 <= self.random_state <= 2**32 - self.n_trials:
            raise ValueError("The seed and trial count must fit the RandomState seed range")
