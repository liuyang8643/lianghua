"""The sole evaluation protocol has no staged or train-only CLI switch."""
import pytest
from ai.rl.train import build_parser

@pytest.mark.parametrize("flags", [["--holdout-mode", "formal"], ["--holdout-mode", "three-split-diagnostic"], ["--train-only"]])
def test_removed_evaluation_modes_are_rejected(flags):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--runtime", "unused", *flags])

def test_periodic_defaults():
    args = build_parser().parse_args(["--runtime", "unused"])
    assert args.eval_every_rollouts == 50
    assert args.learning_rate == 3e-4
    assert args.n_steps == 64 and args.batch_size == 640 and args.n_epochs == 3
