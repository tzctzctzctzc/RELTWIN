from pathlib import Path


FUNCTIONS = {
    "relation_objective",
    "rehearsal_objective",
    "token_kl_from_logits",
}


def test_lightweight_objectives_match_executed_training_source():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    training_source = (scripts / "train_reltwin_setpo.py").read_text(encoding="utf-8")
    assert "from setpo_objective import" in training_source
    assert all(name in training_source for name in FUNCTIONS)
