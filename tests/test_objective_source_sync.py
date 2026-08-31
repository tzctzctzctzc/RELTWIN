import ast
from pathlib import Path


FUNCTIONS = {
    "listwise_cross_entropy",
    "relation_objective",
    "rehearsal_objective",
    "token_kl_from_logits",
}


def function_asts(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in FUNCTIONS
    }


def test_lightweight_objectives_match_executed_training_source():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    assert function_asts(scripts / "setpo_objective.py") == function_asts(
        scripts / "train_reltwin_setpo.py"
    )
