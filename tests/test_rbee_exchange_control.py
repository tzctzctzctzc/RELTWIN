"""Exercise the executed objective without importing the GPU/model stack."""
import ast
from pathlib import Path

import torch
import torch.nn.functional as F


def load_objective():
    path = Path(__file__).resolve().parents[1] / "scripts/train_reltwin_micro.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "outer_objective")
    namespace = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["outer_objective"]


def legacy_terms(scores):
    seq = -(scores[0] + scores[3]) / 2
    ab, ba = scores[:2], scores[2:]
    ce = (F.cross_entropy(ab[None], torch.tensor([0])) + F.cross_entropy(ba[None], torch.tensor([1]))) / 2
    p, q = F.softmax(ab, 0), F.softmax(ba, 0).flip(0)
    middle = (p + q) / 2
    js = (F.kl_div(middle.log(), p, reduction="sum") + F.kl_div(middle.log(), q, reduction="sum")) / 2
    return seq, ce, js


def test_default_preserves_legacy_value_and_gradient():
    scores = torch.tensor([-.4, -1.2, -.8, -.2], requires_grad=True)
    seq, ce, js = legacy_terms(scores)
    expected = seq + 1.3 * (ce + js)
    actual, _ = load_objective()("rbee", scores, 1., 1.3, .2)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.autograd.grad(actual, scores, retain_graph=True)[0], torch.autograd.grad(expected, scores)[0])


def test_zero_exchange_retains_exact_candidate_cross_entropy():
    scores = torch.tensor([-.4, -1.2, -.8, -.2], requires_grad=True)
    seq, ce, _ = legacy_terms(scores)
    actual, diagnostic = load_objective()("rbee", scores, 1., 1.3, .2, 0.)
    expected = seq + 1.3 * ce
    assert torch.equal(actual, expected)
    assert diagnostic["exchange_js"] > 0
    assert torch.equal(torch.autograd.grad(actual, scores, retain_graph=True)[0], torch.autograd.grad(expected, scores)[0])


def test_exchange_toggle_changes_only_js_term():
    scores = torch.tensor([-.3, -1.8, -.7, -.5], requires_grad=True)
    f = load_objective()
    full, _ = f("rbee", scores, 1., .7, .2, 1.)
    control, _ = f("rbee", scores, 1., .7, .2, 0.)
    assert torch.allclose(full-control, .7*legacy_terms(scores)[2])


def test_sft_is_independent_of_exchange_weight():
    s = torch.tensor([-.3, -1.8, -.7, -.5])
    f = load_objective()
    assert torch.equal(f("sft", s, 1., 1., .2, 0.)[0], f("sft", s, 1., 1., .2, 1.)[0])
