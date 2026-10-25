from fit_nova_router import select, split_bucket


def test_official_fallback_respects_threshold():
    row = {"candidates": {"official": {}, "sft": {}, "setpo": {}}}
    assert select(row, {"official": 0.0, "sft": 0.1, "setpo": 0.2}, "official", 0.3) == "official"
    assert select(row, {"official": 0.0, "sft": 0.1, "setpo": 0.4}, "official", 0.3) == "setpo"


def test_inverse_relation_rows_share_a_split_bucket():
    ab = {"pair_id": 7, "template": "followed_by", "variant": 1, "relation": "AB"}
    ba = {"pair_id": 7, "template": "followed_by", "variant": 1, "relation": "BA"}
    assert split_bucket(ab) == split_bucket(ba)
