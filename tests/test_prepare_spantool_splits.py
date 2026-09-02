from prepare_spantool_splits import row_stratum, stratified_gate, train_dev_split


def rows(count=100):
    result = []
    for index in range(count):
        cardinality = 1 if index % 3 else 2
        annotations = [[1.0, 2.0]] * cardinality
        result.append(
            {
                "audio_path": f"audio_{index // 2}.wav",
                "caption": f"query {index}",
                "annotations": annotations,
                "duration": 10.0 if index % 2 else 60.0,
                "qid": index,
            }
        )
    return result


def test_train_dev_split_is_audio_disjoint_and_deterministic():
    source = rows()
    train_a, dev_a = train_dev_split(source, 0.2, 7)
    train_b, dev_b = train_dev_split(source, 0.2, 7)
    assert [row["qid"] for row in train_a] == [row["qid"] for row in train_b]
    assert [row["qid"] for row in dev_a] == [row["qid"] for row in dev_b]
    assert not ({row["audio_path"] for row in train_a} & {row["audio_path"] for row in dev_a})


def test_stratified_gate_uses_full_manifest_not_prefix():
    source = rows(200)
    gate = stratified_gate(source, 60, 11)
    assert len(gate) == 60
    assert len({row["qid"] for row in gate}) == 60
    assert max(row["qid"] for row in gate) > 100
    source_strata = {row_stratum(row) for row in source}
    gate_strata = {row_stratum(row) for row in gate}
    assert gate_strata == source_strata
