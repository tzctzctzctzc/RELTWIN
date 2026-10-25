import numpy as np

from analyze_oracle_headroom import metric_summary, oracle_summary


def test_oracle_takes_per_record_maximum_without_regressing_baseline():
    baseline = np.asarray([0.2, 0.6, 0.8])
    candidates = np.asarray([[0.4, 0.5, 0.8], [0.1, 0.9, 0.7]])
    summary, oracle = oracle_summary(
        ["first", "second", "baseline"],
        np.vstack([candidates, baseline]),
        baseline,
    )
    np.testing.assert_allclose(oracle, [0.4, 0.9, 0.8])
    assert summary["rows_improved_vs_baseline"] == 2
    assert summary["rows_regressed_vs_baseline"] == 0
    assert summary["winner_tie_rows"] == 1


def test_metric_summary_reports_percent_scale():
    summary = metric_summary(np.asarray([0.2, 0.5, 0.8, 1.0]))
    assert summary["mIoU_percent"] == 62.5
    assert summary["R1@0.3_percent"] == 75.0
    assert summary["R1@0.5_percent"] == 75.0
    assert summary["R1@0.7_percent"] == 50.0
