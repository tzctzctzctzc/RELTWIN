from build_reltwin_ordinary import derive_event_spans


def test_derive_event_spans_from_relation_windows():
    row = {
        "window_ab": [[1.5, 11.75]],
        "window_ba": [[14.75, 25.0]],
    }
    assert derive_event_spans(row, 5.0, 5.0) == {
        "A": [[1.5, 6.5], [20.0, 25.0]],
        "B": [[6.75, 11.75], [14.75, 19.75]],
    }
