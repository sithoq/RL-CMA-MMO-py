from scripts.run_f1_f20 import _parse_funcs, _resolve_funcs


def test_parse_funcs_keeps_existing_range_and_list_behavior():
    assert _parse_funcs("1:3") == [1, 2, 3]
    assert _parse_funcs("1,3,10") == [1, 3, 10]


def test_resolve_func_group_presets():
    assert _resolve_funcs("1:20", "core") == [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13]
    assert _resolve_funcs("1:20", "hard") == [14, 15, 16, 17, 18, 19, 20]
    assert _resolve_funcs("1:20", "high_peak") == [8, 9]
    assert _resolve_funcs("1:20", "focus") == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
    ]

