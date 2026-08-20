from scripts.aggregate_results import build_final_rows


def test_final_table_has_fixed_rows_and_hides_unsupported_metrics():
    rows = build_final_rows({
        "qwen2.5-vl-7b/base/vidhalluc": {"bqa": {"accuracy": 0.99}}
    })
    assert len(rows) == 12
    qwen_base = next(row for row in rows if row["Models"] == "Qwen2.5-VL-7B - Base")
    assert qwen_base["VidHalluc_BQA"] == "N/A"
    assert all(row["Training-free"] == "Yes" for row in rows)
