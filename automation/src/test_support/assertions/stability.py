"""Domain expectations for stability unit scenarios."""


def check_golden_summary_outcome(rows, same_case):
    if same_case:
        assert len(rows) == 1
        assert rows[0]["configuration_consistent"] is False
    else:
        assert len(rows) == 2
        assert all(row["runs"] == 1 for row in rows)
        assert all(not row["mixed_pass_fail_observed"] for row in rows)


def check_conversation_summary_outcome(change, rows):
    if change == "different-cases":
        assert len(rows) == 2
        assert all(row["runs"] == 1 and not row["mixed_pass_fail_observed"] for row in rows)
    else:
        assert len(rows) == 1
        assert rows[0]["configuration_consistent"] is False
        assert rows[0]["metadata_complete"] is (change == "changed-catalog")
