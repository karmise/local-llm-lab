"""Isolated pytest source templates; no application/model execution."""

from test_support.builders.quality_gates import measured_report
from test_support.data.quality_gates import GATES


def render_pytest_exit_code_makepyfile_source():
    return f"""
        from pathlib import Path
        from llm_testkit.reporting.gates import apply_quality_gates
        from llm_testkit.assertions import assert_quality_report
        def test_gate():
            report = {measured_report()!r}
            report['dimensions'][2]['details']['value'] = 0.0
            assert_quality_report(apply_quality_gates(report, Path({str(GATES)!r})))
    """
