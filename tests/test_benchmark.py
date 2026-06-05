import pytest

np = pytest.importorskip("numpy")

from rlmmo.benchmarks.cec2013 import CEC2013Problem


def test_cec2013_f1_metadata_and_eval():
    problem = CEC2013Problem(1)
    assert problem.dim == 1
    assert problem.expected_peaks == 2
    x = np.array([(problem.lb[0] + problem.ub[0]) / 2.0])
    val = problem.evaluate(x)
    assert isinstance(val, float)
