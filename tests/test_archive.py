import numpy as np

from rlmmo.core.archive import PeakArchive


def test_archive_add_update_and_deduplicate():
    archive = PeakArchive(radius=0.5, max_size=3)
    assert archive.add_or_update(np.array([0.0, 0.0]), 1.0) == 1.0
    assert archive.add_or_update(np.array([0.1, 0.0]), 0.5) == 0.0
    assert archive.add_or_update(np.array([0.1, 0.0]), 2.0) == 0.25
    assert archive.add_or_update(np.array([2.0, 0.0]), 1.5) == 1.0
    pop, fit = archive.as_arrays()
    assert pop.shape == (2, 2)
    assert np.max(fit) == 2.0
