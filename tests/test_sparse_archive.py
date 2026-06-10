import numpy as np

from rlmmo.core.archive import PeakArchive


def test_sparse_quality_trim_keeps_spatially_distant_candidate():
    archive = PeakArchive(radius=0.001, max_size=3, trim_mode="sparse_quality", novelty_weight=2.0)

    archive.add_or_update(np.array([0.00]), 10.0)
    archive.add_or_update(np.array([0.02]), 9.0)
    archive.add_or_update(np.array([0.04]), 8.0)
    archive.add_or_update(np.array([1.00]), 1.0)

    pop, fit = archive.as_arrays()

    assert len(archive) == 3
    assert np.max(pop[:, 0]) > 0.9
    assert np.min(fit) == 1.0
