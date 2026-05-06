import pytest
from dev.tests.support.network_builders import build_network_object

from source.io import partition_network


def test_partition_network_rejects_nonpositive_chunks():
    network = build_network_object(vertices=[[0.0, 0.0, 0.0]], edges=[], radii=[])

    with pytest.raises(ValueError, match="chunks must contain positive"):
        partition_network(network, (0, 1))
