import numpy as np
import pytest

from slavv.io import Network, partition_network


def test_partition_network_rejects_nonpositive_chunks():
    network = Network(
        vertices=np.array([[0.0, 0.0, 0.0]], dtype=float),
        edges=np.empty((0, 2), dtype=int),
    )

    with pytest.raises(ValueError, match="chunks must contain positive"):
        partition_network(network, (0, 1))
