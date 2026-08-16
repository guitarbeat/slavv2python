"""Policy: MKL spike cannot replace Approach A / mark stretch complete (U7)."""

from __future__ import annotations

from slavv_python.analytics.parity.proof.stretch import mkl_spike_cannot_complete_stretch


def test_mkl_pass_does_not_mean_stretch_complete() -> None:
    assert mkl_spike_cannot_complete_stretch(mkl_bit_equal=True) is True
    assert mkl_spike_cannot_complete_stretch(mkl_bit_equal=False) is True
