from __future__ import annotations

from pathlib import Path

from source.analysis.ml_curator_io import materialize_model_source


class _UploadedModel:
    def __init__(self, path: Path):
        self.name = path.name
        self._payload = path.read_bytes()

    def getvalue(self):
        return self._payload


def test_materialize_model_source_supports_uploaded_file_objects(tmp_path):
    source_path = tmp_path / "model.joblib"
    source_path.write_bytes(b"model-bytes")

    with materialize_model_source(_UploadedModel(source_path)) as materialized_path:
        assert Path(materialized_path).read_bytes() == b"model-bytes"
