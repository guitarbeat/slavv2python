import source.visualization as visualization


def test_visualization_wildcard_import_exports_only_defined_names():
    namespace = {}

    exec("from source.visualization import *", namespace, namespace)

    assert "NetworkVisualizer" in namespace
    assert all(hasattr(visualization, name) for name in visualization.__all__)
