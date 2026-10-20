import halprobe


def test_package_version():
    assert halprobe.__version__ == "0.1.0"


def test_torch_is_the_pinned_build():
    import torch

    assert torch.__version__.startswith("2.9.1")
