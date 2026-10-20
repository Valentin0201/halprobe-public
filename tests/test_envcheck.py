import pytest

from halprobe.device import resolve_device
from halprobe.envcheck import kernel_checks, main, tiny_cpu_check


def test_tiny_cpu_check_passes():
    assert tiny_cpu_check()


def test_main_cpu_mode_returns_zero():
    assert main(["--cpu"]) == 0


@pytest.mark.gpu
def test_kernel_checks_on_gpu():
    assert kernel_checks(resolve_device("auto", "6700"))
