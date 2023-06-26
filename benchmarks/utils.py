from pynvml3.pynvml import NVMLLib


class GPUEnergyMeter:
    """A context manager based GPU energy meter."""

    def __init__(self, device_index=0) -> None:
        self.start = None
        self.end = None
        self.lib = NVMLLib()
        self.device_index = device_index
        self.device = None

    @property
    def energy(self) -> float:
        """Return the total amount of consumed energy in J."""
        return (self.end - self.start) / 1_000

    def __enter__(self):
        self.lib.open()
        self.device = self.lib.device[self.device_index]
        self.start = self.device.get_total_energy_consumption()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = self.device.get_total_energy_consumption()
        self.lib.close()


def get_gpu_name() -> str:
    """Return the name of the first GPU."""
    with NVMLLib() as lib:
        return lib.device[0].get_name()
