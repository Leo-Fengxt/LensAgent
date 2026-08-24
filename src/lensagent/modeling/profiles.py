"""Custom mass profiles used by fixed-count multisubhalo RSI."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.LensModel.Profiles.nfw import NFW


class MassConcentrationNFW(LensProfileBase):
    """NFW profile parameterized by halo mass and a fixed concentration law."""

    param_names: ClassVar[list[str]] = ["logM", "center_x", "center_y"]
    lower_limit_default: ClassVar[dict[str, float]] = {
        "logM": 7.0,
        "center_x": -100.0,
        "center_y": -100.0,
    }
    upper_limit_default: ClassVar[dict[str, float]] = {
        "logM": 11.0,
        "center_x": 100.0,
        "center_y": 100.0,
    }
    _lens_cosmo: LensCosmo | None = None
    _z_lens: float | None = None

    def __init__(self) -> None:
        self.nfw = NFW()
        super().__init__()

    @classmethod
    def configure(cls, z_lens: float, z_source: float) -> None:
        cosmology = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        cls._z_lens = float(z_lens)
        cls._lens_cosmo = LensCosmo(float(z_lens), float(z_source), cosmo=cosmology)

    @classmethod
    def angular_parameters(cls, log_mass: float) -> tuple[float, float]:
        if cls._lens_cosmo is None or cls._z_lens is None:
            raise RuntimeError("MassConcentrationNFW must be configured before use")
        mass = 10 ** np.clip(log_mass, 7.0, 11.0)
        concentration = (
            5.71 * (mass / 2.0e12) ** -0.084 * (1.0 + cls._z_lens) ** -0.47 * 4.0
        )
        return cls._lens_cosmo.nfw_physical2angle(M=mass, c=concentration)

    def function(self, x, y, logM, center_x=0.0, center_y=0.0):
        scale_radius, deflection = self.angular_parameters(logM)
        return self.nfw.function(x, y, scale_radius, deflection, center_x, center_y)

    def derivatives(self, x, y, logM, center_x=0.0, center_y=0.0):
        scale_radius, deflection = self.angular_parameters(logM)
        return self.nfw.derivatives(x, y, scale_radius, deflection, center_x, center_y)

    def hessian(self, x, y, logM, center_x=0.0, center_y=0.0):
        scale_radius, deflection = self.angular_parameters(logM)
        return self.nfw.hessian(x, y, scale_radius, deflection, center_x, center_y)


def register_mass_concentration_nfw(z_lens: float, z_source: float) -> None:
    """Register the mass-concentration profile as lenstronomy BLANK_PLANE."""
    from lenstronomy.LensModel.Profiles import blank_plane

    MassConcentrationNFW.configure(z_lens, z_source)
    blank_plane.BlankPlane = MassConcentrationNFW
