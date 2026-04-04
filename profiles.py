import numpy as np
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.LensModel.Profiles.nfw import NFW


class fR_NFW(LensProfileBase):
    """f(R)-modified NFW lens profile for dark matter halos.

    Extends the standard NFW deflection with a fifth-force correction
    parameterized by fR0.  A sigmoid screening function centered at the
    scale radius Rs approximates chameleon screening: the fifth force is
    suppressed at r << Rs and unsuppressed at r >> Rs.  The 1/3 prefactor
    is the theoretical maximum force enhancement in f(R) gravity.
    """
    param_names = ['Rs', 'alpha_Rs', 'fR0', 'center_x', 'center_y']
    lower_limit_default = {
        'Rs': 0, 'alpha_Rs': 0, 'fR0': 1e-15,
        'center_x': -100, 'center_y': -100,
    }
    upper_limit_default = {
        'Rs': 100, 'alpha_Rs': 10, 'fR0': 1e-2,
        'center_x': 100, 'center_y': 100,
    }

    def __init__(self):
        self.nfw = NFW()
        super(fR_NFW, self).__init__()

    def function(self, x, y, Rs, alpha_Rs, fR0, center_x=0, center_y=0):
        return 0

    def derivatives(self, x, y, Rs, alpha_Rs, fR0, center_x=0, center_y=0):
        f_x_nfw, f_y_nfw = self.nfw.derivatives(
            x, y, Rs, alpha_Rs, center_x, center_y)
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        fifth_force = self._calculate_fifth_force_gradient(r, Rs, fR0)
        f_x_fr = (1./3.) * fifth_force * (x - center_x) / (r + 1e-9)
        f_y_fr = (1./3.) * fifth_force * (y - center_y) / (r + 1e-9)
        return f_x_nfw + f_x_fr, f_y_nfw + f_y_fr

    def _calculate_fifth_force_gradient(self, r, Rs, fR0):
        enhancement_factor = 1 / (1 + np.exp(-2.0 * (r - Rs)))
        return fR0 * enhancement_factor

    def hessian(self, x, y, Rs, alpha_Rs, fR0, center_x=0, center_y=0):
        return self.nfw.hessian(x, y, Rs, alpha_Rs, center_x, center_y)

    def density(self, r, **kwargs):
        return self.nfw.density(r, kwargs['Rs'], kwargs['alpha_Rs'])

    def mass_3d_lens(self, r, **kwargs):
        return self.nfw.mass_3d(r, kwargs['Rs'], kwargs['alpha_Rs'])


_custom_profiles_registered = False


def setup_custom_profiles():
    """Register fR_NFW as the BLANK_PLANE profile in lenstronomy.

    Safe to call multiple times; the monkey-patch is applied only once.
    """
    global _custom_profiles_registered
    if _custom_profiles_registered:
        return
    from lenstronomy.LensModel.Profiles import blank_plane
    blank_plane.BlankPlane = fR_NFW
    _custom_profiles_registered = True
