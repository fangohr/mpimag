"""
=========================================================================
Single Macrospin Setup
=========================================================================

Setup of a for the single Macrospin.

The Macrospin object has the following properties associated with it:

=========================================================================
Property    | Description                       | Usage
=========================================================================
Ms          | The saturation magnetisation, Ms  | int or float
            | of the Macrospin                  |
-------------------------------------------------------------------------
alpha       | The alpha value of the Macrospin  | int or float
-------------------------------------------------------------------------
gamma       | The gamma value of the Macrospin  | int or float
-------------------------------------------------------------------------
zeeman      | The Zeeman field applied to the   | list or np.array
            | Macrospin (3D vector)             | [Hx, Hy, Hz]
-------------------------------------------------------------------------
m           | The m field Macrospin (3D vector) | list or np.array
            |                                   | [mx, my, mz]
-------------------------------------------------------------------------
t           | The time, t associated with the   | int or float
            | Macrospin. Must be positive.      |
=========================================================================


"""
import numpy as np
import scipy
from scipy import integrate

def llg(m, t, heff, alpha, gamma):
    # Computing dmdt
    # First compute the cross products
    mCrossH = np.cross(m, heff)
    mCrossmCrossH = np.cross(m, mCrossH)
    return (-1 * gamma * mCrossH) - (gamma * alpha * mCrossmCrossH)

class Macrospin(object):
    """
    Single Macrospin setup class
    """
    def __init__(self):
        self._Ms = None
        self._alpha = None
        self._gamma = None
        self._zeeman = None
        self._m = None
        self.t = 0.0

    # -------------------------------------------------------------------
    # Ms property
    # -------------------------------------------------------------------
    def _set_Ms(self, Ms):
        self._Ms = Ms

    def _get_Ms(self):
        """
        Set the value of Ms of the single Macrospin. Expects an int
        or float
        """
        if self._Ms is None:
            raise AttributeError('Ms not yet set')
        return self._Ms

    Ms = property(_get_Ms, _set_Ms)

    # -------------------------------------------------------------------
    # alpha property
    # -------------------------------------------------------------------
    def _set_alpha(self, alpha):
        self._alpha = alpha

    def _get_alpha(self):
        """
        Set the value of alpha of the single Macrospin. Expects an int
        or float
        """
        if self._alpha is None:
            raise AttributeError('Alpha not yet set')
        return self._alpha

    alpha = property(_get_alpha, _set_alpha)

    # -------------------------------------------------------------------
    # gamma property
    # -------------------------------------------------------------------
    def _set_gamma(self, gamma):
        self._gamma = gamma

    def _get_gamma(self):
        """
        Set the value of gamma of the single Macrospin. Expects an int
        or float
        """
        if self._gamma is None:
            raise AttributeError('Gamma not yet set')
        return self._gamma

    gamma = property(_get_gamma, _set_gamma)

    # -------------------------------------------------------------------
    # zeeman property
    # -------------------------------------------------------------------
    def _set_zeeman(self, zeeman):
        if type(zeeman) not in [list, np.ndarray]:
            raise ValueError('Expecting a 3D list or array')
        if np.shape(zeeman) != (3,):
            raise ValueError('Expecting a zeeman in the form [Hx, Hy, Hz]\
                              Supplied value is not of this form.')
        self._zeeman = np.array(zeeman)

    def _get_zeeman(self):
        """
        Set the Zeeman field interaction on the single Macrospin. Expects
        a 3D list in the form [Hx, Hy, Hz].
        """
        if self._zeeman is None:
            raise AttributeError('Zeeman field not yet set')
        return self._zeeman

    zeeman = property(_get_zeeman, _set_zeeman)

    # -------------------------------------------------------------------
    # m property
    # -------------------------------------------------------------------
    def _v_normalise(self, v):
        """
        Takes a list v, of the form [vx, vy, vz] and returns a normalised
        list of v:

            [vx, vy, vz] / (vx**2 + vy**2 + vz**2)**0.5

        """
        norm = np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        return v / norm
        # return [v[0] / norm, v[1] / norm, v[2] / norm]

    def _set_m(self, m):
        if type(m) not in [list, np.ndarray]:
            raise ValueError('Expecting a 3D list')
        if np.shape(m) != (3,):
            raise ValueError('Expecting a list of for [mx, my, mz]\
                              List supplied is not of this\
                              form.')
        self._m = self._v_normalise(m)

    def _get_m(self):
        """
        Set the magnetisation, m, of the single Macrospin. Expects
        a 3D list in the form [mx, my, mz].
        """
        if self._m is None:
            raise AttributeError('magnetisation, m, not yet set')
        return self._m

    m = property(_get_m, _set_m)

    # -------------------------------------------------------------------
    # t property
    # -------------------------------------------------------------------
    def _set_t(self, t):
        # check if t is greater than or equal to zero. If not, raise value
        # error
        if t < 0.0:
            raise ValueError('Attempting to set t with negative value.\
                              t must be postitive.')
        self._t = t

    def _get_t(self):
        """
        Set the time, t asscoiated with the single Macrospin. Expects a
        postitive int or float.
        """
        return self._t

    t = property(_get_t, _set_t)

    # -------------------------------------------------------------------
    # Effective field
    # -------------------------------------------------------------------
    def _compute_heff(self):
        """
        Computing the Effective field
        """
        self._heff = self._zeeman

    # -------------------------------------------------------------------
    # Solve LLG
    # -------------------------------------------------------------------

    def _compute_llg(self, t):
        """
        Compute the llg equation
        """
        # Compute the effective field
        self._compute_heff()
        # compute m
        m  = integrate.odeint(llg, self._m, [self._t, t], args=(self._heff, self._alpha, self._gamma))
        return m

    # -------------------------------------------------------------------
    # run until function
    # -------------------------------------------------------------------
    def run_until(self, t):
        """
        Run the simulation until the time, t.

        Update the value of m accordingly.
        """
        m  = self._compute_llg(t)
        self._m = [m[-1][0], m[-1][1], m[-1][2]]
        self._t = t
