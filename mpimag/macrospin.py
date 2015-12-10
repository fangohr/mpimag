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
from mpi4py import MPI

def llg(m_flat, t, heff_flat, alpha, gamma):
    """
    Computing the LLG equation.

    Expects flattened arrays for m and heff. E.g. an array in the form:
        [m(x0), m(y0), m(z0), m(x1), m(y1), m(z1), ...]

    returns a flattened array

    """
    # Computing dmdt
    # First compute the cross products
    m = m_flat.reshape(-1, 3)
    heff = heff_flat.reshape(-1, 3)
    mCrossH = np.cross(m, heff)
    mCrossmCrossH = np.cross(m, mCrossH)
    result = (-1 * gamma * mCrossH) - (gamma * alpha * mCrossmCrossH)
    return result.flatten()

class Macrospin(object):
    """
    Single Macrospin setup class
    """
    def __init__(self, mesh):

        # Setting up of process ranks
        self._comm = MPI.COMM_WORLD#comm
        self._size = self._comm.size
        self._rank = self._comm.rank

        self._Ms = None
        self._alpha = None
        self._gamma = None
        self._zeeman = None
        self._m_local = None
        self._t = 0.0
        self.mesh = mesh

        self._ncells, self._ncells_local, self._ncells_local_full, self._ncells_local_partial = self.mesh.ncells

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
    # m local property
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

    def _set_m_local(self, m):
        if type(m) not in [list, np.ndarray]:
            raise ValueError('Expecting a 3D list')
        if np.shape(m) != (3,):
            raise ValueError('Expecting a zeeman in the form [mx, my, mz]\
                              Supplied value is not of this form.')

        m = np.array(m)
        if self.mesh.dims == 0:
            self._m_local = self._v_normalise(m)
        if self.mesh.dims == 1:
            m_normalised = self._v_normalise(m)
            self._m_local = np.ones((self._ncells_local, 3)) * m_normalised

    def _get_m_local(self):
        """
        Set the magnetisation, m, of the single Macrospin. Expects
        a 3D list in the form [mx, my, mz].
        """
        if self._m_local is None:
            raise AttributeError('magnetisation, m, not yet set')
        return self._m_local

    m_local = property(_get_m_local)#, _set_m_local)


    # -------------------------------------------------------------------
    # m global property
    # -------------------------------------------------------------------

    def _get_m_global(self):
        """
        Gathers m_local arrays on process 0
        """
        # Create array in which global m data is gathered into.
        # It is gathered onto process rank 0, thus this the array
        # on this process is initialised as the same length as the
        # global number of cells
        # note only 1d array can be gathered, so data needs to be
        # flattened before sending

        if self._rank == 0:
            mGathered = np.empty(self._ncells * 3)
        else:
            mGathered = None
        self._comm.Barrier()
        # set up sendcounts tuple for comm.Gather. This is a tuple containing
        # the length of the array which each process send. Thus it is in the format
        # (ncells_process0, ncells_process1, ..., ncell_processn-1)
        sendcounts = tuple([self._ncells_local_full * 3] * (self._size - 1) + [self._ncells_local_partial * 3])
        # Set up the displacements tuple for comm.Gather. This is a tuple containing
        # the indicies where each set of local data should be placed from in the global
        # array (cellsGathered).
        displacements = tuple([rank * self._ncells_local_full * 3 for rank in range(self._size)])
        self._comm.Barrier()
        # Gatherv is used instead of comm.gather as the length of _cells_local
        # is not neccessarily the same on each process. comm.gather only works if
        # the local data arrays being gathered is all the same legnth.
        self._comm.Gatherv(self._m_local.flatten(), [mGathered, sendcounts, displacements, MPI.DOUBLE])#self._comm.gather(self._cells_local.tolist(), cells, root=0)
        self._comm.Barrier()
        # return self._cellsGathered
        if self._rank != 0:
            # TODO: would be good to raise an attribute error here when trying to access
            # global cell data from process other than 0th.
            mGathered = "Global cell data only available from process 0."
        else:
            # reshape m array
            mGathered.shape = (-1,3)
        return mGathered

    m = property(fget=_get_m_global, fset=_set_m_local)

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
        if self.mesh.dims == 0:
            self._heff = self._zeeman
        if self.mesh.dims == 1:
            self._heff = self._zeeman * np.ones((self._ncells_local, 3))
        

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
        m_flat = integrate.odeint(llg, self._m_local.flatten(), [self._t, t],
                                args=(self._heff.flatten(), self._alpha, self._gamma))
        if self.mesh.dims == 0:
            m = m_flat[1]
        if self.mesh.dims == 1:
            m = m_flat[1].reshape(-1, 3)
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
        self._m_local = m
        self._t = t
