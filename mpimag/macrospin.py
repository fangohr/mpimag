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
    """Calculates right hand side of the Landau-Lifshitz-Gilbert equation.

    The Landau-Lifshitz-Gilbert (LLG) equation is:

        dm/dt = -gamma * (m x Heff) - gamma * alpha * (m x m x Heff)

    The equation calculates the right hand side of this equation for a 0D or
    1D system.

    This function is used with in scipy.integrate.odeint and therefore
    flattened versions of the magnetisation and effective field vectors are
    required. Due to the use with odeint, the time, t is required as the
    second parameter. It is not actually used in the function though.

    In a 1D system, the total number of cells is ncells. As there are three
    magnetisation components, (mx, my, mz), the flattened arrays should be of
    length ncells * 3 e.g.
        [mx(x0), my(x0), mz(x0), ..., mx(x1), my(x1), mz(x1)]

    The function reshapes m_flat and heff_flat into the shape (ncells, 3) e.g.
        [[mx(x0), my(x0), mz(x0)], [...], [mx(x1), my(x1), mz(x1)]
    in order to calculate the cross products.

    Parameters
    ----------

    ==========================================================================
    Parameter   | Description                            | Usage
    ==========================================================================
    m_flat      | The flattened magnetisation vector e.g.| numpy array
                | in the form:                           | 
                | [mx(x0), my(x0), mz(x0), mx(x1), ...]  | 
    --------------------------------------------------------------------------
    t           | The time                               | int
    --------------------------------------------------------------------------
    heff_flat   | The flattened effective field vector   | numpy array
    --------------------------------------------------------------------------
    alpha       | Alpha parameter from LLG equation      | float
    --------------------------------------------------------------------------
    gamma       | Gamma parameter from the LLG equation  | float
    --------------------------------------------------------------------------

    Returns
    -------

    Returns a flattened version of the vector calculated from the right hand
    side.

    """
    # Reshape the magnetisation and effective field arrays
    m = m_flat.reshape(-1, 3)
    heff = heff_flat.reshape(-1, 3)
    # calculate the cross products
    mCrossH = np.cross(m, heff)
    mCrossmCrossH = np.cross(m, mCrossH)
    # calculate the rhs of the equation
    result = (-1 * gamma * mCrossH) - (gamma * alpha * mCrossmCrossH)
    return result.flatten()

class Macrospin(object):
    """
    Single Macrospin setup class
    """
    def __init__(self, mesh):

        # Setting up of process ranks
        self._comm = MPI.COMM_WORLD
        self._size = self._comm.size
        self._rank = self._comm.rank

        self._Ms = None
        self._alpha = None
        self._gamma = None
        self._zeeman = None
        self._m_local = None
        self._t = 0.0
        self.mesh = mesh

        if self.mesh.dims > 0:
            (self._ncells,
                self._ncells_local,
                self._ncells_locals) = self.mesh.ncells

    # -------------------------------------------------------------------
    # Ms property
    # -------------------------------------------------------------------
    def _set_Ms(self, Ms):
        """Set the Saturation magnetisation, Ms.

        Expects an int or float

        """
        self._Ms = Ms

    def _get_Ms(self):
        """Get or Set value of the Saturation magnetisation, Ms.

        Set: Expects an int or float
        Get: Returns an int or float

        """
        if self._Ms is None:
            raise AttributeError('Ms not yet set')
        return self._Ms

    Ms = property(_get_Ms, _set_Ms)

    # -------------------------------------------------------------------
    # alpha property
    # -------------------------------------------------------------------
    def _set_alpha(self, alpha):
        """Set the alpha value in the LLG equation.

        Expects an int or float

        """
        self._alpha = alpha

    def _get_alpha(self):
        """Get or Set value of the alpha value used in the LLG equation.

        Set: Expects an int or float
        Get: Returns an int or float

        """
        if self._alpha is None:
            raise AttributeError('Alpha not yet set')
        return self._alpha

    alpha = property(_get_alpha, _set_alpha)

    # -------------------------------------------------------------------
    # gamma property
    # -------------------------------------------------------------------
    def _set_gamma(self, gamma):
        """Set the gamma value in the LLG equation.

        Expects an int or float

        """
        self._gamma = gamma

    def _get_gamma(self):
        """Get or Set value of the gamma value used in the LLG equation.

        Set: Expects an int or float
        Get: Returns an int or float

        """
        if self._gamma is None:
            raise AttributeError('Gamma not yet set')
        return self._gamma

    gamma = property(_get_gamma, _set_gamma)

    # -------------------------------------------------------------------
    # zeeman property
    # -------------------------------------------------------------------
    def _set_zeeman(self, zeeman):
        """Sets the Zeeman field applied the simulation object

        Zeeman field is assumed to be uniform across the sample, applied in
        the (Hx, Hy, Hz).

        Function expects the values for (Hx, Hy, Hz) in a 1D list or array
        form of length 3.

        """
        if type(zeeman) not in [list, np.ndarray]:
            raise ValueError('Expecting a 3D list or array')
        if np.shape(zeeman) != (3,):
            raise ValueError('Expecting a zeeman in the form [Hx, Hy, Hz]\
                              Supplied value is not of this form.')
        self._zeeman = np.array(zeeman)

    def _get_zeeman(self):
        """Set or return current Zeeman field.

        Zeeman field is assumed to be uniform across the sample, applied in
        the (Hx, Hy, Hz).

        Set: function expects the values for (Hx, Hy, Hz) in a 1D list or array
             form of length 3.
        Get: returns a numpy array of the uniform Zeeman field, [Hx, Hy, Hz]

        """
        if self._zeeman is None:
            raise AttributeError('Zeeman field not yet set')
        return self._zeeman

    zeeman = property(_get_zeeman, _set_zeeman)

    # -------------------------------------------------------------------
    # m local property
    # -------------------------------------------------------------------
    def _v_normalise(self, v):
        """Normalise a vector, v, [vx, vy, vz]

        Takes a list or numpy array, v, of the form [vx, vy, vz] and returns a
        normalised v as an numpy array:

            [vx, vy, vz] / (vx**2 + vy**2 + vz**2)**0.5

        """
        norm = np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        return v / norm

    def _set_m_local(self, m):
        """Set the magnetisation of the part of the simulation object held on
        the particular process which calls the function

        Function to set the magnetisation of a simulation object with a
        uniform magnetisation [mx, my, mz], everywhere over the sample.

        Accepts a 1D list or numpy array of the form [mx, my, mz]

        The function will automatically normalise the supplied magnetisation.

        """
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
            self._m_local = np.ones((self._ncells_locals[self._rank], 3)) * \
                                m_normalised

    def _get_m_local(self):
        """Get or Set local magnetisation.

        Get or Set the magnetisation of the part of the simulation object
        held on the particular process which calls the function

        The magnetisation of a simulation object is set with a normalised
        uniform magnetisation, [mx, my, mz], everywhere over the sample.

        Set: Accepts a 1D list or numpy array of the form [mx, my, mz]
             The function will automatically normalise the supplied
             magnetisation.
        Get: Returns the magnetisation, [mx, my, mz] of the simuation object.
        
        """

        if self._m_local is None:
            raise AttributeError('magnetisation, m, not yet set')
        return self._m_local

    m_local = property(_get_m_local)


    # -------------------------------------------------------------------
    # m global property
    # -------------------------------------------------------------------

    def _get_m_global(self):
        """Gets the entire magnetisation array on process 0.

        Gathers the m_local arrays on process 0.

        Return the magnetisation array of the entire simulation object.
        """
        # Create array in which global m data is gathered into.
        # It is gathered onto process rank 0, thus this the array on this
        # process is initialised as the same length as the global number of
        # cells.
        # note only 1d array can be gathered, so data needs to be
        # flattened before sending
        # TODO: can actually gather in mutli dim arrays.

        if self._rank == 0:
            mGathered = np.empty(self._ncells * 3)
        else:
            mGathered = None
        self._comm.Barrier()
        # set up sendcounts tuple for comm.Gather. This is a tuple containing
        # the length of the array which each process send. Thus it is in the
        # format:
        # (ncells_process0 * 3, 
        #       ncells_process1  * 3, ..., ncell_processn-1 * 3)
        # * 3 due to there being 3 magnetisation components per
        # cell, mx, my, mz
        sendcounts = self._ncells_locals * 3

        # Set up the displacements tuple for comm.Gather. This is a tuple
        # containing the indicies where each set of local data should be
        # placed from in the global array (mGathered).
        displacements = np.concatenate(([0],
                            self._ncells_locals[0:-1].cumsum())) * 3

        self._comm.Barrier()
        # Gatherv is used instead of comm.gather as the length of _cells_local
        # is not neccessarily the same on each process. comm.gather only works
        # if the local data arrays being gathered is all the same length.
        self._comm.Gatherv(self._m_local.flatten(), [mGathered,
                                                     sendcounts,
                                                     displacements,
                                                     MPI.DOUBLE])
        self._comm.Barrier()
        if self._rank != 0:
            # TODO: would be good to raise an attribute error here when trying
            # to access global cell data from process other than 0th.
            mGathered = "Global cell data only available from process 0."
        else:
            # reshape m array
            mGathered.shape = (-1,3)
        return mGathered

    def _get_m(self):
        if self.mesh.dims == 0:
            return self._get_m_local()
        if self.mesh.dims == 1:
            return self._get_m_global()

    m = property(fget=_get_m, fset=_set_m_local)

    # -------------------------------------------------------------------
    # t property
    # -------------------------------------------------------------------
    def _set_t(self, t):
        # FINDME!
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
