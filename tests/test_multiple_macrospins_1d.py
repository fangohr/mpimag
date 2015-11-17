"""
=========================================================================
Multiple Macrospins on 1D mesh Tests
=========================================================================

Tests for multiple macrospin on a 1d finite difference mesh.
Comparisons made with an analytical model.

The mesh nodes are assumed to be equally distributed along the length, x.
The mesh nodes are assumed to be located at:
    x0, x1,...,xn

The coordinate of the centres of the mesh cells are at:
    c0, c1, ..., cn-1

The magnetisation is defined at the centre of the mesh cells.

The magnetisation vector is assumed to be have the following structure:

    [[mx(c0), my(c0), mz(c0)], [mx(c1), my(c2), mz(c2)], ...]

Analytical solution found at:
    https://github.com/fangohr/micromagnetics/blob/master/testcases/macrospin/analytic_solution.py

Macrospin simulations is for 5ns, in time steps of 0.01ns.

The value of mx should be found at the times steps and stored. These
values of mx are then compared to analytical solutions computed at the
same times.

The magnetisation, m is dependent on the LLG equation:

    dm/dt = -g * (m x Heff) - g * a (m x m x Heff) / Ms

where:
    g - the gamma value
    a - alpha value
    Heff - effective field

The effective field, H, is only dependent on the Zeeman field, thus:
    Heff = Hzeeman
"""
import numpy as np
import pytest
from mpimag import Macrospin


def setup_macrospin():
    """
    Defines the simulation parameters.
    Creates a Macrospin simulation object
    Assigns the parameters to the Macrospin object
    Return the sim object
    """
    from mpimag import FDmesh1D

    # Define simulation parameters
    Ms = 8.6e5 # saturation magnetisation (A/m)
    alpha = 0.1 # Gilbert damping
    gamma = 2.211e5 # gyromagnetic ratio
    B = 0.1 # External magentic field (T)
    mu0 = 4*np.pi*1e-7 # vacuum permeability
    H = B/mu0
    m_init = [1, 0, 0]
    zeeman = [0, 0, H]

    # mesh parameters
    x0 = 0
    x1 = 10 # nm
    nx = 6 # number of nodes in x-dir
    mesh = FDmesh1D(x0, x1, nx)

    # setup sim object
    sim = Macrospin()
    sim.mesh = mesh
    sim.Ms = Ms
    sim.alpha = alpha
    sim.gamma = gamma
    sim.m = m_init
    sim.zeeman = zeeman

    return sim

@pytest.mark.xfail
def test_compare_with_analytical_sol(do_plot=False):
    import micromagnetictestcases

    # create pre-setup simulation object
    sim = setup_macrospin()

    # total simulation time
    t_total = 5e-9
    # time step for simulation
    t_step = 0.01e-9
    # Sampling time steps
    t_array = np.arange(0.0, t_total, t_step)

    # run simulation for specified time
    mx_computed = np.ndarray(len(t_array, len(sim.cells))) # sim.cells is number of cells
    for ti, t in enumerate(t_array):
        # run simulation until specified time
        sim.run_until(t)
        mx_computed[ti, :] = sim.m[:, 0] # [mx(x0), mx(x1), mx(x2),...]

    mx_analytic = micromagnetictestcases.macrospin.solution(sim.alpha, sim.gamma, sim.zeeman[2], t_array)

    if do_plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(t_array / 1e-9, mx_analytic, 'o', label='analytic')
        plt.plot(t_array / 1e-9, mx_computed[:, 0], linewidth=2, label='simulation')
        plt.xlabel('t (ns)')
        plt.ylabel('mx')
        plt.grid()
        plt.legend()
        plt.savefig('macrospin.pdf', format='pdf', bbox_inches='tight')

    # For each time step compare mx value from each cell with the analytical value.
    # Return true or false for each time step.
    compare_cells = [np.allclose(mx_computed[ti, :], mx) for ti, mx in enumerate(mx_analytic)]
    # Assert all values are equal at each time step are equal to the analytical value.
    assert np.all(compare_cells)

if __name__ == '__main__':
    test_compare_with_analytical_sol(do_plot=True)
