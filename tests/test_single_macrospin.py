"""
=========================================================================
Single Macrospin Tests
=========================================================================

Tests for a single macrospin. Comparisons made with an analytical model.

Analytical solution found at:
    https://github.com/fangohr/micromagnetics/blob/master/testcases/macrospin/analytic_solution.py

Macrospin is simulations for 5ns, in time steps of 0.01ns.

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
from mpimag import FDmesh0D


def setup_macrospin():
    """
    Defines the simulation parameters.
    Creates a Macrospin simulation object
    Assigns the parameters to the Macrospin object
    Return the sim object
    """
    # Define simulation parameters
    Ms = 8.6e5 # saturation magnetisation (A/m)
    alpha = 0.1 # Gilbert damping
    gamma = 2.211e5 # gyromagnetic ratio
    B = 0.1 # External magentic field (T)
    mu0 = 4*np.pi*1e-7 # vacuum permeability
    H = B/mu0
    m_init = [1, 0, 0]
    zeeman = [0, 0, H]
    mesh = FDmesh0D()

    sim = Macrospin(mesh)
    sim.Ms = Ms
    sim.alpha = alpha
    sim.gamma = gamma
    sim.m = m_init
    sim.zeeman = zeeman

    return sim

# @pytest.mark.xfail
def test_run_until_zero_seconds():
    import micromagnetictestcases

    # create pre-setup simulation object
    sim = setup_macrospin()

    t_array = np.array([0.])

    # run simulation for specified time
    mx_computed = np.ndarray(len(t_array))
    for i, t in enumerate(t_array):
        # run simulation until specified time
        sim.run_until(t)
        mx_computed[i] = sim.m[0]

    mx_analytic = micromagnetictestcases.macrospin.solution(sim.alpha, sim.gamma, sim.zeeman[2], t_array)

    assert (mx_analytic == mx_computed)

# @pytest.mark.xfail
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
    mx_computed = np.ndarray(len(t_array))
    for i, t in enumerate(t_array):
        # run simulation until specified time
        sim.run_until(t)
        mx_computed[i] = sim.m[0]

    mx_analytic = micromagnetictestcases.macrospin.solution(sim.alpha, sim.gamma, sim.zeeman[2], t_array)

    if do_plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(t_array / 1e-9, mx_analytic, 'o', label='analytic')
        plt.plot(t_array / 1e-9, mx_computed, linewidth=2, label='simulation')
        plt.xlabel('t (ns)')
        plt.ylabel('mx')
        plt.grid()
        plt.legend()
        plt.savefig('macrospin.pdf', format='pdf', bbox_inches='tight')

    assert np.allclose(mx_analytic, mx_computed, rtol=1e-05, atol=1e-05)

if __name__ == '__main__':
    test_compare_with_analytical_sol(do_plot=True)
