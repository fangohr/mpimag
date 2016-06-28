"""
=========================================================================
Exchange tests
=========================================================================

Tests for the computation of the exchange field on a 1D finite difference
mesh. Tests compare computed results with an analytical solution,


    H_ex = (2 * A / (mu0 * Ms)) * nabla^2 (mx, my, mz) 

where:
    A   - Exchange constant
    Ms  - Saturation magnetisation
    mu0 - Vacuum permability physical constant

Test 1: test_no_exchange_from_uniform_m
    The exchange field is expected to be zero for uniform magnetisation

Test 2: test_compare_with_analytical_solution
    Compare the computed result when 

        m = (0, sin(nx), cos(nx))

    where nx = k(x-0.5) for Ms = 1/mu0 and A = 1. The expected result
    analytical result should be
    
        H_ex = -2 * k^2(0, sin(nx), cos(nx))

In both tests, a 1D mesh is created of length 200nm, with 100 cells
(so 101 nodes). A simulation objects is created from this mesh. The
simulation object is assumed to have the following parameters
associated with it:
    mesh: the mesh
    A: exchange parameter
    Ms: saturation magnetisation
    m: the magnetisation array

Both tests should work in serial and parallel
"""
import numpy as np
from mpi4py import MPI
import pytest

from .skips import xfail_not_implemented

def m_init_uniform(pos):
    """
    Function to initialise a uniform magnetic field in the (1,1,1)
    direction.
    """
    return (1, 1, 1)

def m_init_non_uniform(pos):
    """
    The initial magnetisation function defined in the
    introduction
    """
    x, y, z = pos

    k = 0.1
    nx = k * (x - 0.5)

    return (0, np.sin(nx), np.cos(nx))

def setup_simulation():
    """
    Defines the simulation parameters.
    Creates a simulation object
    Assigns the parameters to the simulation object
    Return the sim object
    """
    from mpimag import FDmesh1D
    from mpimag import Sim
    
    # mesh parameters
    x0 = -100
    x1 = 100 # nm
    nx = 101 # number of nodes in x-dir
    mesh = FDmesh1D(x0, x1, nx)

    mu0 = 4*np.pi*1e-7
    Ms = 1/mu0
    A = 1

    sim = Sim(mesh)
    sim.Ms = Ms
    sim.A = A
    return sim

@xfail_not_implemented
def test_no_exchange_from_uniform_m():
    # mesh parameters
    sim = setup_simulation()
    #initialise the magnetisation
    sim.m = m_init_uniform

    # compute the exchange field
    h_ex_computed = sim.exchange_field
    # define the analytical solution
    h_ex_analytical = np.zeros(100)

    # compare the computed solution with the analytical solution
    if MPI.COMM_WORLD.rank == 0:
        assert (h_ex_computed == h_ex_analytical).all()

@xfail_not_implemented
def test_compare_with_analytical_solution():
    # mesh parameters
    sim = setup_simulation()
    #initialise the magnetisation
    sim.m = m_init_non_uniform

    # compute the exchange field
    h_ex_computed = sim.exchange_field
    # define the analytical solution
    #TODO: define the the analytical solution
    x_coords = np.linspace(-100, 100, 101)

    # compare the computed solution with the analytical solution
    if MPI.COMM_WORLD.rank == 0:
        assert len(cells) == 5
        assert (h_ex_computed == h_ex_analytical).all()