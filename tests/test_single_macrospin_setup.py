"""
=========================================================================
Single Macrospin Setup Tests
=========================================================================

Tests for the setup of a single macrospin

A single macrospin is required. It is neccessary to specify the following
parameters associated with it:

- The coordinate where it is located in space (assuming cartesian)
- Saturation Magnetisation, Ms
- alpha
- gamma
- The Zeeman field
- initial magnetisation direction, m
- time, t

The following tests check the assignment of these variables.

"""


import numpy as np
import pytest


def setup():
    from mpimag import Macrospin
    return Macrospin()


def test_set_coordinate():
    sim = setup()

    # test error is raised if user tries to call value of
    # coordinate before it has been set
    with pytest.raises(AttributeError):
        sim.coord

    # try to set coordinate with non-2D coordinate
    with pytest.raises(ValueError):
        sim.coord = 0

    # try to set coordinate with 3D coordinate
    with pytest.raises(ValueError):
        sim.coord = (0, 0, 0)

    # set coordinate with 2D coordinate (0,0) and assert these
    # values are returned when requested
    sim.coord = (0, 0)
    assert np.all(sim.coord == (0, 0))


def test_set_Ms():
    sim = setup()

    # test error is raised if user tries to call value of
    # Ms before it has been set
    with pytest.raises(AttributeError):
        sim.Ms

    # set Ms and assert this value is returned
    sim.Ms = 8.6e5
    assert (sim.Ms == 8.6e5)


def test_set_alpha():
    sim = setup()

    # test error is raised if user tries to call value of
    # alpha before it has been set
    with pytest.raises(AttributeError):
        sim.alpha

    # set alpha and assert this value is returned
    sim.alpha = 0.1
    assert (sim.alpha == 0.1)


def test_set_gamma():
    sim = setup()

    # test error is raised if user tries to call value of
    # gamma before it has been set
    with pytest.raises(AttributeError):
        sim.gamma

    # set gamma and assert this value is returned
    sim.gamma = 2.211e5
    assert (sim.gamma == 2.211e5)


def test_set_zeeman():
    sim = setup()

    # test error is raised if user tries to call value of
    # Zeeman before it has been set
    with pytest.raises(AttributeError):
        sim.zeeman

    # try to set Zeeman with non-3D vector
    with pytest.raises(ValueError):
        sim.zeeman = 0

    # try to set Zeeman with 2D vector
    with pytest.raises(ValueError):
        sim.zeeman = [0, 0]

    # set zeeman with 3D Vector (0,0, B/mu0) and assert these
    # values are returned when requested
    mu0 = 4 * np.pi * 1e-7
    B = 0.1
    sim.zeeman = [0, 0, B / mu0]
    assert np.all(sim.zeeman == [0, 0, B / mu0])


def test_set_m():
    sim = setup()

    # test error is raised if user tries to call value of
    # m before it has been set
    with pytest.raises(AttributeError):
        sim.m

    # try to set m with non-3D vector
    with pytest.raises(ValueError):
        sim.m = 0

    # try to set m with 2D vector
    with pytest.raises(ValueError):
        sim.m = [0, 0]

    # set m with 3D Vector (0,0, 1.) and assert these
    # values are returned when requested
    sim.m = [0., 0., 1.]
    assert np.all([sim.m[0], sim.m[1], sim.m[2]] == [0., 0., 1.])

    # set m with 3D Vector (1,1,1) and assert the returned vector values
    # are normalised when requested.
    sim.m = [1, 2, 3]
    norm = np.sqrt(1**2 + 2**2 + 3**2)
    expected = np.array([1 / norm, 2 / norm, 3 / norm])
    assert np.all(sim.m == expected)


def test_sim_time():
    sim = setup()

    # assert that sim is initialised with t=0
    assert (sim.t == 0.0)

    # assert that if time is changed, this new time will be returned
    # when requested
    sim.t = 1.0e-9
    assert sim.t == 1.0e-9

    # check ValueError is raised when attempt is made to
    # set time with a negative value.
    with pytest.raises(ValueError):
        sim.t = -2e-9
