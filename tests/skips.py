import pytest
from mpi4py import MPI

# Skips with respect to parallel or serial
xfail_not_implemented = pytest.mark.xfail(reason="This test does not work as simulation code not yet implemented")
xfail_in_parallel = pytest.mark.xfail(MPI.COMM_WORLD.size > 1,
                                      reason="This test does not yet work in parallel.")
skip_in_parallel = pytest.mark.skipif(MPI.COMM_WORLD.size > 1,
                                      reason="This test should only be run in serial.")
skip_in_serial = pytest.mark.skipif(MPI.COMM_WORLD.size == 1,
                                    reason="This test should only be run in parallel.")
skip_if_not_parallel4 = pytest.mark.skipif(MPI.COMM_WORLD.size != 4,
                                    reason="This test should only be run in parallel.")
skip_if_not_process0 = pytest.mark.skipif(MPI.COMM_WORLD.rank != 0,
                                    reason="This test should only be run from process zero.")