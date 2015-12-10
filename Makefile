test:
	py.test -v

test-parallel:
	mpiexec -n 3 py.test -v

test-parallel-4:
	mpiexec -n 4 py.test -v -m parallel4

coverage:
	py.test  --cov mpimag --cov-report=term --cov-report=html
