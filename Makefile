test:
	py.test -v


coverage:
	py.test  --cov mpimag --cov-report=term --cov-report=html
