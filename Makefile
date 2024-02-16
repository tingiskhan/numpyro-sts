format:
	ruff format ./numpyro_sts --check

lint:
	ruff check ./numpyro_sts

test:
	pytest ./tests

# TODO: add coverage
