format:
	ruff format ./numpyro_sts --check

lint:
	ruff check ./numpyro_sts

test:
	coverage run -m pytest ./tests

coverage: test
	coverage report --fail-under=100

# TODO: add coverage
