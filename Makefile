# Makefile

test-notebooks:
	py.test --nbval-lax `cat successful_run_notebooks.txt`

test-units:
	py.test -v test/test_units.py