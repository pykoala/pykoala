# Makefile
# py.test -v test/*test?.py && py.test --nbval-lax `cat testable_notebooks.txt`
default: all

test-notebooks:
	py.test --nbval-lax `cat testable_notebooks.txt`

test-units:
	py.test -v -s test/sky_tests.py
all:
	py.test -v -s test/sky_tests.py
