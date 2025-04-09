# Makefile
default: all

test-notebooks:
	py.test --nbval-lax `cat testable_notebooks.txt`

test-units:
	py.test -v test/*test?.py

all:
	py.test -v test/*test?.py && py.test --nbval-lax `cat testable_notebooks.txt`


