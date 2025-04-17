# Makefile
default: all

test-notebooks:
	py.test --nbval-lax `cat testable_notebooks.txt`

test-units:
	py.test -v test/unit_tests.py test/rss_tests.py test/cubing_tests.py test/astrometry_corr_tests.py test/throughput_corr_tests.py test/sky_tests.py

all:
	py.test -v test/unit_tests.py test/rss_tests.py test/cubing_tests.py test/astrometry_corr_tests.py test/throughput_corr_tests.py test/sky_tests.py  && py.test --nbval-lax `cat testable_notebooks.txt`


