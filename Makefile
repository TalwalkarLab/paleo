test:
	python -m unittest discover tests "*_test.py" --verbose

lint:
	flake8 --exclude paleo/third_party paleo

# vim:ft=make
