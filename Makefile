check_dirs := preprocessing
notebooks := notebooks

quality:
	black --check --preview $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)

style:
	black --line-length 119 --preview $(check_dirs)
	isort $(check_dirs)

style_nb:
	black --line-length 115 --preview $(notebooks)
	isort $(notebooks)