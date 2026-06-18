ORDL
=====

Environment
-----------
Create the conda environment from the exported YAML file:

	conda env create -f environment.yaml
	conda activate ORDL

Reproduce results
-----------------
Run the three dataset scripts:

	python run_experiment.py
	python run_jobs_experiment.py
	python run_twins_experiment.py
