.PHONY: download_all_datasets

download_all_datasets:
	@echo "Downloading datasets..."
	@mkdir -p data/test data/train data/validate
	@for dir in test train validate; do \
		for file in Cas1.faa Cas1.faa.fai Cas10.faa Cas10.faa.fai Cas12.faa Cas12.faa.fai Cas13.faa Cas13.faa.fai Cas2.faa Cas2.faa.fai Cas3.faa Cas3.faa.fai Cas4.faa Cas4.faa.fai Cas5.faa Cas5.faa.fai Cas6.faa Cas6.faa.fai Cas7.faa Cas7.faa.fai Cas8.faa Cas8.faa.fai Cas9.faa Cas9.faa.fai nocas.faa nocas.faa.fai; do \
			wget -O data/$$dir/$$file https://raw.githubusercontent.com/LUCA-BioTech/AIL-scan/main/datasets/$$dir/$$file; \
			done \
		done

train:
	uv run .venv/bin/accelerate launch --config_file configs/accelerate.yaml src/esm2_classification.py --config_file configs/train.yaml