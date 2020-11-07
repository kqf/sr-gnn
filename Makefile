# kaggle API and UI archives have different names

data/: 
	kaggle datasets download -d chadgostopp/recsys-challenge-2015

	unzip recsys-challenge-2015.zip -d data/

	@# Remove the duplicate file
	rm -rf data/yoochoose-data

	mv recsys-challenge-2015.zip data/ 



clean:
	rm -rf data/processed/

.PHONY: train
