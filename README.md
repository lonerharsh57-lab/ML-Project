# ml project review 2 package

project: malicious traffic pattern clustering using k-means and hierarchical clustering

## included
- Project_Review_1.pptx
- EDA_Report.docx
- run_clustering.py
- requirements.txt
- dataset_info.md
- results/README.md

## dataset
use the official CICIDS2017 machine-learning csv release. place the downloaded csv files in `data/`.

## run
1. install dependencies: `pip install -r requirements.txt`
2. place CICIDS2017 csv files in `data/`
3. run: `python run_clustering.py`
4. upload the generated files from `results/` to github.

the numerical results are generated from the actual dataset at runtime; this package does not fabricate silhouette scores or cluster counts.
