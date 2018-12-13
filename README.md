# EECS 595 Project: Word Sense Induction

## Python dependencies
* tensorflow-gpu v1.12 or later
* scikit-learn v0.20.1 or later
* numpy v1.15 or later

## Usage for word2vecEmbeddingForm.py:
This python file takes and only takes three extra argument:
The first argument takes the trial data directory name.
This argument can only be: 'TrialDatasets'
The second argument takes the test data directory name.
This argument can only be: 'TestDatasets'
The third argument takes the clustering method name.
This argument can only be one of the following three:
DBSCAN or MeanShift or AffinityPropagation
(corresponding to use DBSCAN clustering method, MeanShift clustering method and AffinityPropagation clustering method separately).

The output will be in stdout, which shows the b_cubed score, nmi and harmonic mean of the former two.

Usage example:
cd to the folder which contains the file word2vecEmbeddingForm.py,
Then, run:

python3 word2vecEmbeddingForm.py TrialDatasets TestDatasets DBSCAN      
Or
python3 word2vecEmbeddingForm.py TrialDatasets TestDatasets MeanShift
Or
python3 word2vecEmbeddingForm.py TrialDatasets TestDatasets AffinityPropagation

At present, the project can work only when typing the above three kinds of command.

(Sample Final Output for 
python3 word2vecEmbeddingForm.py TrialDatasets TestDatasets DBSCAN:
Final result for DBSCAN:
Final res for f1 socre for b_cubed 0.3362702736786711
Final res for nmi socre for nmi 0.04481229207745211
harmonic mean: 0.0790854427630621
)


## Usage for bert_embedding.py
This script assumes the following:
* You have cloned the Google BERT repository (https://github.com/google-research/bert) to the same directory.
* You have downloaded and unzipped the BERT-Base, uncased model to the same directory (links to download the model can be found in the README of the BERT repo - do not change the resulting directory's name).
* You have the TrialDatasets and TestDatasets directories set up. 

The script can be run directly using the following command:
`python bert_embedding.py`

In its current state, the model will perform a hyperparameter search for gamma and min_samples for the BERT+DBSCAN model. The results of the hyperparameter search will be written to a file called `hyperparameter_results.txt`. The induced sense labels for the test data will be written to a file called `bert-dynamicdbscan-gamma.out`. Evaluation of these labels can be run using the following command: 
`java -jar SemEval-2013-Task-13-test-data/scoring/fuzzy-bcubed.jar SemEval-2013-Task-13-test-data/keys/gold/all.singlesense.key bert-dynamicdbscan-gamma.out`

GMM clustering can be used instead by changing the cluster function call in the `cluster_all_words` function to `cluster_embeddings_gmm` (instead of `cluster_embeddings_dbscan`). 