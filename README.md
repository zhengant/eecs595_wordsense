# EECS 595 Project: Word Sense Induction

Introduction to use and execute the file: word2vecEmbeddingForm.py:

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





