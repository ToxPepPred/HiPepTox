# ToxMSRC

An innovative peptide toxicity prediction model based on multi-scale convolutional neural network and residual connection

# 1 Description

ToxMSRC is a deep learning model for peptide toxicity prediction that integrates multi-scale convolutional neural networks, bidirectional long short-term memory, and residual connections. The model was trained using 1,818 positive samples and 4,569 negative samples, with the positive samples balanced to 3,636 through SMOTE. It achieved favorable performance on two independent test sets. Peptide toxicity is a critical concern in the development of peptide-based therapeutics, as toxic peptides can cause severe side effects, including organ damage, immune reactions, and cytotoxicity. Accurately predicting peptide toxicity is therefore essential to ensure the safety and efficacy of these drugs. Consequently, we developed the ToxMSRC model to improve the prediction accuracy of peptide toxicity and support future advances in medical drug therapy.


# 2 Requirements


Before running, please make sure the following packages are installed in Python environment:

keras==2.10.0
matplotlib==3.6.2
numpy==1.23.5
pandas==1.5.3
scikit_learn==0.24.2
tensorflow==2.10.0
Python==3.9.13

# 3 Running


1. The folder "Raw data" contains the initial dataset files in FASTA format, the X.npz file processed by Word2Vec, and the CSV files of data labels used for training and testing.
2. The model folder contains the successfully trained model files (.h5).
3. The preprocess folder contains the data processing documentation Jupyter notebook (.ipynb) and the trained Word2Vec model file (.model).
4. The requirements.txt file specifies all dependency libraries and Python versions used in the code. The smote.py contains the complete implementation for handling imbalanced data, while test.py and train.py are the testing and training scripts respectively.

Note: both test.py and train.py are executable scripts for testing and training respectively, and can be run directly to reproduce the results.


# 4 Predict
The predict.py file can predict the peptide toxicity probability for any amino acid sequence. Prepare the sequences to be predicted in FASTA format, modify the input file path in predict.py, and the program will output the probabilities of being toxic and non-toxic. Among them, predict.ipynb provides examples and demonstrates the output results.
