This folder contains the programs used to rank the unknown disease-gene pairs.

The numbers in the file names indicate the order to run the code.

"6-top10_01.py" and "6-top10_02.py" learn the latent representations from two submodels, respectively.

"7-X_test_combine.py"	concatenates the latent representation from the submodels and generates the joint input data used to train the joint DBN.

"8-top10_score.py"	predicts the probabilities of the unknown disease--gene pairs being disease-associated.

"9-rank.py" rank all the disease--gene pairs based on their corresponding probabilities.
