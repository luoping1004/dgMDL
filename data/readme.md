This folder contains preprocessed data that can be used to test the program.

The four 'npy' files are the raw features extracted for genes and diseases in the PPI-based and GO-based models.

"InBioList.txt" contains all the genes being analyzed in the study.
"GeneList_v1.1.txt" contains all the disease-associated genes.

"DiseaseList_v1.1.txt" contains all the diseases being analyzed in the study. The names of the diseases are temporary and will be modified in a few days.

"5foldIdx_shuffle_RN" contains the indices of the disease-gene pairs used in the 5-fold cross-validation. In the _de novo_ prediction, the training and testing indices are concatenated to generate known data which are used to train the multimodel DBN.
