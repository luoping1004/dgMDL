This folder contains preprocessed data that can be used to test the program.

The four 'npy' files are the raw features extracted for genes and diseases in the PPI-based and GO-based models.

"InBioList.txt" contains all the genes being analyzed in the study.
"GeneList_v1.1.txt" contains all the disease-associated genes.

"Classified_disease_id.txt" contains the classified OMIM IDs of the 1154 diseases analyzed in the study. For those diseases without official IDs, a temporary ID is assigned for each of them at the end of the file.

"5foldIdx_shuffle_RN" contains the indices of the disease-gene pairs used in the 5-fold cross-validation. In the _de novo_ prediction, the training and testing indices are concatenated to generate known data which are used to train the multimodel DBN.


The association matrix file "Adj_v1.1.npy" is larger than 25 mb which cannot be uploaded to github. But it can be obtained from the following two cloud drives.

https://pan.baidu.com/s/109MwXysI5PlMzO2LkNEFNQ code: htu3
or
https://drive.google.com/file/d/1890OQnpB5nwquhfyRTa9m4J0yaYVMiZr/view?usp=sharing
