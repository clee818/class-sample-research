# Planned Experiments

For each combination of 2 numbers:
	For the each of the following experiments, run each 10 times for 20 epochs each. At the end of each epoch, calculate accuracy an AUC on  the test dataset of just the 2 classes. 
		- train on 10 classes using softmax
		- train on 10 classes using sigmoid
		- train on 2 classes softmax
		- train on 2 classes sigmoid
		- train on 2 classes with class imbalance softmax
		- train on 2 classes with class imbalance sigmoid

		- for each of the above:
			- save auc for each of the epochs to a file
			- create a graph plotting auc over the 20 epochs

