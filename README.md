# INF368A Comparing Models 

- Extend train/validation dataset to include more classes
- Extend the unseen dataset
- Train the following models on the train dataset
	- SM Classifier (Cross Entropy Loss)
	- Triplet Network (Triplet Margin Loss)
	- ArcFace (Angular Margin Loss)
	- SimCLR (Normalized Temperature-scaled Cross Entropy Loss)
- Plot: Training loss and validation accuracies
- Validation metric: simple linear classifier on top of embeddings top 1 and top 3 accuracy
- Test metric: simple linear classifier on top of embeddings
- Hyper-parameter search
	- Triplet Margin
	- ArcFace Margin and Shape
	- SimCLR Temperature
