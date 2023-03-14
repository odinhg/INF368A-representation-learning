# Some models for metric and representation learning

Implementations of the following models / loss functions:

- Standard Softmax Classifier (Cross Entropy Loss)
- Triplet Network (Triplet Margin Loss)
- ArcFace (Angular Margin Loss)
- CosFace (Large Margin Cosine Loss)
- SimCLR (Normalized Temperature-scaled Cross Entropy Loss)
	
See `loss_functions.py` for implementations of loss functions. I do not claim these implementations to be optimal in any sense. See some of my other repositories for evaluation of the different models.

**Note**: The triplet margin loss implements both negative and positive mining. To get good convergence, you might want to tweak the mining policy during training.

I do not own the dataset used, but the code can easily be adapted new datasets.
