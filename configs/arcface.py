# Settings file for baseline SoftMax classifier
batch_size = 64
epochs = 50 
lr = 0.0014
val = 0.05 #Use 5% for validation data 
test = 0.2 #Use 20% for test data
image_size = (128, 128)

model_type = "arcface"

# Parameters for the angular margin loss
margin = 0.5
scale = 64
