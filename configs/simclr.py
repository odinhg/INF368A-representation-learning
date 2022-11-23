# Settings SimCLR self-supervised learning 
batch_size = 256 
epochs = 50 
lr = 0.0014
val = 0.05#0.05 #Use 5% for validation data 
test = 0.01#0.2 #Use 20% for test data
image_size = (128, 128)

temperature = 0.1 # NTXent loss temperature parameter

model_type = "simclr"
