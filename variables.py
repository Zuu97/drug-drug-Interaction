n_drug_pairs = 10000
seed = 42
batch_size = 64
val_split = 0.2
host = '0.0.0.0'
port = 5000
threshold = 0.5

# Input feature Dimensions
n_ssp = 1000
n_tsp = 1100
n_gsp = 1200

# AutoEncoder parameters
dim1 = 512
dim2 = 256
dim3 = 64
n_epochs = 50
ssp_weights = "Weights/autoencoder_ssp.h5"
tsp_weights = "Weights/autoencoder_tsp.h5"
gsp_weights = "Weights/autoencoder_gsp.h5"

# DNN parameters
dense1 = 64
dense2 = 64
dense3 = 64
dense4 = 10
dnn_weights = "Weights/dnn.h5"
dnn_epoches = 50