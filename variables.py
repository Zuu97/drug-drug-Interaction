n_drug_pairs = 10000
n_drugs = 200
seed = 42
batch_size = 64
val_split = 0.2
host = '0.0.0.0'
port = 5000
threshold = 0.5
table_name = 'ddi_prediction'
root_password = '1234'
db_url = 'mysql+pymysql://root:{}@localhost:3306/ddi'.format(root_password)
# Input feature Dimensions
n_ssp = 1000
n_tsp = 1000
n_gsp = 1000

# AutoEncoder parameters
dim1 = 512
dim2 = 256
dim3 = 64
n_epochs = 50
ssp_weights = "Weights/autoencoder_ssp.h5"
tsp_weights = "Weights/autoencoder_tsp.h5"
gsp_weights = "Weights/autoencoder_gsp.h5"

# DNN parameters
dense1 = 256
dense2 = 128
dense3 = 64
dense4 = 2
dnn_weights = "Weights/dnn.h5"
dnn_epoches = 50
learning_rate = 0.0001

# Visualization
autoencoder_loss_img = 'Visualization/{}_loss.png'

dnn_loss_img = 'Visualization/dnn_loss.png'
dnn_acc_img = 'Visualization/dnn_acc.png'