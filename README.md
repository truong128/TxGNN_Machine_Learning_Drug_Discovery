# TxGNN_Machine_Learning_Drug_Discovery

💥 Step by Step to reproduce the works from this paper:

💯 A foundation model for clinician-centered drug repurposing
Nature Medicine (2024)

1. Installation

conda create --name txgnn_env python=3.8

conda activate txgnn_env

conda install -c dglteam dgl-cuda{$CUDA_VERSION}==0.5.2 

pip install TxGNN


3. Download/load knowledge graph dataset
TxData = TxData(data_folder_path = './data')

TxData.prepare_split(split = 'complex_disease', seed = 42)

TxGNN = TxGNN(data = TxData, 
 weight_bias_track = False,
 proj_name = 'TxGNN', 
 exp_name = 'TxGNN', 
 device = 'cuda:0' # define your cuda device
 )


5. Initialize a new model


👉 Instead of initializing a new model, you can also load a saved model:
TxGNN.load_pretrained('./model_ckpt')


👉 To do pre-training using link prediction for all edge types, you can type:
TxGNN.pretrain(n_epoch = 2, 
 learning_rate = 1e-3,
 batch_size = 1024, 
 train_print_per_n = 20)

 
👉 Lastly, to do finetuning on drug-disease relation with metric learning, you can type:
TxGNN.finetune(n_epoch = 500, 
 learning_rate = 5e-4,
 train_print_per_n = 5,
 valid_per_n = 20,
 save_name = finetune_result_path)
 
-> Output: Testing Loss 0.6381 Testing Micro AUROC 0.7020 Testing Micro AUPRC 0.6851 Testing Macro AUROC 0.6949 Testing Macro AUPRC 0.6878
----- AUROC Performance in Each Relation -----


👉 To save the trained model, you can type:
TxGNN.save_model('./model_ckpt')


👉 To evaluate the model on the entire test set using disease-centric evaluation, you can type:

from txgnn import TxEval
TxEval = TxEval(model = TxGNN)
result = TxEval.eval_disease_centric(disease_idxs = 'test_set', 
 show_plot = False, 
 verbose = True, 
 save_result = True,
 return_raw = False,
 save_name = 'SAVE_PATH')
 

👉 If you want to look at specific disease, you can also do:

result = TxEval.eval_disease_centric(disease_idxs = [9907.0, 12787.0], 
 relation = 'indication', save_result = False)
 
-> Output will be: 
>>> result
 ID                     Name                    Ranked List ... AP@100 Hits@100 Missed@100
2954.0 2954.0 superficial multifocal basal cell carcinoma [Oxaliplatin, Morphine, Rifampicin, Aminosalic... ...   -1    []     []
5173.0 5173.0         actinic keratosis (disease) [Flumethasone, Aminosalicylic acid, Desoximeta... ...   -1    []     []

[2 rows x 59 columns]


👉 After training a satisfying link prediction model, we can also train graph XAI model by:
TxGNN.train_graphmask(relation = 'indication',
 learning_rate = 3e-4,
 allowance = 0.005,
 epochs_per_layer = 3,
 penalty_scaling = 1,
 valid_per_n = 20)

 

👉 gates = TxGNN.retrieve_save_gates('SAVED_PATH')

👉 save and load graphmask model as well via:

TxGNN.save_graphmask_model('./graphmask_model_ckpt')

TxGNN.load_pretrained_graphmask('./graphmask_model_ckpt')
