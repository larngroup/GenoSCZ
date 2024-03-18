import os
import sys
import warnings
import matplotlib
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import tables
import sklearn.metrics as skm
import scipy
warnings.filterwarnings('ignore')
matplotlib.use('agg')
sys.path.insert(1, os.path.dirname(os.getcwd()) + "/")
import tensorflow as tf
import tensorflow.keras as K
tf.keras.backend.set_epsilon(0.0000001)
from LocallyDirectedConnected_tf2 import LocallyDirected1D
from sklearn.model_selection import StratifiedKFold, train_test_split

class TrainDataGenerator(K.utils.Sequence):
    # Adapted from GenNet Framework
    def __init__(self, datapath, batch_size, trainsize, train_indices, epoch_size):
        self.datapath = datapath
        self.batch_size = batch_size
        self.shuffledindexes = np.arange(trainsize)
        np.random.shuffle(self.shuffledindexes)
        self.trainsize = trainsize
        self.training_subjects = pd.read_csv(self.datapath + "/subjects.csv")
        self.training_subjects = self.training_subjects[self.training_subjects['id'].isin(train_indices)]
        self.epoch_size = epoch_size
        self.left_in_greater_epoch = trainsize
        self.count_after_shuffle = 0

    def __len__(self):
        return int(np.ceil(self.epoch_size / float(self.batch_size)))
        
    def __getitem__(self, idx):
        xbatch, ybatch = self.single_genotype_matrix(idx)
        return xbatch, ybatch

    def single_genotype_matrix(self, idx):
        idx2 = idx + self.count_after_shuffle      
        genotype_hdf = tables.open_file(self.datapath + "/genotype.h5", "r")
        batchindexes = self.shuffledindexes[idx2 * self.batch_size:((idx2 + 1) * self.batch_size)]
        ybatch = self.training_subjects["labels"].iloc[batchindexes]
        xbatchid = np.array(self.training_subjects["genotype_row"].iloc[batchindexes], dtype=np.int64)
        xbatch = genotype_hdf.root.data[xbatchid, :]
        ybatch = np.reshape(np.array(ybatch), (-1, 1))
        genotype_hdf.close()
        return xbatch, ybatch

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        left_in_epoch = self.left_in_greater_epoch - self.epoch_size
        if  left_in_epoch < self.epoch_size:
            np.random.shuffle(self.shuffledindexes)
            self.left_in_greater_epoch = self.trainsize
            self.count_after_shuffle = 0
        else:
            self.left_in_greater_epoch = self.left_in_greater_epoch - self.epoch_size
            self.count_after_shuffle = self.count_after_shuffle + int(np.ceil(self.epoch_size / float(self.batch_size)))
    
class EvalGenerator(K.utils.Sequence):
    # Adapted from GenNet Framework
    def __init__(self, datapath, batch_size, val_indices, setsize):
        self.datapath = datapath
        self.batch_size = batch_size
        self.yvalsize = setsize
        self.eval_subjects = pd.read_csv(self.datapath + "/subjects.csv")
        self.eval_subjects = self.eval_subjects[self.eval_subjects['id'].isin(val_indices)]
        
    def __len__(self):
        val_len = int(np.ceil(self.yvalsize / float(self.batch_size)))
        return val_len

    def __getitem__(self, idx):
        xbatch, ybatch = self.single_genotype_matrix(idx)
        return xbatch, ybatch

    def single_genotype_matrix(self, idx):
        genotype_hdf = tables.open_file(self.datapath + "/genotype.h5", "r")
        ybatch = self.eval_subjects["labels"].iloc[idx * self.batch_size:((idx + 1) * self.batch_size)]
        xbatchid = np.array(self.eval_subjects["genotype_row"].iloc[idx * self.batch_size:((idx + 1) * self.batch_size)],
                            dtype=np.int64)
        xbatch = genotype_hdf.root.data[xbatchid, :]                   
        ybatch = np.reshape(np.array(ybatch), (-1, 1))
        genotype_hdf.close()
        return xbatch, ybatch
        
def EncoderDecoder(l1_value = 0.001):
    # Definition of the Encoder-Decoder Architecture
    input_encoder = K.Input(shape=(9160,))

    enc_dec = K.layers.Dense(units = 4500)(input_encoder)
    for layer_size in [1500, 256, 1500, 4500, 9160]:
        enc_dec = K.layers.Activation("relu")(enc_dec)
        enc_dec = K.layers.BatchNormalization(center=False, scale=False)(enc_dec)
        
        enc_dec = K.layers.Dense(units = layer_size)(enc_dec)
    output_decoder = K.layers.Activation("linear")(enc_dec)
    
    autoencoder = K.Model(input_encoder, output_decoder)
    
    return autoencoder
    
def create_network(datapath, inputsize, use_encoder = False, EDM = None, l1_value=0.01):
    # Definition of the Annotation-Based Neural Network (ABNN) Architecture was adapted from GenNet Framework
    masks = []
    
    network_csv = pd.read_csv(datapath + "/topology.csv")
    network_csv = network_csv.filter(like="node", axis=1)
    columns = list(network_csv.columns.values)
    network_csv = network_csv.sort_values(by=columns, ascending=True)

    input_layer = K.Input((inputsize,), name='input_layer')
    model = K.layers.Reshape(input_shape=(inputsize,), target_shape=(inputsize, 1))(input_layer)

    for i in range(len(columns) - 1):
        matrix_ones = np.ones(len(network_csv[[columns[i], columns[i + 1]]]), np.bool)
        matrix_coord = (network_csv[columns[i]].values, network_csv[columns[i + 1]].values)
        if i == 0:
            matrixshape = (inputsize, network_csv[columns[i + 1]].max() + 1)
        else:
            matrixshape = (network_csv[columns[i]].max() + 1, network_csv[columns[i + 1]].max() + 1)
        mask = scipy.sparse.coo_matrix(((matrix_ones), matrix_coord), shape = matrixshape)
        masks.append(mask)
        
        model = LocallyDirected1D(mask=mask, filters=1, input_shape=(mask.shape[0], 1),
                                  name="LocallyDirected_" + str(i))(model)
        model = K.layers.Activation("tanh")(model)
        model = K.layers.BatchNormalization(center=False, scale=False)(model)
           
    model = K.layers.Flatten()(model)
    
    # Extension with Encoder and FCNN branches
    if use_encoder:
        
        Encoder_branch = EDM.layers[1](model)
        for i in range(2,9):
            Encoder_branch = EDM.layers[i](Encoder_branch)
          
        initialize_branch = True    
        for layer_size in [2500, 625, 256]:
            if initialize_branch:
                Trainable_branch = K.layers.Dense(units=layer_size, kernel_regularizer=tf.keras.regularizers.l1(l=l1_value), 
                                       bias_initializer= tf.keras.initializers.Constant(0))(model)
                initialize_branch = False
            else:
                Trainable_branch = K.layers.Dense(units=layer_size, kernel_regularizer=tf.keras.regularizers.l1(l=l1_value), 
                                       bias_initializer= tf.keras.initializers.Constant(0))(Trainable_branch)
        
            Trainable_branch = K.layers.Activation("relu")(Trainable_branch)
            Trainable_branch = K.layers.BatchNormalization(center=False, scale=False)(Trainable_branch)

        # Concatenation
        
        concatted = K.layers.Concatenate()([Trainable_branch, Encoder_branch])
        concatted = K.layers.Dense(units=256, kernel_regularizer=tf.keras.regularizers.l1(l=l1_value), 
                               bias_initializer= tf.keras.initializers.Constant(0))(concatted)
        concatted = K.layers.Activation("relu")(concatted)
        concatted = K.layers.BatchNormalization(center=False, scale=False)(concatted)
            
        concatted = K.layers.Dense(units=1, name="output_layer",
                               kernel_regularizer=tf.keras.regularizers.l1(l=l1_value), 
                               bias_initializer= tf.keras.initializers.Constant(0))(Trainable_branch)    
        output_layer = K.layers.Activation("sigmoid")(concatted)
        
    # For the first training round, the ABNN is directly connected to the output layer
    else:
        model = K.layers.Dense(units=1, name="output_layer",
                               kernel_regularizer=tf.keras.regularizers.l1(l=l1_value), 
                               bias_initializer= tf.keras.initializers.Constant(0))(model)    
        output_layer = K.layers.Activation("sigmoid")(model)
   
    model = K.Model(inputs=input_layer, outputs=output_layer)

    return model, masks

# Auxiliary functions for Sensitivity and Specifity for model evaluation metrics during training
def sensitivity(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.keras.backend.clip(y_pred, 0, 1)
    y_true = tf.keras.backend.clip(y_true, 0, 1)

    y_pred = tf.keras.backend.round(y_pred)

    true_p = K.backend.sum(K.backend.round(y_pred) * y_true)
    pos = tf.keras.backend.sum(y_true)
    sensitivity = tf.keras.backend.clip((true_p / (pos + 0.00001)), 0, 1)
    return sensitivity

def specificity(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.keras.backend.clip(y_pred, 0, 1)
    y_true = tf.keras.backend.clip(y_true, 0, 1)

    neg_y_true = 1 - y_true
    neg_y_pred = 1 - K.backend.round(y_pred)
    fp = K.backend.sum(neg_y_true * K.backend.round(y_pred))
    tn = K.backend.sum(neg_y_true * neg_y_pred)
    specificity = tn / (tn + fp + 0.00001)
    return tf.keras.backend.clip(specificity, 0, 1)

def weighted_binary_crossentropy(y_true, y_pred):
    y_true = K.backend.clip(tf.cast(y_true, dtype=tf.float32), 0.0001, 1)
    y_pred = K.backend.clip(tf.cast(y_pred, dtype=tf.float32), 0.0001, 1)

    return K.backend.mean(
        -y_true * K.backend.log(y_pred + 0.0001) * weight_positive_class - (1 - y_true) * K.backend.log(
            1 - y_pred + 0.0001) * weight_negative_class)

# Functions to generate results files
def evaluate_performance(y, p, fold, resultpath, subset):
    confusion_matrix = skm.confusion_matrix(y, p.round())

    fpr, tpr, thresholds = skm.roc_curve(y, p)
    roc_auc = skm.auc(fpr, tpr)
    sensitivity = confusion_matrix[1, 1] / (confusion_matrix[1, 0] + confusion_matrix[1, 1])
    specificity = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])
    accuracy = (confusion_matrix[1, 1] + confusion_matrix[0, 0])/ len(y)
    precision = confusion_matrix[1, 1] / (confusion_matrix[0, 1] + confusion_matrix[1, 1])
    f1_score = skm.f1_score(y, p.round())
    
    if subset == 'validation':
        f = open(resultpath + '/Results.txt', 'a')
        f.write("Validation set - Fold " + str(fold+1) + '\n')
            
    elif subset == 'test':
        f = open(resultpath + '/Results.txt', 'a')
        f.write("Test set - Fold " + str(fold+1) + '\n')
        
    elif subset == 'pre_train':
        f = open(resultpath + '/Pre_train_Results.txt', 'a')
        f.write("Test set \n")
            
    f.write('Sensitivity = %.4f \n' % sensitivity)
    f.write('Specificity = %.4f \n' % specificity)
    f.write('Accuracy = %.4f \n' % accuracy)
    f.write('Precision = %.4f \n' % precision)       
    f.write('F1 Score = %.4f \n' % f1_score)
    f.write('Score auc = %.4f \n' % roc_auc)
    f.write(' Confusion matrix:')
    f.write(str(confusion_matrix))
    f.write('\n\n')
    f.close()   

def create_importance_csv(datapath, model, masks):
    # Adapted from GenNet Framework
    # Used only on ABNN
    network_csv = pd.read_csv(datapath + "/topology.csv")

    coordinate_list = []
    for i, mask in zip(np.arange(len(masks)), masks):
        coordinates = pd.DataFrame([])

        if (i == 0):
            if 'chr' in network_csv.columns:
                coordinates["chr"] = network_csv["chr"]
        coordinates["node_layer_" + str(i)] = mask.row
        coordinates["node_layer_" + str(i + 1)] = mask.col
        coordinates = coordinates.sort_values("node_layer_" + str(i), ascending=True)
        coordinates["weights_" + str(i)] = model.get_layer(name="LocallyDirected_" + str(i)).get_weights()[0]

        coordinate_names = network_csv[["layer" + str(i) + "_node", "layer" + str(i) + "_name"]].drop_duplicates()
        coordinate_names = coordinate_names.rename({"layer" + str(i) + "_node": "node_layer_" + str(i)}, axis=1)
        coordinates = coordinates.merge(coordinate_names, on="node_layer_" + str(i))
        coordinate_list.append(coordinates)

        if i == 0:
            total_list = coordinate_list[i]
        else:
            total_list = total_list.merge(coordinate_list[i], on="node_layer_" + str(i))

    i += 1
    coordinates = pd.DataFrame([])
    coordinates["weights_" + str(i)] = model.get_layer(name="output_layer").get_weights()[0].flatten()
    coordinates["node_layer_" + str(i)] = np.arange(len(coordinates))
    coordinate_names = network_csv[["layer" + str(i) + "_node", "layer" + str(i) + "_name"]].drop_duplicates()
    coordinate_names = coordinate_names.rename({"layer" + str(i) + "_node": "node_layer_" + str(i)}, axis=1)
    coordinates = coordinates.merge(coordinate_names, on="node_layer_" + str(i))
    total_list = total_list.merge(coordinates, on="node_layer_" + str(i))
    total_list["raw_importance"] = total_list.filter(like="weights").prod(axis=1)
    return total_list

# function to compute Prediction Score aggregations and append to results file
def aggregated_preds(preds, true_labels, resultpath, factor = 0.2):
    with open(resultpath + '/Results.txt', 'a') as f:
        f.write("\n Aggregated Predictions on Test Set")
        f.write("\n -- Strategy: Mean of Predictions")
        rearranged_preds = []
        
        for l in range(len(true_labels)):
            preds_by_ind = [preds[fold][l] for fold in range(len(preds))]
            rearranged_preds.append(preds_by_ind)
        
        mean_values = [np.mean(j) for j in rearranged_preds]
        mean_values = np.array(mean_values)
            
        confusion_matrix = skm.confusion_matrix(true_labels, mean_values.round())
        fpr, tpr, thresholds = skm.roc_curve(true_labels, mean_values)
        roc_auc = skm.auc(fpr, tpr)
        sensitivity = confusion_matrix[1, 1] / (confusion_matrix[1, 0] + confusion_matrix[1, 1])  
        specificity = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])
        accuracy = (confusion_matrix[1, 1] + confusion_matrix[0, 0])/ len(true_labels)
        precision = confusion_matrix[1, 1] / (confusion_matrix[0, 1] + confusion_matrix[1, 1])
        f1_score = skm.f1_score(true_labels, mean_values.round())
            
        f.write('\n Sensitivity = %.4f' % sensitivity)
        f.write('\n Specificity = %.4f' % specificity)
        f.write('\n Accuracy = %.4f' % accuracy)
        f.write('\n Precision = %.4f' % precision)       
        f.write('\n F1 Score = %.4f' % f1_score)
        f.write('\n Score auc = %.4f' % roc_auc)
        f.write('\n Confusion matrix:')
        f.write(str(confusion_matrix))
        f.write('\n\n')

# Model
def train_classification(args):
    jobid = args.ID
    datapath = args.path
    resultpath = str(args.out) + "/Experiment_" + str(jobid) + "_/"
    
    wpc = args.wpc
    lr_opt = args.learning_rate
    l1_value = args.L1
    batch_size = args.batch_size
    epochs = args.epochs 
    patience = 5
    lr_patience = 3
    pre_train = True
    train_encoder = True
    
    if pre_train and train_encoder:
        os.mkdir(resultpath)
        os.mkdir(resultpath + 'weights/')
        os.mkdir(resultpath + 'preds/')
        os.mkdir(resultpath + 'images/')

    global weight_positive_class, weight_negative_class
    weight_positive_class, weight_negative_class = wpc, 1
    
    h5file = tables.open_file(datapath + '/genotype.h5', "r")
    subject_size, inputsize = h5file.root.data.shape
    h5file.close()
    
    subjects = pd.read_csv(datapath + "/subjects.csv")
    subj_labels = subjects['labels']
    
    #%% Pre-Train
    if pre_train:
        train_val_set, test_set = train_test_split(subjects, test_size = 0.10, shuffle = True, stratify=subj_labels)
        train_val_set.sort_index(inplace=True)
        train_samples = train_val_set['id'].tolist()
        train_size = len(train_samples)
        test_set.sort_index(inplace=True)
        test_samples = test_set['id'].tolist()
        test_set['patient_id'].to_csv(resultpath + 'preds/test_samples.csv', header=False, index=False)
        ytest_ind = subjects.iloc[test_samples]
        ytest = np.reshape(np.array(ytest_ind["labels"].values), (-1, 1))
        
        model, masks = create_network(datapath=datapath, inputsize=inputsize, l1_value=l1_value)
        
        ##### Pre-training of first component (ABNN)
        fh = open(resultpath + '/model_architecture.txt', 'w')
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
        fh.close()
    
        optimizer_model = tf.keras.optimizers.Adam(lr=lr_opt)
        model.compile(loss=weighted_binary_crossentropy, optimizer=optimizer_model,metrics=["accuracy", sensitivity, specificity])
        early_stop = K.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=patience, verbose=1, mode='auto',
                                               restore_best_weights=True)
        reduce_lr = K.callbacks.ReduceLROnPlateau(monitor='loss', patience = lr_patience, factor = 0.33, min_lr = 0.00001, verbose = 1)
        save_best_model = K.callbacks.ModelCheckpoint(resultpath + "weights/bestweights_job.h5", monitor='loss',
                                                      verbose=1, save_best_only=True, mode='auto')

        train_generator = TrainDataGenerator(datapath=datapath, batch_size=batch_size, trainsize=train_size,
                                             train_indices=train_samples, epoch_size=train_size)
        
        model.fit_generator(generator=train_generator, shuffle=True, epochs=200, verbose=0,
            callbacks=[early_stop, reduce_lr, save_best_model], workers=1, use_multiprocessing=False,
            validation_data=EvalGenerator(datapath=datapath, batch_size=batch_size, val_indices=train_samples, setsize=train_size))
        
        importance_csv = create_importance_csv(datapath, model, masks)
        importance_csv.to_csv(resultpath + "weights/connection_weights_pre_train.csv")
        
        model.load_weights(resultpath + 'weights/bestweights_job.h5')
        
        pre_train_test = model.predict_generator(EvalGenerator(datapath=datapath, batch_size=batch_size, val_indices=test_samples, setsize=len(test_samples)))
        evaluate_performance(ytest, pre_train_test, None, resultpath, "pre_train")
            
        np.save(resultpath + "preds/ptest_pre.npy", pre_train_test)
    
    else:
        test_samples_file = open(resultpath + 'preds/test_samples.csv', 'r')
        test_samples_list = [s[0] for s in list(csv.reader(test_samples_file))]
        test_samples_file.close()
        
        train_val_set = subjects[-subjects['patient_id'].isin(test_samples_list)]
        train_samples = train_val_set['id'].tolist()
        test_set = subjects[subjects['patient_id'].isin(test_samples_list)]
        test_samples = test_set['id'].tolist()
        ytest_ind = subjects.iloc[test_samples]
        ytest = np.reshape(np.array(ytest_ind["labels"].values), (-1, 1))
        
        model, masks = create_network(datapath=datapath, inputsize=inputsize, l1_value=l1_value)
        model.load_weights(resultpath + 'weights/bestweights_job.h5')
        
    #%% Encoder
    if train_encoder:
        ### Data loading for encoder-decoder
        all_subjects = pd.read_csv(datapath + "/subjects.csv")
        training_subjects = all_subjects[all_subjects['id'].isin(train_samples)]
        test_subjects = all_subjects[all_subjects['id'].isin(test_samples)]
        
        genotype_hdf = tables.open_file(datapath + "/genotype.h5", "r")
        xbatchid = np.array(training_subjects["genotype_row"])
        xbatch = genotype_hdf.root.data[xbatchid, :]
        ybatchid = np.array(test_subjects["genotype_row"])
        ybatch = genotype_hdf.root.data[ybatchid, :]
        genotype_hdf.close()
        
        inp = model.input
        output_genes = model.layers[5].output
        functor = K.backend.function([inp], [output_genes])
        
        gene_layer_values_x = functor([xbatch])
        gene_layer_values_y = functor([ybatch])
        
        Enc_Dec_Model = EncoderDecoder()
        
        Enc_Dec_Model.compile(optimizer='adam', loss='mse')
        
        early_stop = K.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=patience, verbose=1, mode='auto', restore_best_weights=True)
        save_best_model = K.callbacks.ModelCheckpoint(resultpath + "weights/encoder_bestweights_job.h5", monitor='loss',
                                                      verbose=1, save_best_only=True, mode='auto')
        
        Enc_Dec_Model.fit(gene_layer_values_x, gene_layer_values_x,
                        epochs = 100, callbacks=[early_stop,save_best_model], batch_size = 256)
        
        fh = open(resultpath + '/encoder_results.txt', 'w')
        Enc_Dec_Model.summary(print_fn=lambda x: fh.write(x + '\n'))
        fh.write('Batch Size = 256\n')
        fh.write('Batch Normalization = Yes\n')
        fh.write('Dropout = None\n')
        fh.write('Activations: ReLU. linear for output')
        
        AE_predictions = Enc_Dec_Model.predict(gene_layer_values_y)
        fh.write('\n')
        score_mse = skm.mean_squared_error(gene_layer_values_y[0], AE_predictions, squared = True)
        fh.write('MSE: ' + str(score_mse) + '\n')
        score_rmse = skm.mean_squared_error(gene_layer_values_y[0], AE_predictions, squared = False)
        fh.write('RMSE: ' + str(score_rmse) + '\n')
        score_mae = skm.mean_absolute_error(gene_layer_values_y[0], AE_predictions)
        fh.write('MAE: ' + str(score_mae) + '\n\n')
        
        AE_predictions_x = Enc_Dec_Model.predict(gene_layer_values_x)
        fh.write('Training Reconstruction: \n')
        score_mse = skm.mean_squared_error(gene_layer_values_x[0], AE_predictions_x, squared = True)
        fh.write('MSE: ' + str(score_mse) + '\n')
        score_rmse = skm.mean_squared_error(gene_layer_values_x[0], AE_predictions_x, squared = False)
        fh.write('RMSE: ' + str(score_rmse) + '\n')
        score_mae = skm.mean_absolute_error(gene_layer_values_x[0], AE_predictions_x)
        fh.write('MAE: ' + str(score_mae) + '\n\n')
        fh.close()
        
    else:                
        Enc_Dec_Model = EncoderDecoder()
        
    Enc_Dec_Model.load_weights(resultpath + 'weights/encoder_bestweights_job.h5')
    
    #%% Subseting
    subject_size = train_val_set.shape[0]
    subj_labels = train_val_set['labels']
    
    test_preds = []
    num_folds = 10
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=None)
    
    #%% Multiple Fold Training
    for fold, (train_indices, val_indices) in enumerate(kfold.split(np.zeros(subject_size), subj_labels)):
        print(f"Fold {fold + 1}/{num_folds}")
        
        val_samples_file = open(resultpath + 'preds/val_samples_'+str(fold+1)+'.csv', 'r')
        val_samples_list = [s[0] for s in list(csv.reader(val_samples_file))]
        val_samples_file.close()
        
        train_set = train_val_set[-train_val_set['patient_id'].isin(val_samples_list)]
        train_samples = train_set['id'].tolist()
        val_set = train_val_set[train_val_set['patient_id'].isin(val_samples_list)]
        val_samples = val_set['id'].tolist()
        
        train_size = len(train_samples)
        val_size = len(val_samples)
        
        model, masks = create_network(datapath=datapath, inputsize=inputsize, use_encoder=True, EDM=None, l1_value=l1_value)
        
        if not os.path.exists(resultpath + '/second_model_architecture.txt'):
            fh = open(resultpath + '/second_model_architecture.txt', 'w')
            model.summary(print_fn=lambda x: fh.write(x + '\n'))
            fh.close()
    
        optimizer_model = tf.keras.optimizers.Adam(lr=lr_opt)
        model.compile(loss=weighted_binary_crossentropy, optimizer=optimizer_model,metrics=["accuracy", sensitivity, specificity])

        csv_logger = K.callbacks.CSVLogger(resultpath + 'preds/train_log_' + str(fold+1) + '.csv', append=True)
        early_stop = K.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=patience, verbose=1, mode='auto',
                                               restore_best_weights=True)
        reduce_lr = K.callbacks.ReduceLROnPlateau(monitor='val_loss', patience = lr_patience, factor = 0.33, min_lr = 0.00001, verbose = 1)
        save_best_model = K.callbacks.ModelCheckpoint(resultpath + "weights/bestweights_job_" + str(fold+1) + ".h5", monitor='val_loss',
                                                      verbose=1, save_best_only=True, mode='auto')

        train_generator = TrainDataGenerator(datapath=datapath, batch_size=batch_size, trainsize=train_size,
                                             train_indices=train_samples, epoch_size=train_size)

        model.fit_generator(generator=train_generator, shuffle=True, epochs=epochs, verbose=0,
            callbacks=[early_stop, save_best_model, csv_logger, reduce_lr], workers=1, use_multiprocessing=False,
            validation_data=EvalGenerator(datapath=datapath, batch_size=batch_size, val_indices=val_samples, setsize=val_size))
        """
        #%% Loss Curve Plots
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(resultpath + "images/train_val_loss_" + str(fold+1) + ".png")
        plt.show()
        """
        print(f"Finished fold {fold + 1}/{num_folds}")
        
        #%% Validation and Test
        model.load_weights(resultpath + 'weights/bestweights_job_' + str(fold+1) + '.h5')
        pval = model.predict_generator(EvalGenerator(datapath=datapath, batch_size=batch_size, val_indices=val_samples, setsize=val_size))
        yval_ind = subjects.iloc[val_samples]
        yval = np.reshape(np.array(yval_ind["labels"].values), (-1, 1))
            
        if not os.path.exists(resultpath + '/Results_trainable_branch.txt'):
            with open(resultpath + '/Results_trainable_branch.txt', 'a') as f:
                f.write('\n Jobid = ' + str(jobid))
                f.write('\n Batchsize = ' + str(batch_size))
                f.write('\n Weight positive class = ' + str(weight_positive_class))
                f.write('\n Weight negative class= ' + str(weight_negative_class))
                f.write('\n Learningrate = ' + str(lr_opt))
                f.write('\n Optimizer = ' + model.optimizer.get_config()['name'])
                f.write('\n L1 value = ' + str(l1_value))
                f.write('\n Patience = ' + str(patience))
                f.write('\n Folds = ' + str(num_folds))
                f.write('\n Training set size = ' + str(len(train_samples)))
                f.write('\n Validation set size = ' + str(val_size))
                f.write('\n Hold-out test set')
                f.write('\n Test set size = ' + str(len(test_samples)))
                f.write('\n\n')
        
        evaluate_performance(yval, pval, fold, resultpath, "validation")
        
        np.save(resultpath + "preds/pval_" + str(fold+1) + ".npy", pval)
        
        ptest = model.predict_generator(EvalGenerator(datapath=datapath, batch_size=batch_size, val_indices=test_samples, setsize=len(test_samples)))
        test_preds.append(ptest)
        evaluate_performance(ytest, ptest, fold, resultpath, "test")
            
        np.save(resultpath + "preds/ptest_" + str(fold+1) + ".npy", ptest)


    #%% Final Aggregation
    aggregated_preds(test_preds, ytest, resultpath, factor = 0.2)
