#importing libraries
from sklearn.model_selection import StratifiedKFold
import imblearn
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import deepchem as dc
import numpy as np
import pandas as pd
# import pickle
# import matplotlib.pyplot as plt
# import seaborn as sns
import tensorflow as tf
from collections import Counter
# sns.set_style('white')
import warnings
warnings.filterwarnings("ignore")
from statistics import mean
from sklearn.metrics import roc_curve, auc
import numpy as np
import os
from deepchem.feat import MolGraphConvFeaturizer
from deepchem.feat.graph_data import GraphData
from deepchem.feat.mol_graphs import ConvMol
from deepchem.data import NumpyDataset
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"]=""

def fetch_all_jobs_path():
  """
  This method creates (if doesnot exists) a folder where all jobs will be stored
  :return: return all jobs folder path
  """
  prnt_fld = Path.home()
  fld_path = os.path.join(prnt_fld, "ml_olfa", "all_jobs")
  print(fld_path)
  if not os.path.exists(fld_path):
    os.makedirs(fld_path, exist_ok=True)

  return fld_path

def get_labels(pred_train,pred_test): #Getting discrete labels from probability values
    train_pred = []
    test_pred = []
    for i in range(pred_train.shape[0]):
        if(pred_train[i][0]>pred_train[i][1]):
          train_pred.append(0)
        else:
          train_pred.append(1)
    for i in range(pred_test.shape[0]):
        if(pred_test[i][0]>pred_test[i][1]):
          test_pred.append(0)
        else:
          test_pred.append(1)
    return train_pred,test_pred

def get_labels_DAG(pred_train,pred_test): #Getting discrete labels from probability values
  train_pred = []
  test_pred = []
  for i in range(pred_train.shape[0]):
    if(pred_train[i][0][0]>pred_train[i][0][1]):
      train_pred.append(0)
    else:
      train_pred.append(1)
  for i in range(pred_test.shape[0]):
    if(pred_test[i][0][0]>pred_test[i][0][1]):
      test_pred.append(0)
    else:
      test_pred.append(1)
  return train_pred,test_pred


def evaluate(model,metrics,train_dataset,test_dataset,y_train,y_test):
    pred_test = model.predict(test_dataset) #model gives probability values of shape (n_samples,n_tasks,n_classes)
    pred_train = model.predict(train_dataset) #n_tasks = 1,n_classes = 2 in our case
    pred_train_labels,pred_test_labels = get_labels(pred_train,pred_test)
    precision_train = dc.metrics.precision_score(y_train,pred_train_labels)
    precision_test = dc.metrics.precision_score(y_test,pred_test_labels)
    recall_train = dc.metrics.recall_score(y_train,pred_train_labels)
    recall_test = dc.metrics.recall_score(y_test,pred_test_labels)
    results = []
    results.extend(list(model.evaluate(train_dataset, metrics).values()))
    results.extend([precision_train,recall_train])
    results.extend(list(model.evaluate(test_dataset, metrics).values()))
    results.extend([precision_test,recall_test])
    return results


def generate_dataset(data,feat):
    labels = data['Activation Status']
    smiles = data['SMILES']
    featurizer = feat
    features = featurizer.featurize(smiles)
    dataset = NumpyDataset(features, labels)
    labels = np.array(labels)
    return dataset,labels

def generate_dataset_DAG(data,feat):
    labels = data['Activation Status']
    smiles = data['SMILES']
    featurizer = feat
    features = featurizer.featurize(smiles)
    dataset = NumpyDataset(features, labels)
    trans = dc.trans.DAGTransformer(max_atoms=50)
    dataset = trans.transform(dataset)
    labels = np.array(labels)
    return dataset,labels


def evaluate_DAG(model,metrics,train_dataset,test_dataset,y_train,y_test):
  pred_test = model.predict(test_dataset) #model gives probability values of shape (n_samples,n_tasks,n_classes)
  pred_train = model.predict(train_dataset) #n_tasks = 1,n_classes = 2 in our case
  pred_train_labels,pred_test_labels = get_labels_DAG(pred_train,pred_test)
  precision_train = dc.metrics.precision_score(y_train,pred_train_labels)
  precision_test = dc.metrics.precision_score(y_test,pred_test_labels)
  recall_train = dc.metrics.recall_score(y_train,pred_train_labels)
  recall_test = dc.metrics.recall_score(y_test,pred_test_labels)
  results = []
  results.extend(list(model.evaluate(train_dataset, metrics).values()))
  results.extend([precision_train,recall_train])
  results.extend(list(model.evaluate(test_dataset, metrics).values()))
  results.extend([precision_test,recall_test])
  return results

def applyCV(mode,crossValid,Jobid):
    if mode == 'attentiveFP':
        # model = dc.models.AttentiveFPModel
        feat = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    if mode == 'gcn':
        # model = dc.models.GCNModel(n_tasks=1)
        feat = dc.feat.MolGraphConvFeaturizer(use_edges=True)

    if mode == 'dag':
        # model = dc.models.GCNModel(n_tasks=1)
        feat = dc.feat.ConvMolFeaturizer()

    if mode == 'gat':
        feat = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    cv =5
    if crossValid == "5":
        cv = 5
    else:
        cv = 10
    prnt_fld = Path.home()
    fld_path = os.path.join(prnt_fld, "ml_olfa", "all_jobs", str(Jobid), "data", "user_data.csv")
    print(fld_path)
    if not os.path.exists(fld_path):
        print("path doesnt exist")
    else:
        print("user data path exist")

    data = pd.read_csv(fld_path)

    # data = pd.read_csv('DIR_Train_MOA.csv')

    X = data['SMILES']
    y = data['Activation Status']
    skf = StratifiedKFold(n_splits = cv, shuffle=True)  # Stratified KFold split to maintain same ratio of classes in all the splits
    metrics = [dc.metrics.Metric(dc.metrics.roc_auc_score), dc.metrics.Metric(dc.metrics.kappa_score),
               dc.metrics.Metric(dc.metrics.accuracy_score), dc.metrics.Metric(dc.metrics.f1_score)]
    i = 1
    results = []
    train_f1 = []
    test_f1 = []
    for train_index, test_index in skf.split(X, y):  # iterating through the 10 splits
        print("Processing {}/10 Folds".format(i))
        x_train, x_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        # print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
        sampler = RandomOverSampler(sampling_strategy='minority',
                                    random_state=42)  # Oversampling training data because of class imbalance
        x_train = x_train.to_numpy()
        x_train = x_train.reshape((-1, 1))
        X_over, y_over = sampler.fit_resample(x_train, y_train)
        data_oversampled = pd.DataFrame(columns=['Activation Status', 'SMILES'])
        data_oversampled['Activation Status'] = y_over
        data_oversampled['SMILES'] = X_over[:, 0]
        # data_oversampled.to_csv('data_oversampled.csv'.format(i), index=False)
        test_data = pd.DataFrame(columns=['Activation Status', 'SMILES'])
        test_data['Activation Status'] = y_test
        test_data['SMILES'] = x_test
        # test_data.to_csv('test_data.csv'.format(i), index=False)
        
        if mode == 'gcn':
            train_dataset, y_train = generate_dataset(data_oversampled, feat)
            test_dataset, y_test = generate_dataset(test_data, feat)
            model = dc.models.GCNModel(mode='classification', n_tasks=1, batch_size=32, learning_rate=0.001,num_layers=2, graph_feat_size=200, dropout=0.5)
            model.fit(train_dataset, nb_epoch=5)
            result = evaluate(model, metrics, train_dataset, test_dataset, y_train, y_test)
        if mode == 'dag':
            train_dataset, y_train = generate_dataset_DAG(data_oversampled, feat)
            test_dataset, y_test = generate_dataset_DAG(test_data, feat)
            model = dc.models.DAGModel(mode='classification', n_tasks=1, batch_size=32, learning_rate=0.001)
            model.fit(train_dataset, nb_epoch=5)
            result = evaluate_DAG(model, metrics, train_dataset, test_dataset, y_train, y_test)
        if mode == 'attentiveFP':
            train_dataset, y_train = generate_dataset(data_oversampled, feat)
            test_dataset, y_test = generate_dataset(test_data, feat)
            model = dc.models.AttentiveFPModel(mode='classification', n_tasks=1, batch_size=32, learning_rate=0.001,
                                               num_layers=2, graph_feat_size=200, dropout=0.5)
            model.fit(train_dataset, nb_epoch=5)  # training model for 100 epochs
            result = evaluate(model, metrics, train_dataset, test_dataset, y_train, y_test)
        if mode == 'gat':
            train_dataset, y_train = generate_dataset(data_oversampled, feat)
            test_dataset, y_test = generate_dataset(test_data, feat)
            model = dc.models.GATModel(mode='classification', n_tasks=1, batch_size=32, learning_rate=0.001,
                                       num_layers=2, graph_feat_size=200, dropout=0.5)
            model.fit(train_dataset, nb_epoch=5)  # training model for 100 epochs
            result = evaluate(model, metrics, train_dataset, test_dataset, y_train, y_test)
        # result = evaluate(model, metrics, train_dataset, test_dataset, y_train, y_test)
        print(result)
        train_f1.append(result[3])
        test_f1.append(result[9])
        results.append(result)
        i += 1

    print("Computing Results...")
    auc_train = 0
    auc_test = 0
    kappa_train = 0
    kappa_test = 0
    acc_train = 0
    acc_test = 0
    f1_train = 0
    f1_test = 0
    precision_train = 0
    recall_train = 0
    precision_test = 0
    recall_test = 0
    i = 1
    results_dict = dict()
    for res in results:
        results_dict['FOLD ' + str(i)] = dict()

        auc_train += res[0]
        results_dict['FOLD ' + str(i)]['auc_train'] = res[0]
        kappa_train += res[1]
        results_dict['FOLD ' + str(i)]['kappa_train'] = res[1]
        acc_train += res[2]
        results_dict['FOLD ' + str(i)]['acc_train'] = res[2]
        f1_train += res[3]
        results_dict['FOLD ' + str(i)]['f1_train'] = res[3]
        precision_train += res[4]
        results_dict['FOLD ' + str(i)]['precision_train'] = res[4]
        recall_train += res[5]
        results_dict['FOLD ' + str(i)]['recall_train'] = res[5]
        auc_test += res[6]
        results_dict['FOLD ' + str(i)]['auc_test'] = res[6]
        kappa_test += res[7]
        results_dict['FOLD ' + str(i)]['kappa_test'] = res[7]
        acc_test += res[8]
        results_dict['FOLD ' + str(i)]['acc_test'] = res[8]
        f1_test += res[9]
        results_dict['FOLD ' + str(i)]['f1_test'] = res[9]
        precision_test += res[10]
        results_dict['FOLD ' + str(i)]['precision_test'] = res[10]
        recall_test += res[11]
        results_dict['FOLD ' + str(i)]['recall_test'] = res[11]
        i = i + 1
    print(
        "Training Results \n AUC: {} Cohen Kappa: {} ACC: {} F1: {} Precision : {} Recall : {} \n Test Results \n AUC: {} Cohen Kappa: {} ACC: {} F1: {} Precision : {} Recall : {}".format(
            auc_train / 10, kappa_train / 10, acc_train / 10, f1_train / 10, precision_train / 10,
            recall_train / 10,
            auc_test / 10, kappa_test / 10, acc_test / 10, f1_test / 10, precision_test / 10, recall_test / 10))
    Input_data_cross_validation = pd.DataFrame(results_dict)
    # Input_data_cross_validation.to_csv('Input_data_cross_validation.csv')
    all_jobs_fld = fetch_all_jobs_path()
    job_fld = os.path.join(all_jobs_fld, str(Jobid))
    data_fld_path = os.path.join(job_fld, "result")
    os.makedirs(data_fld_path, exist_ok=True)
    Input_data_cross_validation_path = os.path.join(data_fld_path, "Input_data_cross_validation.csv")
    Input_data_cross_validation.to_csv(Input_data_cross_validation_path)


# applyCV('gcn',10,12)