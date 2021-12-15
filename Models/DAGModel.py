import pandas as pd
import deepchem as dc
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from deepchem.models import DAGModel
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from itertools import cycle
from scipy import interp
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import init_db as database
import sys
import os
from pathlib import Path
import numpy as np
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

class DagModel:
  def __init__(self,job_id,created,email,job_name,status,mode,n_tasks,n_classes,dropout,no_atom_features,max_atoms,n_graph_feat,n_outputs,layer_sizes_gather,layer_sizes,uncertainity,self_loop,learning_rate,epoch,csv_name):
    self.job_id = job_id
    self.created = created
    self.email = email
    self.job_name = job_name
    self.status = status
    self.mode = mode
    self.n_tasks = n_tasks
    self.no_classes = n_classes
    self.dropout = dropout
    self.no_atom_features = no_atom_features
    self.max_atoms = max_atoms
    self.n_graph_feat = n_graph_feat
    self.n_outputs = n_outputs
    self.layer_sizes_gather = layer_sizes_gather
    self.layer_sizes = layer_sizes
    self.uncertainity = uncertainity
    self.self_loop = self_loop
    self.learning_rate = learning_rate
    self.epoch = epoch
    self.csv_name = csv_name
    self.test_gy = None
    self.train_gy = None
    self.pred_test_y = None
    self.pred_train_y = None
    self.query_data_df = None
    self.user_test_data = None


  def load_csv(self):
    dataset_url = 'Dataset/' + self.csv_name
    print("csv name of the experiment")
    print(dataset_url)

    prnt_fld = Path.home()
    fld_path = os.path.join(prnt_fld, "ml_olfa", "all_jobs", str(self.job_id), "data", "user_data.csv")
    print(fld_path)
    if not os.path.exists(fld_path):
      print("path doesnt exist")
    else:
      print("user data path exist")

    data = pd.read_csv(fld_path)
    data = data[['SMILES', 'Activation Status']]
    with dc.utils.UniversalNamedTemporaryFile(mode='w') as tmpfile:
        data.to_csv(tmpfile.name)
        loader = dc.data.CSVLoader(["Activation Status"], feature_field="SMILES",
                                     featurizer=dc.feat.ConvMolFeaturizer())
        dataset = loader.create_dataset(tmpfile.name)
    trans = dc.trans.DAGTransformer(max_atoms=50)
    dataset = trans.transform(dataset)
    splitter = dc.splits.RandomSplitter()
    train_dataset, test_dataset = splitter.train_test_split(dataset)
    test_y = test_dataset.y
    test_y.tolist()
    train_y = train_dataset.y
    train_y.tolist()
    self.test_gy = test_y
    self.train_gy = train_y
    database.updateCurrentStatus(self.job_id, "dag", 3)
    return train_dataset, test_dataset

  def load_test_data(self):
    # sys.path.append(os.path.abspath(os.path.join('..', 'Dataset')))
    dataset_url = 'test_data.csv'

    prnt_fld = Path.home()
    fld_path = os.path.join(prnt_fld, "ml_olfa", "all_jobs", str(self.job_id), "data", "test_data.csv")
    print(fld_path)
    if not os.path.exists(fld_path):
      print("path doesnt exist")
    else:
      print("test data path exist")

    data = pd.read_csv(fld_path)

    # data = pd.read_csv(dataset_url)
    featurizer = dc.feat.ConvMolFeaturizer()
    data = data[['SMILES', 'Activation Status']]
    ret = featurizer.featurize(data['SMILES'])
    indexes_notF = list()
    for i in range(0, len(data)):
      if not ret[i]:
        indexes_notF.append(i)
    data = data.drop(indexes_notF)
    self.query_data_df = data
    with dc.utils.UniversalNamedTemporaryFile(mode='w') as tmpfile:
      data.to_csv(tmpfile.name)
      loader = dc.data.CSVLoader(["Activation Status"], feature_field="SMILES",
                                 featurizer=dc.feat.ConvMolFeaturizer())
      dataset = loader.create_dataset(tmpfile.name)
    trans = dc.trans.DAGTransformer(max_atoms=50)
    dataset = trans.transform(dataset)
    print(dataset)
    return dataset

  def write_user_result(self,label3):
    dataset_url = 'test_data.csv'

    prnt_fld = Path.home()
    fld_path = os.path.join(prnt_fld, "ml_olfa", "all_jobs", str(self.job_id), "data", "test_data.csv")
    print(fld_path)
    if not os.path.exists(fld_path):
      print("path doesnt exist")
    else:
      print("test data path exist")
    print("query data df")
    data = self.query_data_df
    print("query data df")
    # data = pd.read_csv(dataset_url)
    data['Activation Status'] = label3

    all_jobs_fld = fetch_all_jobs_path()
    # create the respective job id folder inside all jobs folder
    job_fld = os.path.join(all_jobs_fld, str(self.job_id))
    data_fld_path = os.path.join(job_fld, "result")
    os.makedirs(data_fld_path, exist_ok=True)
    result_test_data = os.path.join(data_fld_path, "Query_data_prob_matrix.csv")
    self.query_data_df.to_csv(result_test_data)

  def fit_predict(self):
    train_dataset, test_dataset = self.load_csv()
    featurizer = dc.feat.MolGraphConvFeaturizer()
    tasks = ['1', '0']
    self.no_classes = int(self.no_classes)
    model = DAGModel(mode='classification', n_tasks=1, batch_size=32, learning_rate=0.001,n_classes = self.no_classes)
    model.fit(train_dataset, nb_epoch=5)
    print("model Dag trained")
    database.updateCurrentStatus(self.job_id, "dag", 4)
    predict_label = model.predict(test_dataset)
    print("model Dag train predicted")
    self.predict_label = predict_label

    database.updateCurrentStatus(self.job_id, "dag", 5)
    print("model Dag test to be predicted")
    # predict_train = model.predict(train_dataset)
    print("model Dag test predicted")

    database.updateCurrentStatus(self.job_id, "dag", 6)
    # self.predict_train = predict_train

    self.user_test_data = self.load_test_data()
    user_predict_data = model.predict(self.user_test_data)
    self.user_predict_data = user_predict_data

    predict_label = list(predict_label)
    label = np.zeros(len(predict_label))
    for i in range(0, len(predict_label)):
      indx = np.where(predict_label[i][0] == np.amax(predict_label[i][0]))
      label[i] = indx[0]

    # predict_train = list(predict_train)
    # label2 = np.zeros(len(predict_train))
    # for i in range(0, len(predict_train)):
    #   indx = np.where(predict_train[i][0] == np.amax(predict_train[i][0]))
    #   label2[i] = indx[0]
    label2 = 1

    user_predict_data = list(user_predict_data)
    label3 = np.zeros(len(user_predict_data))
    for i in range(0, len(user_predict_data)):
      indx = np.where(user_predict_data[i][0] == np.amax(user_predict_data[i][0]))
      label3[i] = indx[0]

    pred = np.array(user_predict_data)
    print(user_predict_data)
    print(pred)
    for i in range(0, len(pred[0])):
      ithcol = pred[:, i]
      stringClass = 'class ' + str(i)
      self.query_data_df[stringClass] = ithcol

    self.write_user_result(label3)
    self.pred_test_y = label
    self.pred_train_y = label2
    return model , label2, label

  def results(self):
    test_y = self.test_gy
    label = self.pred_test_y

    cm = confusion_matrix(test_y, label)
    cls = np.arange(self.no_classes).astype('str')
    database.updateCurrentStatus(self.job_id, "dag", 7)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cls[:len(cm[0])])
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)

    all_jobs_fld = fetch_all_jobs_path()
    # create the respective job id folder inside all jobs folder
    job_fld = os.path.join(all_jobs_fld, str(self.job_id))
    data_fld_path = os.path.join(job_fld, "result")
    os.makedirs(data_fld_path, exist_ok=True)
    confMatrix = os.path.join(data_fld_path, "Input_data_test_confusion_matrix.png")

    plt.savefig(confMatrix)

    test_y = self.test_gy
    train_y = self.train_gy
    makingtestLabel = np.zeros((len(test_y), self.no_classes))
    makingtrainLabel = np.zeros((len(train_y), self.no_classes))
    for i in range(0, len(train_y)):
      x = train_y[i]
      makingtrainLabel[i][int(x)] = 1
    for i in range(0, len(test_y)):
      x = test_y[i]
      makingtestLabel[i][int(x)] = 1

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    predict_label = self.predict_label
    predict_label = np.array(predict_label)
    for i in range(self.no_classes):
      clas_pred = list()
      for k in range(0, len(predict_label)):
        clas_pred.append(predict_label[k][0][i])
      fpr[i], tpr[i], _ = roc_curve(makingtestLabel[:, i], clas_pred)
      roc_auc[i] = auc(fpr[i], tpr[i])

    n_classes = self.no_classes
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
      mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves

    plt.figure(figsize=(20, 20))
    plt.plot(
      fpr["macro"],
      tpr["macro"],
      label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
      color="navy",
      linestyle=":",
      linewidth=4,
    )
    lw = 2
    colors = cycle(mcolors.CSS4_COLORS)
    for i, color in zip(range(n_classes), colors):
      plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=lw,
        label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
      )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Some extension of Receiver operating characteristic to multiclass")
    plt.legend(loc="best")

    print("roc curve list")
    print(fpr)
    print(tpr)
    database.updateCurrentStatus(self.job_id, "dag", 8)

    test_rocplot = os.path.join(data_fld_path, "Input_data_test_AUC_ROC.png")
    plt.savefig(test_rocplot)

    self.status = "Completed"

    names = np.arange(self.no_classes)
    names = names.tolist()

    test_dict2 = classification_report(test_y, label, labels=names, output_dict=True)
    clf_report = pd.DataFrame(test_dict2).iloc[:-1, :].T
    clf_report_path = os.path.join(data_fld_path, "Input_data_test_confusion_report_raw.csv")
    clf_report.to_csv(clf_report_path)
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(pd.DataFrame(test_dict2).iloc[:-1, :].T, annot=True, ax=ax)

    test_confusionReport = os.path.join(data_fld_path, "Input_data_test_confusion_report.jpg")
    plt.savefig(test_confusionReport)

    database.updateCurrentStatus(self.job_id, "dag", 9)
    return self.status
