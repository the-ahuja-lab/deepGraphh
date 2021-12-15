import pandas as pd
import deepchem as dc
import sys
import os

sys.path.append(os.path.abspath(os.path.join('..', 'Dataset')))


data = pd.read_csv('test_data.csv')
data = data[['SMILES', 'Activation Status']]
with dc.utils.UniversalNamedTemporaryFile(mode='w') as tmpfile:
    data.to_csv(tmpfile.name)
    loader = dc.data.CSVLoader(["Activation Status"], feature_field="SMILES",
                               featurizer=dc.feat.MolGraphConvFeaturizer())
    dataset = loader.create_dataset(tmpfile.name)
print(dataset)