from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer
import pandas as pd

dataframe = pd.read_csv('resultados.csv')
dataset = ClassificationDataSet(2, 1, nb_classes=3)

for entry in dataframe.to_dict(orient='records'):
    dataset.addSample([entry['tam_frase'], entry['timedelta']], [entry['age_id']])

#particionando dataset para treino
train_data_temp, part_data_temp = dataset.splitWithProportion(0.7)

#particao para teste/validacao
test_data_temp, val_data_temp = part_data_temp.splitWithProportion(0.5)

#conversao dataset treino
train_data = ClassificationDataSet(2, 1, nb_classes=3)
for n in range(train_data_temp.getLength()):
    train_data.addSample(train_data_temp.getSample(n)[0], train_data_temp.getSample(n)[1])

#conversao dataset teste
test_data = ClassificationDataSet(2, 1, nb_classes=3)
for n in range(test_data_temp.getLength()):
    test_data.addSample(test_data_temp.getSample(n)[0], test_data_temp.getSample(n)[1])

#conversao dataset validacao
val_data = ClassificationDataSet(2, 1, nb_classes=3)
for n in range(val_data_temp.getLength()):
    val_data.addSample(val_data_temp.getSample(n)[0], val_data_temp.getSample(n)[1])

print(train_data.outdim)