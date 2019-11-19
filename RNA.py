from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer
import pandas as pd

dataframe = pd.read_csv('resultados.csv')
dataset = ClassificationDataSet(2, 1, nb_classes=3)

for entry in dataframe.to_dict(orient='records'):
    dataset.addSample([entry['tam_frase'], entry['timedelta']], [entry['age_id']])

#particionando dataset para treino
train_data, part_data = dataset.splitWithProportion(0.7)

#particao para teste/validacao
test_data, val_data = part_data.splitWithProportion(0.5)


network = buildNetwork(dataset.indim, 4, dataset.outdim)
trainer = BackpropTrainer(network, dataset=train_data, learningrate=0.01, momentum=0.1, verbose=True)

training_errors, val_errors = trainer.trainUntilConvergence(dataset=train_data, maxEpochs=1000)

out = network.activateOnDataset(test_data)
for i in range(len(out)):
    print("out: %f, correct: %f" % (out[i], test_data['target'][i]))
