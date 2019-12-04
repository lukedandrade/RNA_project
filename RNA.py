from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer
import pandas as pd
from datetime import datetime, timedelta


dataframe = pd.read_csv('resultados_oficial.csv')
dataset = ClassificationDataSet(2, 1, nb_classes=3)

for entry in dataframe.to_dict(orient='records'):
    dataset.addSample([entry['tam_frase'], entry['timedelta']], [entry['age_id']])

#particionando dataset para treino
train_data, part_data = dataset.splitWithProportion(0.7)

#particao para teste/validacao
test_data, val_data = part_data.splitWithProportion(0.5)


network = buildNetwork(dataset.indim, 4, dataset.outdim)
trainer = BackpropTrainer(network, dataset=train_data, learningrate=0.01, momentum=0.1, verbose=True)

training_errors, val_errors = trainer.trainUntilConvergence(dataset=train_data)

out = network.activateOnDataset(test_data)
for i in range(len(out)):
    print("out: %f, correct: %f" % (out[i], test_data['target'][i]))

'''
while True:
    while True:
        start = datetime.now()
        user_string = "Eu quero fotos do homem-aranha, na minha mesa, agora."
        user_input = input("Digite a frase abaixo:\n\n %s \n\nPara sair digite 'exit'." % (user_string))

        if user_string == user_input:
            tempo_corrido = datetime.now() - start
            our_out = network.activate([len(user_string), tempo_corrido.seconds])
            print("O output é: %f" % (our_out[0]))

        elif user_input == 'exit':
            break

        else:
            print('Frases não estão iguais, por favor, tente de novo')
    user_input = input("Deseja continuar? Digite 'exit' novamente para encerrar o programa.")

    if user_input == 'exit':
        break
'''