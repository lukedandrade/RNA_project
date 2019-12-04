from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer
import pandas as pd

dataframe = pd.read_csv('resultados_oficial.csv')
dataset = ClassificationDataSet(2, 1, nb_classes=3, class_labels=['jovem', 'adulto', 'idoso'])

for entry in dataframe.to_dict(orient='records'):
    dataset.addSample([entry['tam_frase'], entry['timedelta']], target=entry['age_id'])

train_data_temp, part_data_t = dataset.splitWithProportion(0.6)
test_data_temp, val_data_temp = part_data_t.splitWithProportion(0.5)

train_data = ClassificationDataSet(2, 1, nb_classes=3, class_labels=['jovem', 'adulto', 'idoso'])
for n in range(train_data_temp.getLength()):
    train_data.addSample(train_data_temp.getSample(n)[0], train_data_temp.getSample(n)[1])

test_data = ClassificationDataSet(2, 1, nb_classes=3, class_labels=['jovem', 'adulto', 'idoso'])
for n in range(test_data_temp.getLength()):
    test_data.addSample(test_data_temp.getSample(n)[0], test_data_temp.getSample(n)[1])

val_data = ClassificationDataSet(2, 1, nb_classes=3, class_labels=['jovem', 'adulto', 'idoso'])
for n in range(val_data_temp.getLength()):
    val_data.addSample(val_data_temp.getSample(n)[0], val_data_temp.getSample(n)[1])

train_data._convertToOneOfMany(bounds=[0, 1])
test_data._convertToOneOfMany(bounds=[0, 1])
val_data._convertToOneOfMany(bounds=[0, 1])

from pybrain.structure.modules import SoftmaxLayer
from pybrain.utilities import percentError

net = buildNetwork(train_data.indim, 5, train_data.outdim, outclass=SoftmaxLayer)

def show_weights(net):
    for mod in net.modules:
        for conn in net.connections[mod]:
            print(conn)
            for cc in range(len(conn.params)):
                print(conn.whichBuffers(cc), conn.params)
                print('\n')

#show_weights(net)

trainer = BackpropTrainer(net, dataset=train_data, learningrate=0.01, momentum=0.1)
trainer.trainUntilConvergence(maxEpochs=1000)

#show_weights(net)

out_test = net.activateOnDataset(test_data).argmax(axis=1)
print("Erro de teste: %f" % percentError(out_test, test_data['target'][:,0]))

out_val = net.activateOnDataset(val_data).argmax(axis=1)
print("Erro de validadação: %f" % percentError(out_val, val_data['target'][:,0]))

print("-------------------------------------------\nTeste:")
print("saída da rede:\t", out_test)
print("correto      :\t", test_data['target'][:,0])


'''
from datetime import datetime, timedelta

while True:
    while True:
        start = datetime.now()
        user_string = "Eu quero fotos do homem-aranha, na minha mesa, agora."
        user_input = input("Digite a frase abaixo:\n\n %s \n\nPara sair digite 'exit'." % (user_string))

        if user_string == user_input:
            tempo_corrido = datetime.now() - start
            our_out = net.activate([len(user_string), tempo_corrido.seconds])
            print("O output é: %f" % (our_out[0]))

        elif user_input == 'exit':
            break

        else:
            print('Frases não estão iguais, por favor, tente de novo')
    user_input = input("Deseja continuar? Digite 'exit' novamente para encerrar o programa.")

    if user_input == 'exit':
        break
'''