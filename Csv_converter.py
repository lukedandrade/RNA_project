import csv

#Tamanho da frase | Tempo de digitação | Identificação de idade

data_txt = open('dados.txt', mode='r')
data_csv = open('resultados.csv', mode='w')
fieldnames = ['tam_frase', 'timedelta', 'age_id']


csv_writer = csv.DictWriter(data_csv, fieldnames=fieldnames)
csv_writer.writeheader()


for line in data_txt.readlines():
    aux = line.split('|')
    dados = {
        'tam_frase' : int(aux[0]),
        'timedelta' : float(aux[1]),
        'age_id' : int(aux[2])
    }

    csv_writer.writerow(dados)

data_txt.close()
data_csv.close()