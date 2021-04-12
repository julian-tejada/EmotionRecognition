"""
Created on Thu Mar 18 22:35:00

@author: gabesness
"""

import os
import pandas as pd
import argparse

# Codificar as combinacoes de AUs correspondentes as emocoes
# variaveis do argparse: emocao a ser treinada e banco de dados (BR/CO)

"""
EMOTION                  AUs
happiness................6+12
sadness..................1+4+15
surprise.................1+2+5+26
fear.....................1+2+4+5+7+20+26
anger....................4+5+7+23
disgust..................12+14 (unilateral)*
"""
# possible limitation: OpenFace does not recognise all necessary AUs.
# duracao das emocoes: 5 a 8 segundos
# identificar os melhores frames (com maior intensidade dos AUs)
# top 5 em ordem descendente, do melhor para o pior
# deteccao de AUs eh condicao NECESSARIA para incluir
"""
queremos passar como parametros uma emocao a ser analisada e um determinado banco de dados (brasil ou colombia);
em seguida, vamos procurar nos arquivos de tempo os intervalos de tempo onde essas emocoes estao nos respectivos arquivos
para isso podemos criar uma lista com os arquivos de sujeitos e de tempos e ordena-las para que se possa fazer uma associacao 1:1
"""

def main(database, emotion):
    if database == "br":
        directory = "../facs/brasil"  # esta variavel sera input
    elif database == "co":
        directory = "../facs/colombia"
    else:
        raise ValueError
    dir_subjects = directory + "/subjects"
    dir_timestamps = directory + "/timestamps"

    # criar listas dos arquivos de cada pasta
    times_s1 = [x for x in os.listdir(dir_timestamps) if x.endswith("s1.csv")]
    times_s2 = [x for x in os.listdir(dir_timestamps) if x.endswith("s2.csv")]
    times_s1.sort()
    times_s2.sort()

    # o dicionario abaixo lista a ordem das emocoes nas sessoes 1 e 2 (por isso os valores sao listas de dois elementos, sendo o primeiro a posicao na sessao 1 e o segundo na sessao 2)
    posicoes_emocoes = {
        "alegria": [0,3],
        "nojo": [1,5],
        "tristeza": [2,2],
        "deboche": [3,1],
        "raiva": [4,0],
        "surpresa": [5,4],
        "medo": [6,6]
    }

    # inicializamos um dicionario vazio que contera as informacoes dos arquivos
    dic = {
        "subject": [],
        "time_s1": [],
        "time_s2": []
    }

    lista_tempos = [times_s1, times_s2]
    #print(lista_tempos)
    """
    fpath = dir_timestamps + "/" + times_s1[0]
    temp = pd.read_csv(fpath, header=None)
    print(temp)
    time = temp.iloc[posicoes_emocoes[emotion][0]]
    print(time)
    """

    for i in range(2):
        #print(i)
        for t in lista_tempos[i]:
            print("current file: " + t)
            fpath = dir_timestamps + "/" + t
            temp = pd.read_csv(fpath, header=None)
            #print(temp)
            time = temp.iloc[posicoes_emocoes[emotion][i]]
            sec = time[0] * 60 + time[1]
            sub = t[6] + t[7]
            if sub not in dic["subject"]:
                dic["subject"].append(sub)
            if i == 0:
                dic["time_s1"].append(sec)
            else:
                dic["time_s2"].append(sec)
    print("files read successfully!")


    tabela_tempos = pd.DataFrame(dic)
    print(tabela_tempos)
    print(len(tabela_tempos))

parser = argparse.ArgumentParser(description="select best frames from videos given an emotion and a database")

parser.add_argument("database", help="select dabatase (br/co)")
parser.add_argument("emotion", help="select an emotion")

args = parser.parse_args()

main(args.database, args.emotion)

# main("co", "alegria")
