"""
Created on Thu Mar 18 22:35:00

@author: gabesness
"""

import os
import pandas as pd
import numpy as np
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
disgust..................9+15+16
mockery..................12+14 (unilateral)*
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

parser = argparse.ArgumentParser(description="select best frames from videos given an emotion and a database")

parser.add_argument("database", help="select dabatase (br/co)")
parser.add_argument("emotion", help="select an emotion")

args = parser.parse_args()

if args.database == "br":
    directory = "../facs/brasil"  # esta variavel sera input
elif args.database == "co":
    directory = "../facs/colombia"
else:
    raise ValueError
dir_subjects = directory + "/subjects"
dir_timestamps = directory + "/timestamps"

# criar listas dos arquivos de cada pasta
subs = [x for x in os.listdir(dir_subjects) if x.endswith(".csv")]
times_s1 = [x for x in os.listdir(dir_timestamps) if x.endswith("s1.csv")]
times_s2 = [x for x in os.listdir(dir_timestamps) if x.endswith("s2.csv")]
subs.sort()
times_s1.sort()
times_s2.sort()

print(len(subs))
print(len(times_s1))
print(len(times_s2))
lista_tempos = [times_s1, times_s2]
#print(lista_tempos)

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

#print(pd.DataFrame(posicoes_emocoes))

intervalos_au = {
    "alegria": ["AU06_c", "AU06_r", "AU12_c", "AU12_r"],
    "nojo": ["AU09_c", "AU09_r", "AU15_c", "AU15_r"], # missing AU16
    "tristeza": ["AU01_c", "AU01_r", "AU04_c", "AU04_r", "AU15_c", "AU15_r"],
    "deboche": ["AU12_c", "AU12_r", "AU14_c", "AU14_r"],
    "raiva": ["AU04_c", "AU04_r", "AU05_c", "AU05_r", "AU07_c", "AU07_r", "AU23_c", "AU23_r"],
    "surpresa": ["AU01_c", "AU01_r", "AU02_c", "AU02_r", "AU05_c", "AU05_r", "AU26_c", "AU26_r"],
    "medo": ["AU01_c", "AU01_r", "AU02_c", "AU02_r", "AU04_c", "AU04_r", "AU05_c", "AU05_r", "AU07_c", "AU07_r", "AU20_c", "AU20_r", "AU26_c", "AU26_r"]
}

# HANDLING OF TIME FILES

def main(database, emotion):

    # inicializamos um dicionario vazio que contera as informacoes dos arquivos
    dic = {
        "subject": [],
        "s1": [],
        "s2": []
    }

    for i in range(2):
        #print(i)
        for t in lista_tempos[i]:
            #print("current file: " + t)
            fpath = dir_timestamps + "/" + t
            temp = pd.read_csv(fpath, header=None)
            #print(temp)
            time = temp.iloc[posicoes_emocoes[emotion][i]]
            #print(time)
            sec = time[0] * 60 + time[1]
            sub = t[6] + t[7]
            if sub not in dic["subject"]:
                dic["subject"].append(sub)
            if i == 0:
                dic["s1"].append(sec)
            else:
                dic["s2"].append(sec)
    print("files read successfully!")


    tabela_tempos = pd.DataFrame(dic)
    #print(tabela_tempos)
    #print(len(tabela_tempos))
    return(tabela_tempos)


# SELECIONAR INTERVALOS DOS ARQUIVOS CSV DOS PARTICIPANTES

def select_interval(table):
    intervalos = []
    for s in subs:
        fpath = dir_subjects + "/" + s
        #print(s)
        if args.database == "br":
            sub_number = s[0:2]
        elif args.database == "co":
            sub_number = s[7:9]
        else:
            raise ValueError
        print("current subject: " + sub_number)
        t1 = int(table.loc[table["subject"] == sub_number, "s1"])
        t2 = int(table.loc[table["subject"] == sub_number, "s2"])
        #print(t1)
        #print(t2)
        temp = pd.read_csv(fpath)
        int1 = temp.loc[(temp["timestamp"] >= t1) & (temp["timestamp"] < t1 + 7)] # intervalo da sessao 1
        int2 = temp.loc[(temp["timestamp"] >= t2) & (temp["timestamp"] < t2 + 7)] # intervalo da sessao 2
        intervalo = pd.concat([int1, int2]) # intervalo total da emocao
        #print(intervalo)
        intervalos.append(intervalo)

    return intervalos


def select_frames(lista, emotion):
    cols = ["frame", "timestamp"]
    cols += intervalos_au[emotion]
    print(cols)
    for i in lista:
        temp = i.loc[:, cols]
        print(temp)




#main(args.database, args.emotion)
#select_interval(main(args.database, args.emotion))
select_frames(select_interval(main(args.database, args.emotion)), args.emotion)
