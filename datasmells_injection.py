from pathlib import Path
from random import random, randint
from time import sleep

import numpy as np
import requests
import pandas as pd
from bs4 import BeautifulSoup
import os
import scipy.stats as sp

def inject_missing_values(df, column_name, path):
    tmp_df = df.copy()
    tmp_df[column_name] = np.NaN
    tmp_df.to_csv(path, index=False)

def invert_sign(x):
    y = x
    if (x != 0):
        return -y
    else:
        return y
    
def inject_suspect_sign(df, column_name, path):
    tmp_df = df.copy()
    print(tmp_df[column_name].dtype)
    tmp_df[column_name] = tmp_df[column_name].apply(invert_sign)
    print(tmp_df[column_name].dtype)
    tmp_df.to_csv(path, index=False)

def floating_to_string(x):
    return '\''+str(x)+'\''

def inject_floating_point_number_as_string(df, column_name, path):
    tmp_df = df.copy()
    tmp_df[column_name] = tmp_df[column_name].apply(floating_to_string)
    tmp_df.to_csv(path, index=False)

def change_casing(x):
    y = list(x.lower())
    if len(y)==2:
        y[1] = y[1].upper()
        return "".join(y)
    if '_' in y: #Caso specifico per il dataset nursery
        index_ = y.index("_")
        index_modify1 =  randint(2, index_ - 1)
        index_modify2 =  randint(index_ + 1, len(y) - 2)
        y[index_modify1] = y[index_modify1].upper()
        y[index_modify2] = y[index_modify2].upper()
    else:
        strings = x.lower().split(" ")
        finalstring = ""
        for s in strings:
            tmp = list(s)
            if len(tmp) > 1:
                first_index_base = 0
                if len(tmp) > 2:
                    first_index_base = 1
                index = randint(first_index_base, len(tmp) - 2)
                while tmp[index] == " " or tmp[index] == "_":
                    index = randint(first_index_base, len(y) - 2)
                tmp[index] = tmp[index].upper()
            tmp = "".join(tmp)
            finalstring = (finalstring+" "+tmp)
        y = finalstring
        
    return "".join(y)

def inject_casing(df, column_name, path):
    tmp_df = df.copy()
    tmp_df[column_name] = tmp_df[column_name].apply(change_casing)
    tmp_df.to_csv(path, index=False)

#fonte: https://stackoverflow.com/questions/51471672/reverse-z-score-pandas-dataframe


def calculate_extreme_values(df, column_name, path, round, original_df):
    tmp_df = df.copy()
    from sklearn.preprocessing import StandardScaler

    zscore = StandardScaler()
    tmp = []
    tmp_df_original = original_df.copy()
    index = tmp_df_original.select_dtypes(include='number').columns.get_loc(column_name)
    out = zscore.fit_transform(tmp_df_original.select_dtypes(include='number'))
    for i in range(len(out)):
        out[i][index] = 4
    x = zscore.inverse_transform(out)
    for i in range(len(x)):
        y = (x[i][index])
        if round:
            y = np.round(y).astype('int64')
        tmp.append(y)
    return_df = pd.DataFrame(tmp)
    return_df[1] = tmp_df[column_name]
    tmp_df[column_name] = return_df[0].get(0)
    tmp_df.to_csv(path, index=False)
    print("Valore per Extreme Value Smell: "+ str(return_df[0].get(0)))


if __name__ == "__main__":
    original_df_l_1 = pd.read_csv("./cleaned_datasets/df_l_1.csv", index_col=False)
    df_l_1 = pd.read_csv("./outputSDV/df_l_1_SDV.csv", index_col=False)
    df_l_1_95 = pd.read_csv("outputSDV/df_l_1_95_SDV.csv", index_col=False)
    df_l_1_99 = pd.read_csv("outputSDV/df_l_1_99_SDV.csv", index_col=False)
    original_df_l_2 = pd.read_csv("./cleaned_datasets/df_l_2.csv", index_col=False)
    df_l_2 = pd.read_csv("./outputSDV/df_l_2_SDV.csv", index_col=False)
    df_l_2_95 = pd.read_csv("outputSDV/df_l_2_95_SDV.csv", index_col=False)
    df_l_2_99 = pd.read_csv("outputSDV/df_l_2_99_SDV.csv", index_col=False)
    original_df_l_3 = pd.read_csv("./cleaned_datasets/df_l_3.csv", index_col=False)
    df_l_3 = pd.read_csv("./outputSDV/df_l_3_SDV.csv", index_col=False)
    df_l_3_95 = pd.read_csv("outputSDV/df_l_3_95_SDV.csv", index_col=False)
    df_l_3_99 = pd.read_csv("outputSDV/df_l_3_99_SDV.csv", index_col=False)
    original_df_m_1 = pd.read_csv("./cleaned_datasets/df_m_1.csv", index_col=False)
    df_m_1 = pd.read_csv("./outputSDV/df_m_1_SDV.csv", index_col=False)
    df_m_1_95 = pd.read_csv("outputSDV/df_m_1_95_SDV.csv", index_col=False)
    df_m_1_99 = pd.read_csv("outputSDV/df_m_1_99_SDV.csv", index_col=False)
    df_m_1_FP = pd.read_csv("outputSDV/df_m_1_FP_SDV.csv", index_col=False)
    original_df_m_2 = pd.read_csv("./cleaned_datasets/df_m_2.csv", index_col=False)
    df_m_2 = pd.read_csv("./outputSDV/df_m_2_SDV.csv", index_col=False)
    df_m_2_95 = pd.read_csv("outputSDV/df_m_2_95_SDV.csv", index_col=False)
    df_m_2_99 = pd.read_csv("outputSDV/df_m_2_99_SDV.csv", index_col=False)
    new_path="./outputDSInjection"
    if (not Path(new_path).exists()):
        os.mkdir(new_path)
    new_path_missing_values = new_path+"/MissingValues"
    if (not Path(new_path_missing_values).exists()):
        os.mkdir(new_path_missing_values)
    new_path_casing = new_path+"/Casing"
    if (not Path(new_path_casing).exists()):
        os.mkdir(new_path_casing)
    new_path_extreme_values = new_path+"/ExtremeValues"
    if (not Path(new_path_extreme_values).exists()):
        os.mkdir(new_path_extreme_values)
    new_path_suspect_sign = new_path+"/SuspectSign"
    if (not Path(new_path_suspect_sign).exists()):
        os.mkdir(new_path_suspect_sign)
    new_path_floating_point_number_as_string  = new_path+"/FloatingPointNumberAsString"
    if (not Path(new_path_floating_point_number_as_string).exists()):
        os.mkdir(new_path_floating_point_number_as_string)
    
    #df_l_1 - non ha variabili numeriche float
    inject_missing_values(df_l_1, "age", new_path_missing_values+"/df_l_1_1.csv")
    inject_casing(df_l_1, "major_industry_code", new_path_casing+"/df_l_1_1.csv")
    
    #Viene utilizzata la feature "age" anche se non genera un Suspect Sign per ogni occorrenza a causa della presenza di alcuni 0. 
    #Il numero di data smell così generati non è abbastanza per poter essere rilevato sul dataset integrale, essendo presente in circa il 9,71% del dataset, invece del 10%
    inject_suspect_sign(df_l_1, "age", new_path_suspect_sign+"/df_l_1_1.csv")
    
    #inject_extreme_values(df_l_1_95, "age", new_path_extreme_values+"/df_l_1_95_1.csv", True)
    calculate_extreme_values(df_l_1_99, "age", new_path_extreme_values + "/df_l_1_99_1.csv", True, original_df_l_1)
    
    #df_l_2 - non ha variabili numeriche float
    inject_missing_values(df_l_2, "number_inpatient", new_path_missing_values+"/df_l_2_1.csv")
    
    inject_casing(df_l_2, "insulin", new_path_casing+"/df_l_2_1.csv")
    
    #Viene utilizzata la seconda feature numerica più importante (discharge_disposition_id) perché la prima (number_inpatient) ha maggiormente 0 per 
    #per cui non si può manifestare lo smell generando dati realistici
    inject_suspect_sign(df_l_2, "discharge_disposition_id", new_path_suspect_sign+"/df_l_2_1.csv")
    calculate_extreme_values(df_l_2_99, "number_inpatient", new_path_extreme_values + "/df_l_2_99_1.csv", True, original_df_l_2)

    #df_l_3 - non ha variabili numeriche
    inject_casing(df_l_3, "health", new_path_casing+"/df_l_3_1.csv")

    #df_m_1 - variabili categoriche e numeriche decimali, con e senza virgola
    inject_missing_values(df_m_1, "combine", new_path_missing_values+"/df_m_1_1.csv")
    inject_casing(df_m_1, "position", new_path_casing+"/df_m_1_1.csv")
    print("df_m_1")
    inject_suspect_sign(df_m_1, "combine", new_path_suspect_sign+"/df_m_1_1.csv")
    inject_floating_point_number_as_string(df_m_1_FP, "combine", new_path_floating_point_number_as_string+"/df_m_1_1.csv")
    calculate_extreme_values(df_m_1_99, "combine", new_path_extreme_values + "/df_m_1_99_1.csv", False, original_df_m_1)

    #df_m_2 - non ha variabili categoriche
    inject_missing_values(df_m_2, "Course", new_path_missing_values + "/df_m_2_1.csv")
    print("df_m_2")
    inject_suspect_sign(df_m_2, "Course", new_path_suspect_sign + "/df_m_2_1.csv")
    calculate_extreme_values(df_m_2_99, "Course", new_path_extreme_values + "/df_m_2_99_1.csv", True, original_df_m_2)