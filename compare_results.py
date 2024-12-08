from pathlib import Path
from time import sleep
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Per il warning: "FutureWarning: Calling float on a single element Series is deprecated and will raise a TypeError in the future. Use float(ser.iloc[0]) instead"

import requests
import pandas as pd
import numpy as np 
from bs4 import BeautifulSoup
import os
def compare_results(df_base, df_compare, path_output):
    df_output = pd.DataFrame(columns=df_base.columns)
    for _, row in df_base.iterrows():
        row_compare = df_compare.loc[df_compare["Model"] == row["Model"]]
        df_output.loc[len(df_output)] = [row["Model"], float(row_compare["Accuracy"]) - float(row["Accuracy"]), 
                                         float(row_compare["Balanced Accuracy"]) - float(row["Balanced Accuracy"]), 
                                         float(row_compare["ROC AUC"]) - float(row["ROC AUC"]), 
                                         float(row_compare["F1 Score"]) - float(row["F1 Score"]), 
                                         float(row_compare["Time Taken"]) - float(row["Time Taken"])]
        
    df_output.to_csv(path_output, index=False)

def split_models(df_list, base_path_output):
    df_longest = df_list[0].copy()
    for df in df_list: #Solo df_m_1 e df_l_3 hanno il modello "NuSVC"
        if len(df) > len(df_longest):
            df_longest = df.copy()
    models_list = df_longest["Model"].unique()
    for model in models_list:
        df_output = pd.DataFrame(columns=df_longest.columns)
        for df in df_list:
            if model in df["Model"].unique():
                row = df.loc[df["Model"] == model]
                df_output.loc[len(df_output)] = [model, float(row["Accuracy"]), float(row["Balanced Accuracy"]), float(row["ROC AUC"]), float(row["F1 Score"]), float(row["Time Taken"])]
            
        df_output.to_csv(base_path_output+"/"+model+".csv", index=False)        


def create_heatmap_models_numeric(df_list, base_path_output):
    df_longest = df_list[0].copy()
    for df in df_list: #Solo df_m_1 e df_l_3 hanno il modello "NuSVC"
        if len(df) > len(df_longest):
            df_longest = df.copy()
    models_list = df_longest["Model"].unique()
    print("Models list:")
    print(models_list)
    heatmap_accuracy = np.zeros(shape = (len(models_list), 4))
    heatmap_bal_accuracy = np.zeros(shape = (len(models_list), 4))
    heatmap_f1_score = np.zeros(shape = (len(models_list), 4))
    heatmap_time = np.zeros(shape = (len(models_list), 4))
    i = 0
    for model in models_list:
        df_base = pd.read_csv("./outputComparisons/Models/BaseNumeric/"+model+".csv", index_col= False)
        df_base_synthetic = pd.read_csv("./outputComparisons/Models/BaseSyntheticNumeric/"+model+".csv", index_col= False)
        df_base_missingvalues = pd.read_csv("./outputComparisons/Models/MissingValues/"+model+".csv", index_col= False)
        df_base_suspectsign = pd.read_csv("./outputComparisons/Models/SuspectSign/"+model+".csv", index_col= False)
        #df_base_floatingpoint = pd.read_csv("./outputComparisons/Models/FloatingPointNumberAsString/"+model+".csv", index_col= False)
        new_row_accuracy = [np.mean(df_base["Accuracy"]), np.mean(df_base_synthetic["Accuracy"]), np.mean(df_base_missingvalues["Accuracy"]), np.mean(df_base_suspectsign["Accuracy"])]
        new_row_bal_accuracy = [np.mean(df_base["Balanced Accuracy"]), np.mean(df_base_synthetic["Balanced Accuracy"]), np.mean(df_base_missingvalues["Balanced Accuracy"]), np.mean(df_base_suspectsign["Balanced Accuracy"])]
        new_row_f1_score = [np.mean(df_base["F1 Score"]), np.mean(df_base_synthetic["F1 Score"]), np.mean(df_base_missingvalues["F1 Score"]), np.mean(df_base_suspectsign["F1 Score"])]
        new_row_time = [np.mean(df_base["Time Taken"]), np.mean(df_base_synthetic["Time Taken"]), np.mean(df_base_missingvalues["Time Taken"]), np.mean(df_base_suspectsign["Time Taken"])]
        heatmap_accuracy[i] = new_row_accuracy
        heatmap_bal_accuracy[i] = new_row_bal_accuracy
        heatmap_f1_score[i] = new_row_f1_score
        heatmap_time[i] = new_row_time
        i = i+1
    np.save(base_path_output+"accuracy.npy", heatmap_accuracy)
    np.save(base_path_output+"bal_accuracy.npy", heatmap_bal_accuracy)
    np.save(base_path_output+"f1_score.npy", heatmap_f1_score)
    np.save(base_path_output+"time.npy", heatmap_time)


def create_heatmap_models_casing(df_list, base_path_output):
    df_longest = df_list[0].copy()
    for df in df_list: #Solo df_m_1 e df_l_3 hanno il modello "NuSVC"
        if len(df) > len(df_longest):
            df_longest = df.copy()
    models_list = df_longest["Model"].unique()
    print("Models list:")
    print(models_list)
    heatmap_accuracy = np.zeros(shape = (len(models_list), 3))
    heatmap_bal_accuracy = np.zeros(shape = (len(models_list), 3))
    heatmap_f1_score = np.zeros(shape = (len(models_list), 3))
    heatmap_time = np.zeros(shape = (len(models_list), 3))
    i = 0
    for model in models_list:
        df_base = pd.read_csv("./outputComparisons/Models/BaseCasing/"+model+".csv", index_col= False)
        df_base_synthetic = pd.read_csv("./outputComparisons/Models/BaseSyntheticCasing/"+model+".csv", index_col= False)
        df_base_casing = pd.read_csv("./outputComparisons/Models/Casing/"+model+".csv", index_col= False)
        #df_base_floatingpoint = pd.read_csv("./outputComparisons/Models/FloatingPointNumberAsString/"+model+".csv", index_col= False)
        new_row_accuracy = [np.mean(df_base["Accuracy"]), np.mean(df_base_synthetic["Accuracy"]), np.mean(df_base_casing["Accuracy"])]
        new_row_bal_accuracy = [np.mean(df_base["Balanced Accuracy"]), np.mean(df_base_synthetic["Balanced Accuracy"]), np.mean(df_base_casing["Balanced Accuracy"])]
        new_row_f1_score = [np.mean(df_base["F1 Score"]), np.mean(df_base_synthetic["F1 Score"]), np.mean(df_base_casing["F1 Score"])]
        new_row_time = [np.mean(df_base["Time Taken"]), np.mean(df_base_synthetic["Time Taken"]), np.mean(df_base_casing["Time Taken"])]
        heatmap_accuracy[i] = new_row_accuracy
        heatmap_bal_accuracy[i] = new_row_bal_accuracy
        heatmap_f1_score[i] = new_row_f1_score
        heatmap_time[i] = new_row_time
        i = i+1
    np.save(base_path_output+"accuracy.npy", heatmap_accuracy)
    np.save(base_path_output+"bal_accuracy.npy", heatmap_bal_accuracy)
    np.save(base_path_output+"f1_score.npy", heatmap_f1_score)
    np.save(base_path_output+"time.npy", heatmap_time)


def create_heatmap_models(df_list, base_path_output):
    df_longest = df_list[0].copy()
    for df in df_list: #Solo df_m_1 e df_l_3 hanno il modello "NuSVC"
        if len(df) > len(df_longest):
            df_longest = df.copy()
    models_list = df_longest["Model"].unique()
    print("Models list:")
    print(models_list)
    heatmap_accuracy = np.zeros(shape = (len(models_list), 5))
    heatmap_bal_accuracy = np.zeros(shape = (len(models_list), 5))
    heatmap_f1_score = np.zeros(shape = (len(models_list), 5))
    heatmap_time = np.zeros(shape = (len(models_list), 5))
    i = 0
    for model in models_list:
        df_base = pd.read_csv("./outputComparisons/Models/Base/"+model+".csv", index_col= False)
        df_base_synthetic = pd.read_csv("./outputComparisons/Models/BaseSynthetic/"+model+".csv", index_col= False)
        df_base_casing = pd.read_csv("./outputComparisons/Models/Casing/"+model+".csv", index_col= False)
        df_base_missingvalues = pd.read_csv("./outputComparisons/Models/MissingValues/"+model+".csv", index_col= False)
        df_base_suspectsign = pd.read_csv("./outputComparisons/Models/SuspectSign/"+model+".csv", index_col= False)
        #df_base_floatingpoint = pd.read_csv("./outputComparisons/Models/FloatingPointNumberAsString/"+model+".csv", index_col= False)
        new_row_accuracy = [np.mean(df_base["Accuracy"]), np.mean(df_base_synthetic["Accuracy"]), np.mean(df_base_casing["Accuracy"]), np.mean(df_base_missingvalues["Accuracy"]), np.mean(df_base_suspectsign["Accuracy"])]
        new_row_bal_accuracy = [np.mean(df_base["Balanced Accuracy"]), np.mean(df_base_synthetic["Balanced Accuracy"]), np.mean(df_base_casing["Balanced Accuracy"]), np.mean(df_base_missingvalues["Balanced Accuracy"]), np.mean(df_base_suspectsign["Balanced Accuracy"])]
        new_row_f1_score = [np.mean(df_base["F1 Score"]), np.mean(df_base_synthetic["F1 Score"]), np.mean(df_base_casing["F1 Score"]), np.mean(df_base_missingvalues["F1 Score"]), np.mean(df_base_suspectsign["F1 Score"])]
        new_row_time = [np.mean(df_base["Time Taken"]), np.mean(df_base_synthetic["Time Taken"]), np.mean(df_base_casing["Time Taken"]), np.mean(df_base_missingvalues["Time Taken"]), np.mean(df_base_suspectsign["Time Taken"])]
        heatmap_accuracy[i] = new_row_accuracy
        heatmap_bal_accuracy[i] = new_row_bal_accuracy
        heatmap_f1_score[i] = new_row_f1_score
        heatmap_time[i] = new_row_time
        i = i+1
    np.save(base_path_output+"accuracy.npy", heatmap_accuracy)
    np.save(base_path_output+"bal_accuracy.npy", heatmap_bal_accuracy)
    np.save(base_path_output+"f1_score.npy", heatmap_f1_score)
    np.save(base_path_output+"time.npy", heatmap_time)

def create_heatmap_models_df_m_1(df_list, base_path_output):
    df_longest = df_list[0].copy()
    for df in df_list: #Solo df_m_1 e df_l_3 hanno il modello "NuSVC"
        if len(df) > len(df_longest):
            df_longest = df.copy()
    models_list = df_longest["Model"].unique()
    print("Models list:")
    print(models_list)
    heatmap_accuracy = np.zeros(shape = (len(models_list), 7))
    heatmap_bal_accuracy = np.zeros(shape = (len(models_list), 7))
    heatmap_f1_score = np.zeros(shape = (len(models_list), 7))
    heatmap_time = np.zeros(shape = (len(models_list), 7))
    i = 0
    for model in models_list:
        df_base = pd.read_csv("./outputComparisons/Models/df_m_1/df_m_1/"+model+".csv", index_col= False)
        df_base_synthetic = pd.read_csv("./outputComparisons/Models/df_m_1/df_m_1_Synthetic/"+model+".csv", index_col= False)
        df_base_synthetic_FP = pd.read_csv("./outputComparisons/Models/df_m_1/df_m_1_Synthetic_FP/"+model+".csv", index_col= False)
        df_base_casing = pd.read_csv("./outputComparisons/Models/df_m_1/df_m_1_Casing/"+model+".csv", index_col= False)
        df_base_missingvalues = pd.read_csv("./outputComparisons/Models/df_m_1/df_m_1_MissingValues/"+model+".csv", index_col= False)
        df_base_floating = pd.read_csv("./outputComparisons/Models/df_m_1/df_m_1_FloatingPoint/"+model+".csv", index_col= False)
        df_base_suspectsign = pd.read_csv("./outputComparisons/Models/df_m_1/df_m_1_SuspectSign/"+model+".csv", index_col= False)
        #df_base_floatingpoint = pd.read_csv("./outputComparisons/Models/FloatingPointNumberAsString/"+model+".csv", index_col= False)
        new_row_accuracy = [np.mean(df_base["Accuracy"]), np.mean(df_base_synthetic["Accuracy"]), np.mean(df_base_synthetic_FP["Accuracy"]), np.mean(df_base_casing["Accuracy"]), np.mean(df_base_missingvalues["Accuracy"]), np.mean(df_base_floating["Accuracy"]), np.mean(df_base_suspectsign["Accuracy"])]
        new_row_bal_accuracy = [np.mean(df_base["Balanced Accuracy"]), np.mean(df_base_synthetic["Balanced Accuracy"]), np.mean(df_base_synthetic_FP["Balanced Accuracy"]), np.mean(df_base_casing["Balanced Accuracy"]), np.mean(df_base_missingvalues["Balanced Accuracy"]), np.mean(df_base_floating["Balanced Accuracy"]), np.mean(df_base_suspectsign["Balanced Accuracy"])]
        new_row_f1_score = [np.mean(df_base["F1 Score"]), np.mean(df_base_synthetic["F1 Score"]), np.mean(df_base_synthetic_FP["F1 Score"]), np.mean(df_base_casing["F1 Score"]), np.mean(df_base_missingvalues["F1 Score"]), np.mean(df_base_floating["F1 Score"]), np.mean(df_base_suspectsign["F1 Score"])]
        new_row_time = [np.mean(df_base["Time Taken"]), np.mean(df_base_synthetic["Time Taken"]), np.mean(df_base_synthetic_FP["Time Taken"]), np.mean(df_base_casing["Time Taken"]), np.mean(df_base_missingvalues["Time Taken"]), np.mean(df_base_floating["Time Taken"]), np.mean(df_base_suspectsign["Time Taken"])]
        heatmap_accuracy[i] = new_row_accuracy
        heatmap_bal_accuracy[i] = new_row_bal_accuracy
        heatmap_f1_score[i] = new_row_f1_score
        heatmap_time[i] = new_row_time
        i = i+1
    np.save(base_path_output+"accuracy.npy", heatmap_accuracy)
    np.save(base_path_output+"bal_accuracy.npy", heatmap_bal_accuracy)
    np.save(base_path_output+"f1_score.npy", heatmap_f1_score)
    np.save(base_path_output+"time.npy", heatmap_time)


def calc_means(df_list, path_output):
    df_output = pd.DataFrame(columns=df_list[0].columns)
    df_longest = df_list[0].copy()
    for df in df_list: #Solo df_m_1 e df_l_3 hanno il modello "NuSVC"
        if len(df) > len(df_longest):
            df_longest = df.copy()
    
    for _, row in df_longest.iterrows():
        accuracies = []
        balanced_accuracies = []
        roc_auc = []
        f1_scores = []
        times_taken = []
        for df in df_list:
            if row["Model"] in df["Model"].unique():
                row_compare = df.loc[df["Model"] == row["Model"]]
                accuracies.append(float(row_compare["Accuracy"]))
                balanced_accuracies.append(float(row_compare["Balanced Accuracy"]))
                roc_auc.append(float(row_compare["ROC AUC"]))
                f1_scores.append(float(row_compare["F1 Score"]))
                times_taken.append(float(row_compare["Time Taken"]))
        
        df_output.loc[len(df_output)] = [row["Model"], np.mean(accuracies), np.mean(balanced_accuracies), np.mean(roc_auc), np.mean(f1_scores), np.mean(times_taken)]
                

    df_output.to_csv(path_output, index=False)


if __name__ == "__main__":
    if(not Path("./outputComparisons").exists()):
        os.mkdir("./outputComparisons")

    if(not Path("./outputComparisons/Combined").exists()):
        os.mkdir("./outputComparisons/Combined")
        
    if(not Path("./outputComparisons/Combined/General").exists()):
        os.mkdir("./outputComparisons/Combined/General")
        
    if(not Path("./outputComparisons/MissingValues").exists()):
        os.mkdir("./outputComparisons/MissingValues")

    if(not Path("./outputComparisons/Casing").exists()):
        os.mkdir("./outputComparisons/Casing")

    if(not Path("./outputComparisons/ExtremeValues").exists()):
        os.mkdir("./outputComparisons/ExtremeValues")

    if(not Path("./outputComparisons/SuspectSign").exists()):
        os.mkdir("./outputComparisons/SuspectSign")

    if(not Path("./outputComparisons/FloatingPointNumberAsString").exists()):
        os.mkdir("./outputComparisons/FloatingPointNumberAsString")
        
    if(not Path("./outputComparisons/Means").exists()):
        os.mkdir("./outputComparisons/Means")
        
    if(not Path("./outputComparisons/Means/General").exists()):
        os.mkdir("./outputComparisons/Means/General")

    if(not Path("./outputComparisons/Models").exists()):
        os.mkdir("./outputComparisons/Models")
        
    if(not Path("./outputComparisons/Models/Base").exists()):
        os.mkdir("./outputComparisons/Models/Base")
        
    if(not Path("./outputComparisons/Models/BaseNumeric").exists()):
        os.mkdir("./outputComparisons/Models/BaseNumeric")

    if(not Path("./outputComparisons/Models/BaseCasing").exists()):
        os.mkdir("./outputComparisons/Models/BaseCasing")
        
    if(not Path("./outputComparisons/Models/BaseSynthetic").exists()):
        os.mkdir("./outputComparisons/Models/BaseSynthetic")

    if(not Path("./outputComparisons/Models/BaseSyntheticNumeric").exists()):
        os.mkdir("./outputComparisons/Models/BaseSyntheticNumeric")
        
    if(not Path("./outputComparisons/Models/BaseSyntheticCasing").exists()):
        os.mkdir("./outputComparisons/Models/BaseSyntheticCasing")
    
    if(not Path("./outputComparisons/Models/MissingValues").exists()):
        os.mkdir("./outputComparisons/Models/MissingValues")
        
    if(not Path("./outputComparisons/Models/Casing").exists()):
        os.mkdir("./outputComparisons/Models/Casing")
        
    if(not Path("./outputComparisons/Models/ExtremeValues").exists()):
        os.mkdir("./outputComparisons/Models/ExtremeValues")
        
    if(not Path("./outputComparisons/Models/SuspectSign").exists()):
        os.mkdir("./outputComparisons/Models/SuspectSign")
        
    if(not Path("./outputComparisons/Models/FloatingPointNumberAsString").exists()):
        os.mkdir("./outputComparisons/Models/FloatingPointNumberAsString")

    if(not Path("./outputComparisons/Models/Heatmaps").exists()):
        os.mkdir("./outputComparisons/Models/Heatmaps")
    
    if(not Path("./outputComparisons/Models/HeatmapsNumeric").exists()):
        os.mkdir("./outputComparisons/Models/HeatmapsNumeric")

    if(not Path("./outputComparisons/Models/HeatmapsCasing").exists()):
        os.mkdir("./outputComparisons/Models/HeatmapsCasing")
        
    if(not Path("./outputComparisons/Models/Heatmaps_df_m_1").exists()):
        os.mkdir("./outputComparisons/Models/Heatmaps_df_m_1")
        
    if(not Path("./outputComparisons/Models/df_m_1").exists()):
        os.mkdir("./outputComparisons/Models/df_m_1")
        
    if(not Path("./outputComparisons/Models/df_m_1/df_m_1").exists()):
        os.mkdir("./outputComparisons/Models/df_m_1/df_m_1")
        
    if(not Path("./outputComparisons/Models/df_m_1/df_m_1_Synthetic").exists()):
        os.mkdir("./outputComparisons/Models/df_m_1/df_m_1_Synthetic")
        
    if(not Path("./outputComparisons/Models/df_m_1/df_m_1_Synthetic_FP").exists()):
        os.mkdir("./outputComparisons/Models/df_m_1/df_m_1_Synthetic_FP")
        
    if(not Path("./outputComparisons/Models/df_m_1/df_m_1_Casing").exists()):
        os.mkdir("./outputComparisons/Models/df_m_1/df_m_1_Casing")
        
    if(not Path("./outputComparisons/Models/df_m_1/df_m_1_MissingValues").exists()):
        os.mkdir("./outputComparisons/Models/df_m_1/df_m_1_MissingValues")
        
    if(not Path("./outputComparisons/Models/df_m_1/df_m_1_FloatingPoint").exists()):
        os.mkdir("./outputComparisons/Models/df_m_1/df_m_1_FloatingPoint")
        
    if(not Path("./outputComparisons/Models/df_m_1/df_m_1_SuspectSign").exists()):
        os.mkdir("./outputComparisons/Models/df_m_1/df_m_1_SuspectSign")
        


    df_l_1 = pd.read_csv("./outputLazyPredict/Base/df_l_1/df_l_1.csv")
    df_l_2 = pd.read_csv("./outputLazyPredict/Base/df_l_2/df_l_2.csv")
    df_l_3 = pd.read_csv("./outputLazyPredict/Base/df_l_3/df_l_3.csv")
    df_m_1 = pd.read_csv("./outputLazyPredict/Base/df_m_1/df_m_1.csv")
    df_m_2 = pd.read_csv("./outputLazyPredict/Base/df_m_2/df_m_2.csv")

    
    df_l_1_Synthetic = pd.read_csv("./outputLazyPredict/BaseSynthetic/df_l_1.csv")
    df_l_2_Synthetic = pd.read_csv("./outputLazyPredict/BaseSynthetic/df_l_2.csv")
    df_l_3_Synthetic = pd.read_csv("./outputLazyPredict/BaseSynthetic/df_l_3.csv")
    df_m_1_Synthetic = pd.read_csv("./outputLazyPredict/BaseSynthetic/df_m_1.csv")
    df_m_1_Synthetic_FP = pd.read_csv("./outputLazyPredict/BaseSynthetic/df_m_1_FP.csv")
    df_m_2_Synthetic = pd.read_csv("./outputLazyPredict/BaseSynthetic/df_m_2.csv")

    results_Base = [df_l_1, df_l_2, df_l_3, df_m_1, df_m_2]
    
    results_Base_numeric = [df_l_1, df_l_2, df_m_1, df_m_2]
    results_Base_casing = [df_l_1, df_l_2, df_l_3, df_m_1]

    results_Base_numeric_large = [df_l_1, df_l_2]
    results_Base_casing_large = [df_l_1, df_l_2, df_l_3]

    results_Base_numeric_medium = [df_m_1, df_m_2]
    #results_Base_casing_medium = [df_m_1]

    results_BaseSynthetic = [df_l_1_Synthetic, df_l_2_Synthetic, df_l_3_Synthetic, df_m_1_Synthetic, df_m_2_Synthetic]
    results_BaseSynthetic_numeric = [df_l_1_Synthetic, df_l_2_Synthetic, df_m_1_Synthetic, df_m_2_Synthetic]
    results_BaseSynthetic_casing = [df_l_1_Synthetic, df_l_2_Synthetic, df_l_3_Synthetic, df_m_1_Synthetic]

    results_BaseSynthetic_numeric_large = [df_l_1_Synthetic, df_l_2_Synthetic]
    results_BaseSynthetic_casing_large = [df_l_1_Synthetic, df_l_2_Synthetic, df_l_3_Synthetic]

    results_BaseSynthetic_numeric_medium = [df_m_1_Synthetic, df_m_2_Synthetic]

    df_l_1_1_Casing = pd.read_csv("./outputLazyPredict/Casing/df_l_1/df_l_1_1.csv")
    df_l_2_1_Casing = pd.read_csv("./outputLazyPredict/Casing/df_l_2/df_l_2_1.csv")
    df_l_3_1_Casing = pd.read_csv("./outputLazyPredict/Casing/df_l_3/df_l_3_1.csv")
    df_m_1_1_Casing = pd.read_csv("./outputLazyPredict/Casing/df_m_1/df_m_1_1.csv")

    '''
    df_l_1_1_ExtremeValues = pd.read_csv("./outputLazyPredict/ExtremeValues/df_l_1/df_l_1_1.csv")
    df_l_2_1_ExtremeValues = pd.read_csv("./outputLazyPredict/ExtremeValues/df_l_2/df_l_2_1.csv")
    df_m_1_1_ExtremeValues = pd.read_csv("./outputLazyPredict/ExtremeValues/df_m_1/df_m_1_1.csv")
    df_m_2_1_ExtremeValues = pd.read_csv("./outputLazyPredict/ExtremeValues/df_m_2/df_m_2_1.csv")
    '''

    df_l_1_1_MissingValues = pd.read_csv("./outputLazyPredict/MissingValues/df_l_1/df_l_1_1.csv")
    df_l_2_1_MissingValues = pd.read_csv("./outputLazyPredict/MissingValues/df_l_2/df_l_2_1.csv")
    df_m_1_1_MissingValues = pd.read_csv("./outputLazyPredict/MissingValues/df_m_1/df_m_1_1.csv")
    df_m_2_1_MissingValues = pd.read_csv("./outputLazyPredict/MissingValues/df_m_2/df_m_2_1.csv")

    df_l_1_1_SuspectSign = pd.read_csv("./outputLazyPredict/SuspectSign/df_l_1/df_l_1_1.csv")
    df_l_2_1_SuspectSign = pd.read_csv("./outputLazyPredict/SuspectSign/df_l_2/df_l_2_1.csv")
    df_m_1_1_SuspectSign = pd.read_csv("./outputLazyPredict/SuspectSign/df_m_1/df_m_1_1.csv")
    df_m_2_1_SuspectSign = pd.read_csv("./outputLazyPredict/SuspectSign/df_m_2/df_m_2_1.csv")

    df_m_1_1_FloatingPoint = pd.read_csv("./outputLazyPredict/FloatingPointNumberAsString/df_m_1/df_m_1_1.csv")

    
    compare_results(df_l_1, df_l_1_1_Casing, "./outputComparisons/Casing/df_l_1_1.csv")
    compare_results(df_l_2, df_l_2_1_Casing, "./outputComparisons/Casing/df_l_2_1.csv")
    compare_results(df_l_3, df_l_3_1_Casing, "./outputComparisons/Casing/df_l_3_1.csv")
    compare_results(df_m_1, df_m_1_1_Casing, "./outputComparisons/Casing/df_m_1_1.csv")
    
    compare_results(df_l_1, df_l_1_1_MissingValues, "./outputComparisons/MissingValues/df_l_1_1.csv")
    compare_results(df_l_2, df_l_2_1_MissingValues, "./outputComparisons/MissingValues/df_l_2_1.csv")
    compare_results(df_m_1, df_m_1_1_MissingValues, "./outputComparisons/MissingValues/df_m_1_1.csv")
    compare_results(df_m_2, df_m_2_1_MissingValues, "./outputComparisons/MissingValues/df_m_2_1.csv")
    
    compare_results(df_l_1, df_l_1_1_SuspectSign, "./outputComparisons/SuspectSign/df_l_1_1.csv")
    compare_results(df_l_2, df_l_2_1_SuspectSign, "./outputComparisons/SuspectSign/df_l_2_1.csv")
    compare_results(df_m_1, df_m_1_1_SuspectSign, "./outputComparisons/SuspectSign/df_m_1_1.csv")
    compare_results(df_m_2, df_m_2_1_SuspectSign, "./outputComparisons/SuspectSign/df_m_2_1.csv")

    compare_results(df_m_1, df_m_1_1_FloatingPoint, "./outputComparisons/FloatingPointNumberAsString/df_m_1_1.csv")


    
    results_df_l_1_1_Casing = pd.read_csv("./outputComparisons/Casing/df_l_1_1.csv")
    results_df_l_2_1_Casing = pd.read_csv("./outputComparisons/Casing/df_l_2_1.csv")
    results_df_l_3_1_Casing = pd.read_csv("./outputComparisons/Casing/df_l_3_1.csv")
    results_df_m_1_1_Casing = pd.read_csv("./outputComparisons/Casing/df_m_1_1.csv")
    results_Casing_diff = [results_df_l_1_1_Casing, results_df_l_2_1_Casing, results_df_l_3_1_Casing, results_df_m_1_1_Casing]

    results_df_l_1_1_MissingValues = pd.read_csv("./outputComparisons/MissingValues/df_l_1_1.csv")
    results_df_l_2_1_MissingValues = pd.read_csv("./outputComparisons/MissingValues/df_l_2_1.csv")
    results_df_m_1_1_MissingValues = pd.read_csv("./outputComparisons/MissingValues/df_m_1_1.csv")
    results_df_m_2_1_MissingValues = pd.read_csv("./outputComparisons/MissingValues/df_m_2_1.csv")
    results_MissingValues_diff = [results_df_l_1_1_MissingValues, results_df_l_2_1_MissingValues, results_df_m_1_1_MissingValues, results_df_m_2_1_MissingValues]

    results_df_l_1_1_SuspectSign = pd.read_csv("./outputComparisons/SuspectSign/df_l_1_1.csv")
    results_df_l_2_1_SuspectSign = pd.read_csv("./outputComparisons/SuspectSign/df_l_2_1.csv")
    results_df_m_1_1_SuspectSign = pd.read_csv("./outputComparisons/SuspectSign/df_m_1_1.csv")
    results_df_m_2_1_SuspectSign = pd.read_csv("./outputComparisons/SuspectSign/df_m_2_1.csv")
    results_SuspectSign_diff = [results_df_l_1_1_SuspectSign, results_df_l_2_1_SuspectSign, results_df_m_1_1_SuspectSign, results_df_m_2_1_SuspectSign]

    df_m_1_1_FloatingPoint_diff = pd.read_csv("./outputLazyPredict/FloatingPointNumberAsString/df_m_1/df_m_1_1.csv")

    results_Casing = [df_l_1_1_Casing, df_l_2_1_Casing, df_l_3_1_Casing, df_m_1_1_Casing]
    results_Casing_large = [df_l_1_1_Casing, df_l_2_1_Casing, df_l_3_1_Casing]
    results_MissingValues = [df_l_1_1_MissingValues, df_l_2_1_MissingValues, df_m_1_1_MissingValues, df_m_2_1_MissingValues]
    results_MissingValues_large = [df_l_1_1_MissingValues, df_l_2_1_MissingValues]
    results_MissingValues_medium = [df_m_1_1_MissingValues, df_m_2_1_MissingValues]
    results_SuspectSign = [df_l_1_1_SuspectSign, df_l_2_1_SuspectSign, df_m_1_1_SuspectSign, df_m_2_1_SuspectSign]
    results_SuspectSign_large = [df_l_1_1_SuspectSign, df_l_2_1_SuspectSign]
    results_SuspectSign_medium = [df_m_1_1_SuspectSign, df_m_2_1_SuspectSign]
    calc_means(results_Base, "./outputComparisons/Means/General/base.csv")
    calc_means(results_BaseSynthetic, "./outputComparisons/Means/General/base_synthetic.csv")
    calc_means(results_Base_numeric, "./outputComparisons/Means/General/base_numeric.csv")
    calc_means(results_BaseSynthetic_numeric, "./outputComparisons/Means/General/base_synthetic_numeric.csv")
    calc_means(results_Base_casing, "./outputComparisons/Means/General/base_casing.csv")
    calc_means(results_BaseSynthetic_casing, "./outputComparisons/Means/General/base_synthetic_casing.csv")
    calc_means(results_Casing_diff, "./outputComparisons/Means/General/casing_diff.csv")
    calc_means(results_MissingValues_diff, "./outputComparisons/Means/General/missing_values_diff.csv")
    calc_means(results_SuspectSign_diff, "./outputComparisons/Means/General/suspect_sign_diff.csv")
    calc_means(results_Casing, "./outputComparisons/Means/General/casing.csv")
    calc_means(results_MissingValues, "./outputComparisons/Means/General/missing_values.csv")
    calc_means(results_SuspectSign, "./outputComparisons/Means/General/suspect_sign.csv")

    calc_means(results_Base_numeric_large, "./outputComparisons/Means/General/base_numeric_large.csv")
    calc_means(results_BaseSynthetic_numeric_large, "./outputComparisons/Means/General/base_synthetic_numeric_large.csv")
    calc_means(results_MissingValues_large, "./outputComparisons/Means/General/missing_values_large.csv")
    calc_means(results_SuspectSign_large, "./outputComparisons/Means/General/suspect_sign_large.csv")
    calc_means(results_Base_casing_large, "./outputComparisons/Means/General/base_casing_large.csv")
    calc_means(results_BaseSynthetic_casing_large, "./outputComparisons/Means/General/base_synthetic_casing_large.csv")
    calc_means(results_Casing_large, "./outputComparisons/Means/General/casing_large.csv")

    calc_means(results_Base_numeric_medium, "./outputComparisons/Means/General/base_numeric_medium.csv")
    calc_means(results_BaseSynthetic_numeric_medium, "./outputComparisons/Means/General/base_synthetic_numeric_medium.csv")
    calc_means(results_MissingValues_medium, "./outputComparisons/Means/General/missing_values_medium.csv")
    calc_means(results_SuspectSign_medium, "./outputComparisons/Means/General/suspect_sign_medium.csv")

    
    split_models(results_Base, "./outputComparisons/Models/Base")
    split_models(results_BaseSynthetic, "./outputComparisons/Models/BaseSynthetic")
    split_models(results_Base_numeric, "./outputComparisons/Models/BaseNumeric")
    split_models(results_BaseSynthetic_numeric, "./outputComparisons/Models/BaseSyntheticNumeric")
    split_models(results_Base_casing, "./outputComparisons/Models/BaseCasing")
    split_models(results_BaseSynthetic_casing, "./outputComparisons/Models/BaseSyntheticCasing")
    split_models(results_Casing, "./outputComparisons/Models/Casing")
    split_models(results_MissingValues, "./outputComparisons/Models/MissingValues")
    split_models(results_SuspectSign, "./outputComparisons/Models/SuspectSign")
    split_models([df_m_1], "./outputComparisons/Models/df_m_1/df_m_1")
    split_models([df_m_1_Synthetic], "./outputComparisons/Models/df_m_1/df_m_1_Synthetic")
    split_models([df_m_1_Synthetic_FP], "./outputComparisons/Models/df_m_1/df_m_1_Synthetic_FP")
    split_models([df_m_1_1_Casing], "./outputComparisons/Models/df_m_1/df_m_1_Casing")
    split_models([df_m_1_1_MissingValues], "./outputComparisons/Models/df_m_1/df_m_1_MissingValues")
    split_models([df_m_1_1_FloatingPoint], "./outputComparisons/Models/df_m_1/df_m_1_FloatingPoint")
    split_models([df_m_1_1_SuspectSign], "./outputComparisons/Models/df_m_1/df_m_1_SuspectSign")

    create_heatmap_models(results_Base, "./outputComparisons/Models/Heatmaps/")
    create_heatmap_models_numeric(results_Base, "./outputComparisons/Models/HeatmapsNumeric/")
    create_heatmap_models_casing(results_Base, "./outputComparisons/Models/HeatmapsCasing/")
    create_heatmap_models_df_m_1(results_Base, "./outputComparisons/Models/Heatmaps_df_m_1/")

    df_base_combined = pd.concat(results_Base)
    df_base_combined.to_csv("./outputComparisons/Combined/General/base.csv", index=False)
    df_base_synthetic_combined = pd.concat(results_BaseSynthetic)
    df_base_synthetic_combined.to_csv("./outputComparisons/Combined/General/base_synthetic.csv", index=False)
    df_base_numeric_combined = pd.concat(results_Base_numeric)
    df_base_numeric_combined.to_csv("./outputComparisons/Combined/General/base_numeric.csv", index=False)
    df_base_synthetic_numeric_combined = pd.concat(results_BaseSynthetic_numeric)
    df_base_synthetic_numeric_combined.to_csv("./outputComparisons/Combined/General/base_synthetic_numeric.csv", index=False)
    df_base_casing_combined = pd.concat(results_Base_casing)
    df_base_casing_combined.to_csv("./outputComparisons/Combined/General/base_casing.csv", index=False)
    df_base_synthetic_casing_combined = pd.concat(results_BaseSynthetic_casing)
    df_base_synthetic_casing_combined.to_csv("./outputComparisons/Combined/General/base_synthetic_casing.csv", index=False)
    df_casing_combined = pd.concat(results_Casing)
    df_casing_combined.to_csv("./outputComparisons/Combined/General/casing.csv", index=False)
    df_missingvalues_combined = pd.concat(results_MissingValues)
    df_missingvalues_combined.to_csv("./outputComparisons/Combined/General/missing_values.csv", index=False)
    df_suspectsign_combined = pd.concat(results_SuspectSign)
    df_suspectsign_combined.to_csv("./outputComparisons/Combined/General/suspect_sign.csv", index=False)

    df_base_numeric_large_combined = pd.concat(results_Base_numeric_large)
    df_base_numeric_large_combined.to_csv("./outputComparisons/Combined/General/base_numeric_large.csv", index=False)
    df_base_synthetic_numeric_large_combined = pd.concat(results_BaseSynthetic_numeric_large)
    df_base_synthetic_numeric_large_combined.to_csv("./outputComparisons/Combined/General/base_synthetic_numeric_large.csv", index=False)
 
    df_missingvalues_large_combined = pd.concat(results_MissingValues_large)
    df_missingvalues_large_combined.to_csv("./outputComparisons/Combined/General/missing_values_large.csv", index=False)
 
    df_suspectsign_large_combined = pd.concat(results_SuspectSign_large)
    df_suspectsign_large_combined.to_csv("./outputComparisons/Combined/General/suspect_sign_large.csv", index=False)

    df_base_casing_large_combined = pd.concat(results_Base_casing_large)
    df_base_casing_large_combined.to_csv("./outputComparisons/Combined/General/base_casing_large.csv", index=False)
 
    df_base_synthetic_casing_large_combined = pd.concat(results_BaseSynthetic_casing_large)
    df_base_synthetic_casing_large_combined.to_csv("./outputComparisons/Combined/General/base_synthetic_casing_large.csv", index=False)
  
    df_casing_large_combined = pd.concat(results_Casing_large)
    df_casing_large_combined.to_csv("./outputComparisons/Combined/General/casing_large.csv", index=False)
  

    df_base_numeric_medium_combined = pd.concat(results_Base_numeric_medium)
    df_base_numeric_medium_combined.to_csv("./outputComparisons/Combined/General/base_numeric_medium.csv", index=False)
  
    df_base_synthetic_numeric_medium_combined = pd.concat(results_BaseSynthetic_numeric_medium)
    df_base_synthetic_numeric_medium_combined.to_csv("./outputComparisons/Combined/General/base_synthetic_numeric_medium.csv", index=False)
 
    df_missingvalues_medium_combined = pd.concat(results_MissingValues_medium)
    df_missingvalues_medium_combined.to_csv("./outputComparisons/Combined/General/missing_values_medium.csv", index=False)

    df_suspectsign_medium_combined = pd.concat(results_SuspectSign_medium)
    df_suspectsign_medium_combined.to_csv("./outputComparisons/Combined/General/suspect_sign_medium.csv", index=False)
