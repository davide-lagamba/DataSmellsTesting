from pathlib import Path
from time import sleep

import requests
import pandas as pd
from bs4 import BeautifulSoup
import os
import numpy as np
def combine_dataframe(df, source, name):
    rootdir = '.\outputDSInjection'
    newdir = ".\combined_df"
    if(not Path(newdir).exists()):
        os.mkdir(newdir)
    if (not Path(newdir+"/Casing").exists()):
        os.mkdir(newdir+"/Casing")
    if (not Path(newdir+"/ExtremeValues").exists()):
        os.mkdir(newdir+"/ExtremeValues")
    if (not Path(newdir+"/MissingValues").exists()):
        os.mkdir(newdir+"/MissingValues")
    if (not Path(newdir+"/FloatingPointNumberAsString").exists()):
        os.mkdir(newdir+"/FloatingPointNumberAsString")
    if (not Path(newdir+"/SuspectSign").exists()):
        os.mkdir(newdir+"/SuspectSign")
    casing = "Casing"
    extreme = "ExtremeValues"
    missing = "MissingValues"
    floating = "FloatingPointNumberAsString"
    suspect_sign = "SuspectSign"
    df_base = pd.read_csv("./"+source+"/"+name+".csv", index_col=False)
    #df_base = df_base.head(2)
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            path = str(os.path.join(subdir, file))
            if df in path:
                print(path)
                df_smells = pd.read_csv(path, index_col=False)
                df_new = pd.concat([df_base, df_smells])
                if casing in subdir:
                    if (not Path(newdir+"/Casing/"+name).exists()):
                        os.mkdir(newdir+"/Casing/"+name)
                    df_new.to_csv(newdir+"/Casing/"+name+"/"+file, index=False)
                if missing in subdir:
                    if (not Path(newdir+"/MissingValues/"+name).exists()):
                        os.mkdir(newdir+"/MissingValues/"+name)
                    df_new.to_csv(newdir+"/MissingValues/"+name+"/"+file, index=False)
                if extreme in subdir:
                    if (not Path(newdir+"/ExtremeValues/"+name).exists()):
                        os.mkdir(newdir+"/ExtremeValues/"+name)
                    df_new.to_csv(newdir+"/ExtremeValues/"+name+"/"+file, index=False)
                    from sklearn.preprocessing import StandardScaler
                    zscore = StandardScaler()
                    out = zscore.fit_transform(df_new.select_dtypes(include='number'))
                    print(out[len(out)-1])
                if floating in subdir:
                    if (not Path(newdir+"/FloatingPointNumberAsString/"+name).exists()):
                        os.mkdir(newdir+"/FloatingPointNumberAsString/"+name)
                    df_new.to_csv(newdir+"/FloatingPointNumberAsString/"+name+"/"+file, index=False)
                if suspect_sign in subdir:
                    if (not Path(newdir+"/SuspectSign/"+name).exists()):
                        os.mkdir(newdir+"/SuspectSign/"+name)
                    df_new.to_csv(newdir+"/SuspectSign/"+name+"/"+file, index=False)


if __name__ == "__main__":
    
    combine_dataframe("df_l_1", "cleaned_datasets", "df_l_1_1")
    combine_dataframe("df_l_1", "cleaned_datasets", "df_l_1_2")
    combine_dataframe("df_l_1", "cleaned_datasets", "df_l_1")
    
    combine_dataframe("df_l_2", "cleaned_datasets", "df_l_2_1")
    combine_dataframe("df_l_2", "cleaned_datasets", "df_l_2_2")
    combine_dataframe("df_l_2", "cleaned_datasets", "df_l_2_3")
    combine_dataframe("df_l_2", "cleaned_datasets", "df_l_2")
    
    combine_dataframe("df_l_3", "cleaned_datasets", "df_l_3")
    combine_dataframe("df_m_1", "cleaned_datasets", "df_m_1")
    combine_dataframe("df_m_2", "cleaned_datasets", "df_m_2")

    
    '''
    #Analisi per Suspect Sign smell
    print(np.quantile((pd.read_csv("./combined_df/SuspectSign/df_l_2_1/df_l_2_1.csv", index_col=False)["discharge_disposition_id"]), 0.25))
    print(np.quantile((pd.read_csv("./combined_df/SuspectSign/df_l_2_2/df_l_2_1.csv", index_col=False)["discharge_disposition_id"]), 0.25))
    print(np.quantile((pd.read_csv("./combined_df/SuspectSign/df_l_2_3/df_l_2_1.csv", index_col=False)["discharge_disposition_id"]), 0.25))
    print(np.quantile((pd.read_csv("./combined_df/SuspectSign/df_l_2/df_l_2_1.csv", index_col=False)["discharge_disposition_id"]), 0.25))
    
    print(((pd.read_csv("./combined_df/SuspectSign/df_l_2_1/df_l_2_1.csv", index_col=False)["discharge_disposition_id"])).describe())
    print(((pd.read_csv("./combined_df/SuspectSign/df_l_2_2/df_l_2_1.csv", index_col=False)["discharge_disposition_id"])).describe())
    print(((pd.read_csv("./combined_df/SuspectSign/df_l_2_3/df_l_2_1.csv", index_col=False)["discharge_disposition_id"])).describe())
    print(np.quantile((pd.read_csv("./combined_df/SuspectSign/df_l_2/df_l_2_1.csv", index_col=False)["discharge_disposition_id"]), 0.25))
    print(np.quantile((pd.read_csv("./combined_df/SuspectSign/df_l_1/df_l_1_1.csv", index_col=False)["age"]), 0.25))
    print(np.quantile((pd.read_csv("./combined_df/SuspectSign/df_m_1/df_m_1_1.csv", index_col=False)["combine"]), 0.25))
    print(np.quantile((pd.read_csv("./combined_df/SuspectSign/df_m_2/df_m_2_1.csv", index_col=False)["Course"]), 0.25))
    '''
    
    if(not Path("./combined_df/Base").exists()):
        os.mkdir("./combined_df/Base")
    
    df_l_1_1 = pd.read_csv("./cleaned_datasets/df_l_1_1.csv", index_col=False)
    df_l_1_2 = pd.read_csv("./cleaned_datasets/df_l_1_2.csv", index_col=False)
    df_l_2_1 = pd.read_csv("./cleaned_datasets/df_l_2_1.csv", index_col=False)
    df_l_2_2 = pd.read_csv("./cleaned_datasets/df_l_2_2.csv", index_col=False)
    df_l_2_3 = pd.read_csv("./cleaned_datasets/df_l_2_3.csv", index_col=False)
    df_l_3 = pd.read_csv("./cleaned_datasets/df_l_3.csv", index_col=False)
    df_m_1 = pd.read_csv("./cleaned_datasets/df_m_1.csv", index_col=False)
    df_m_2 = pd.read_csv("./cleaned_datasets/df_m_2.csv", index_col=False)

    df_l_1_SDV = pd.read_csv("./outputSDV/df_l_1_SDV.csv", index_col=False)
    df_l_1_99_SDV = pd.read_csv("./outputSDV/df_l_1_99_SDV.csv", index_col=False)
    
    df_l_2_SDV = pd.read_csv("./outputSDV/df_l_2_SDV.csv", index_col=False)
    df_l_2_99_SDV = pd.read_csv("./outputSDV/df_l_2_99_SDV.csv", index_col=False)
    
    df_l_3_SDV = pd.read_csv("./outputSDV/df_l_3_SDV.csv", index_col=False)
    df_l_3_99_SDV = pd.read_csv("./outputSDV/df_l_3_99_SDV.csv", index_col=False)
    
    df_m_1_SDV = pd.read_csv("./outputSDV/df_m_1_SDV.csv", index_col=False)
    df_m_1_99_SDV = pd.read_csv("./outputSDV/df_m_1_99_SDV.csv", index_col=False)
    df_m_1_FP_SDV = pd.read_csv("./outputSDV/df_m_1_FP_SDV.csv", index_col=False)
    
    df_m_2_SDV = pd.read_csv("./outputSDV/df_m_2_SDV.csv", index_col=False)
    df_m_2_99_SDV = pd.read_csv("./outputSDV/df_m_2_99_SDV.csv", index_col=False)

    df_l_1_1_combined_SDV = pd.concat([df_l_1_1, df_l_1_SDV])
    df_l_1_1_99_combined_SDV = pd.concat([df_l_1_1, df_l_1_99_SDV])
    df_l_1_2_combined_SDV = pd.concat([df_l_1_2, df_l_1_SDV])
    df_l_1_2_99_combined_SDV = pd.concat([df_l_1_2, df_l_1_99_SDV])
    df_l_1_1_combined_SDV.to_csv("./combined_df/Base/df_l_1_1.csv", index = False)
    df_l_1_1_99_combined_SDV.to_csv("./combined_df/Base/df_l_1_1_99.csv", index = False)
    df_l_1_2_combined_SDV.to_csv("./combined_df/Base/df_l_1_2.csv", index = False)
    df_l_1_2_99_combined_SDV.to_csv("./combined_df/Base/df_l_1_2_99.csv", index = False)

    
    df_l_2_1_combined_SDV = pd.concat([df_l_2_1, df_l_2_SDV])
    df_l_2_1_99_combined_SDV = pd.concat([df_l_2_1, df_l_2_99_SDV])
    df_l_2_2_combined_SDV = pd.concat([df_l_2_2, df_l_2_SDV])
    df_l_2_2_99_combined_SDV = pd.concat([df_l_2_2, df_l_2_99_SDV])
    df_l_2_3_combined_SDV = pd.concat([df_l_2_3, df_l_2_SDV])
    df_l_2_3_99_combined_SDV = pd.concat([df_l_2_3, df_l_2_99_SDV])
    df_l_2_1_combined_SDV.to_csv("./combined_df/Base/df_l_2_1.csv", index = False)
    df_l_2_1_99_combined_SDV.to_csv("./combined_df/Base/df_l_2_1_99.csv", index = False)
    df_l_2_2_combined_SDV.to_csv("./combined_df/Base/df_l_2_2.csv", index = False)
    df_l_2_2_99_combined_SDV.to_csv("./combined_df/Base/df_l_2_2_99.csv", index = False)
    df_l_2_3_combined_SDV.to_csv("./combined_df/Base/df_l_2_3.csv", index = False)
    df_l_2_3_99_combined_SDV.to_csv("./combined_df/Base/df_l_2_3_99.csv", index = False)

    
    df_l_3_combined_SDV = pd.concat([df_l_3, df_l_3_SDV])
    df_l_3_99_combined_SDV = pd.concat([df_l_3, df_l_3_99_SDV])
    df_l_3_combined_SDV.to_csv("./combined_df/Base/df_l_3.csv", index = False)
    df_l_3_99_combined_SDV.to_csv("./combined_df/Base/df_l_3_99.csv", index = False)

    
    df_m_1_combined_SDV = pd.concat([df_m_1, df_m_1_SDV])
    df_m_1_99_combined_SDV = pd.concat([df_m_1, df_m_1_99_SDV])
    df_m_1_FP_combined_SDV = pd.concat([df_m_1, df_m_1_FP_SDV])
    df_m_1_combined_SDV.to_csv("./combined_df/Base/df_m_1.csv", index = False)
    df_m_1_99_combined_SDV.to_csv("./combined_df/Base/df_m_1_99.csv", index = False)
    df_m_1_FP_combined_SDV.to_csv("./combined_df/Base/df_m_1_FP.csv", index = False)

    
    df_m_2_combined_SDV = pd.concat([df_m_2, df_m_2_SDV])
    df_m_2_99_combined_SDV = pd.concat([df_m_2, df_m_2_99_SDV])
    df_m_2_combined_SDV.to_csv("./combined_df/Base/df_m_2.csv", index = False)
    df_m_2_99_combined_SDV.to_csv("./combined_df/Base/df_m_2_99.csv", index = False)

    combine_dataframe("df_l_1", "pre-processed", "kdd-census_1")
    combine_dataframe("df_l_1", "pre-processed", "kdd-census_2")
    combine_dataframe("df_l_2", "pre-processed", "diabetic_data_1")
    combine_dataframe("df_l_2", "pre-processed", "diabetic_data_2")
    combine_dataframe("df_l_2", "pre-processed", "diabetic_data_3")
    combine_dataframe("df_l_3", "pre-processed", "nursery")
    combine_dataframe("df_m_1", "pre-processed", "Firefighter_Promotion_Exam_Scores")
    combine_dataframe("df_m_2", "pre-processed", "tae")