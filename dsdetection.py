#Adattato dal file "dataSmellDetection.py" del replication package del lavoro di Recupito et al. 2024: https://ieeexplore.ieee.org/document/10556050
#https://stackoverflow.com/questions/53263854/cant-pass-csrf-token-into-form-when-using-python-requests
from pathlib import Path
from time import sleep

import requests
import pandas as pd
from bs4 import BeautifulSoup
import os

def smell_detection(path, name):
    df = pd.read_csv(path)
    url_upload = 'http://localhost:5005'
    files = {'upload': open(path, 'rb')}
    session = requests.session()
    csrfToken = session.get(url_upload).cookies['csrftoken']
    csrf = {'csrfmiddlewaretoken': csrfToken}
    response = session.post(url_upload, files=files, data=csrf)
    if response.status_code == 200:
        print("Upload completato con successo. Path: " + path)
        sleep(5)
    else:
        print("Si è verificato un errore durante l'upload.")
        return

    url_customize = 'http://localhost:5005/customize.html'
    context = {"smells": [
        "DataSmellType.EXTREME_VALUE_SMELL",
        "DataSmellType.SUSPECT_SIGN_SMELL",
        "DataSmellType.CASING_SMELL",
        "DataSmellType.MISSING_VALUE_SMELL",
        "DataSmellType.FLOATING_POINT_NUMBER_AS_STRING_SMELL"
    ],
        "columns": df.columns,
        'csrfmiddlewaretoken': csrfToken
    }

    response = session.post(url_customize, context)
    if response.status_code == 200:
        print("Customizzazione effettuata.")
        sleep(5)
    else:
        print("Si è verificato un errore durante la customizzazione.")
        return

    url_results = 'http://localhost:5005/results.html'


    response = session.post(url_results, data=csrf)
    if response.status_code == 200:
        print("Detection effettuata.")
    else:
        print("Si è verificato un errore durante la detection.")
        return

    req = session.get(url_results)
    result_page = req.text
    soup = BeautifulSoup(result_page, 'html.parser')
    columns = soup.find_all("div", attrs={"aria-labelledby":"tabs-icons-text-2-tab"})
    df = pd.DataFrame()
    for column in columns:
        tables = column.find_all("tbody", attrs={"class":"list"})

        header = [th.get_text(strip=True) for th in column.find_all('th')]
        for table in tables:
            features = []

            rows = []
            for row in table.find_all('tr'):
                rows.append([td.get_text(strip=True) for td in row.find_all('td')])
                features.append(column.get("id"))

            temp_df = pd.DataFrame(rows, columns=header)
            temp_df["Feature"] = features

            df = pd.concat([df, temp_df])

    df = df.dropna()
    if not df.empty:
        df["Dataset"] = name
    return df



def main(paths, subnames, names, smell_type):
    for p in range(0, len(paths)):
        df = smell_detection(paths[p], subnames[p])
        print(df)
        if(names.__len__() > 1):
            new_path = "outputDSD/"+smell_type+"/"+names[p]+"_datasmells.csv"
            if Path(new_path).exists():
                try:
                    df_tmp = pd.read_csv(new_path, index_col=False)
                    df = pd.concat([df_tmp, df])
                except pd.errors.EmptyDataError:
                    print("\"outputDSD/"+names[p]+"_datasmells.csv\" è vuoto")
            df.to_csv(new_path, index=False)


if __name__ == "__main__":
    
    if not Path("outputDSD").exists():
      os.mkdir("outputDSD")
      
    if not Path("outputDSD/Casing").exists():
      os.mkdir("outputDSD/Casing")

    paths = ["smaller_datasets/kdd-census_1.csv", "smaller_datasets/kdd-census_2.csv", "smaller_datasets/diabetic_data_1.csv", "smaller_datasets/diabetic_data_2.csv", "smaller_datasets/diabetic_data_3.csv", "datasets_no_data_smells/nursery.csv", "datasets_no_data_smells/Firefighter_Promotion_Exam_Scores.csv", "datasets_no_data_smells/tae.csv"]
    subnames = ["kdd-census_1", "kdd-census_2", "diabetic_data_1", "diabetic_data_2", "diabetic_data_3", "nursery", "Firefighter_Promotion_Exam_Scores", "tae"]
    names = ["kdd-census", "kdd-census", "diabetic_data", "diabetic_data", "diabetic_data", "nursery", "Firefighter_Promotion_Exam_Scores", "tae"]
    
    main(paths, subnames, names, "Base")

    #Attualmente i risultati prodotti da questo script non sono sempre in linea con quelli ottenuti utilizzando il tool via web, con i medesimi input. 