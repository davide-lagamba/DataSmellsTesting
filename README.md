# DataSmellsTesting - Tesi Magistrale - Davide La Gamba
Repository tesi magistrale in Informatica, Università degli Studi di Salerno a.a. 2023/2024

Titolo Tesi: Uno Studio Empirico sull’Impatto dei
Data Smell sull’Accuracy e le Performance di Machine Learning

Relatore della Tesi: Professore Fabio Palomba

### Dataset utilizzati
Sono stati utilizzati 5 dei dataset presenti nel _replication package_ del lavoro "[Unmasking Data Secrets: An Empirical Investigation into Data Smells and Their Impact on Data Quality](https://ieeexplore.ieee.org/document/10556050)" di Recupito et al.

In particolare, sono stati utilizzati, come punto di partenza, i seguenti dataset:
- kdd-census (df_l_1)
- diabetic_data (df_l_2)
- nursery (df_l_3)
- Firefighter_Promotion_Exam_Scores (df_m_1)
- tae (df_m_2)


### Strumenti utilizzati

Modelli di classificazione: https://github.com/shankarpandala/lazypredict

Generazione di dati sintetici: https://docs.sdv.dev/sdv

Supporto Tabelle: https://www.tablesgenerator.com/latex_tables

Data Smell Detection (DSD): https://github.com/mkerschbaumer/rb-data-smell-detection

### Descrizione lavoro svolto

Il lavoro svolto mira a valutare l'impatto sulle metriche di valutazione e sulle performance di modelli di classificazione dell'iniezione di alcuni tipi di data smell nei training set.

### Descrizione repository
- cleaned_datasets: cartella per contenere i dataset a cui sono rimosse le feature rappresentanti id o con data smell
- combined_df: cartella per contenere i dataset originali uniti ai dati sintetici con i data smell iniettati, da utilizzare per valutare la presenza dei data smell con il tool DSD, output dello script "combine_df"
- create_plot_scripts: contiene gli script per la creazione dei grafici di confronto dei risultati
- datasets_no_data_smells: contiene i dataset originali
- metadata: cartella per contenere i metadati dei dataset, utili ai data synthesizer del tool SDV
- outputComparison: cartella per contenere i risultati delle comparison tra i risultati
- outputDSD: cartella per contenere i risultati della detection di data smell tramite lo script "dsdetection.py"
- outputDSInjection: cartella per contenere i risultati dell'iniezione dei data smell sui dati sintetici generati
- outputLazyPredict: cartella per contenere i risultati dei modelli di classificazione di LazyPredict
- outputSDV: cartella per contenere i dati sintetici generati dai data synthesizer del tool SDV
- pre-processed_df: cartella per contenere i dataset in seguito alle fasi di pre-processing
- samples: cartella per contenere gli esempi dei record dei dataset da fornire a ChatGPT per effettuare data augmentation
- smaller_datasets: cartella per contenere dei sotto-dataset dei dataset originali da poter fornire in input al tool DSD
- synthesizers: cartella per contenere i metadati e i file .pkl dei data synthesizer del tool SDV
- combine_df.py: script per unire i dataset originali ai dati sintetici con i data smell iniettati
- compare_results.py: script per effettuare i confronti tra i risultati ottenuti
- data_augmentation_with_sdv.ipynb: script per effettuare la data augmentation tramite il tool SDV
- DatasetAnalysisAndPreprocessing.ipynb: script per analizzare i dataset ed effettuare il pre-processing dei dati originali
- datasmells_injection.py: script per iniettare i data smell nei dati sintetici generati
- DataSmellsTestBase/BaseSynthetic/Casing/FloatingPointNumberAsString/MissingValues/SuspectSign.ipynb: script per calcolare le metriche di valutazione dei modelli di classificazione sui diversi scenari dei dataset, utilizzando il tool LazyPredict


I file DataSmellsTest e la fase di data augmentation per il dataset "diabetic_data" del file data_augmentation_with_sdv.ipynb, sono stati eseguiti utilizzando [Google Colab](https://colab.google/), ed in particolare il runtime con TPU v2.8