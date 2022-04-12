import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from data_visualization_v2 import VisualizationGraphs

def main():
    data = pd.read_csv("C:\\Users\\Krzychiu\\Documents\\Analiza_danych_studia\\ED\\KL_projekt_zaliczeniowy\\winequality-white.csv", sep=';')
    random_seed = 316733
    pd.set_option('display.max_columns', 12)
    #fileSafer("tabela_nr_1.txt",data.describe())
    #lack_of_data_checker(data) #- sprawdzenie czy istnieją brakujące dane

    #Wybór zmniennej celu, w przypadku wybranych danych zmniennej quality (11 kolumna) oraz predyktorów
    dependend_variable_col_num = 11
    independend_variables_col_num, dependend_variable, independend_variables = independend_and_dependend_variables_generator(data, dependend_variable_col_num)

    correlation_graph_generator(data, independend_variables, dependend_variable, filename_cor = "Macierz_korelacji.png")
    dispersion_graph_generator(data, independend_variables, dependend_variable, filename_disp="Macierz_rozrzutu.png")
    histogram_generator(data)


    #Ponowny wybór danych i generacja wykresów rozrzutu oraz macierzy korelacji
    data = data[data.columns[[0, 1, 2, 3, 4, 6, 8, 9, 10, 11]]]
    dependend_variable_col_num = 9
    independend_variables_col_num, dependend_variable, independend_variables = independend_and_dependend_variables_generator(data, dependend_variable_col_num)
    correlation_graph_generator(data, independend_variables, dependend_variable, filename_cor="Macierz_korelacji_zmodyfikowana.png")

    #data = riding_off_outliered_data(data) #funkcja do usunięcia obserwacji odstających

    # Podział danych na zbiór uczący i testowy
    data_train, data_test = train_test_split(data, test_size=0.25, random_state=random_seed)
    y_test = data_test[dependend_variable]
    y_train = data_train[dependend_variable]


    #standaryzacja danych
    X_train_norm, X_test_norm, y_train_norm = data_standarization(data_train, data_test, independend_variables_col_num, dependend_variable_col_num)

    #klasyfikacja danych
    #knn_classification_crossvalidation(y_test, y_train, X_train_norm, X_test_norm) - niewykorzystywane gdyż gorzej działą

    knn_classification(y_test, y_train, X_train_norm, X_test_norm)
    perceptor_num = 200
    MLP_classification(y_test, y_train, X_train_norm, X_test_norm, random_seed, perceptor_num=perceptor_num)

    #estymacja
    MLP_estimation(y_test, y_train, X_train_norm, X_test_norm, y_train_norm, random_seed, perceptor_num = perceptor_num)

    # grupowanie
    clustering_data_using_kmeans(data_train, data_test, y_test, independend_variables, random_seed)


def lack_of_data_checker(data):
    nulls_summary = pd.DataFrame(data.isnull().any(), columns=['Nulls'])
    nulls_summary['Num_of_nulls [qty]'] = pd.DataFrame(data.isnull().sum())
    nulls_summary['Num_of_nulls [%]'] = round((data.isnull().mean() * 100), 2)
    print(nulls_summary)


def independend_and_dependend_variables_generator(data, dependend_variable_col_num):
    x = list(range(0, data.shape[1]))
    x.remove(dependend_variable_col_num)
    independend_variables_col_num = x
    dependend_variable = data.columns[dependend_variable_col_num]
    independend_variables = data.columns[independend_variables_col_num]
    return independend_variables_col_num, dependend_variable, independend_variables


def riding_off_outliered_data(data):
    data_without_outliers = data.copy()
    outliers_15iqr = boudaries(data)
    for row in outliers_15iqr.iterrows():
        data_without_outliers = data_without_outliers[(data_without_outliers[row[0]] >= row[1]['lower_boundary']) & (
                    data_without_outliers[row[0]] <= row[1]['upper_boundary'])]
    return data_without_outliers


def boudaries(data):
    q1 = data.quantile(0.05)  # wartości zmiennej na granicy pierwszego i drugiego kwartyla
    q3 = data.quantile(0.95)  # wartości zmiennej na granicy trzeciego i czwartego kwartyla
    iqr = q3 - q1
    low_boundary = (q1 - 1.5 * iqr)
    upp_boundary = (q3 + 1.5 * iqr)
    num_of_outliers_L = (data[iqr.index] < low_boundary).sum()
    num_of_outliers_U = (data[iqr.index] > upp_boundary).sum()
    outliers_15iqr = pd.DataFrame(
        {'lower_boundary': low_boundary, 'upper_boundary': upp_boundary, 'num_of_outliers_L': num_of_outliers_L,
         'num_of_outliers_U': num_of_outliers_U})
    return outliers_15iqr


def data_standarization(data_train, data_test, independend_variable_col_num ,dependend_variable_col_num):
    scaler = MinMaxScaler()
    scaler.fit(data_train)
    data_train_norm = scaler.transform(data_train)
    data_test_norm = scaler.transform(data_test)
    X_train_norm = data_train_norm[:, independend_variable_col_num]
    X_test_norm = data_test_norm[:, independend_variable_col_num]
    y_train_norm = data_train_norm[:, dependend_variable_col_num]
    return X_train_norm, X_test_norm, y_train_norm


def fileSafer(filename, file):
    with open(filename, mode='a', encoding='utf-8') as wf:
        wf.write(f"{file}\n")

main()