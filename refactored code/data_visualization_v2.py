import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm

class VisualizationGraphs:
    def __init__(self, data):
        self.data = data

    def correlation_graph_generator(self, data, independend_variables, dependend_variable, filename_cor):
        names = self.data.columns
        sns.set(font_scale=1.5)
        plt.figure(figsize=(12, 10))
        sns.heatmap(np.corrcoef(data.values.T), cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 13}, yticklabels=names, xticklabels=names)
        plt.savefig(filename_cor)


    def dispersion_graph_generator(self, data, independend_variables, dependend_variable, filename_disp):
        sns.pairplot(data, x_vars=independend_variables, y_vars=dependend_variable, diag_kind=None)
        plt.tight_layout()
        plt.savefig(filename_disp)


    def histogram_generator(self, data):
        #[data.hist(column) for column in data.columns]
        data.hist(column='fixed acidity')
        data.hist(column='volatile acidity')
        data.hist(column='residual sugar')
        data.hist(column='citric acid')
        data.hist(column='chlorides')
        data.hist(column='free sulfur dioxide')
        data.hist(column='total sulfur dioxide')
        data.hist(column='density')
        data.hist(column='pH')
        data.hist(column='sulphates')
        data.hist(column='alcohol')
        data.hist(column='quality')


    def final_plot(self, name, y_actual, y_predicted_denor):
        fig = plt.figure()
        a1 = fig.add_axes([0,0,1,1])
        x = range(len(y_actual))
        a1.plot(x, y_actual, 'ro')
        a1.set_ylabel('Actual')
        a2 = a1.twinx()
        a2.plot(x, y_predicted_denor, 'o')
        a2.set_ylabel('Predicted')
        fig.legend(labels=('Actual', 'Predicted'), loc='upper left')
        plt.savefig(f"{name} Wykres wartości przewidywanych względem obserwowanych.png")

    def graph_for_kmeans_result(self, clusters, y):
        sns.set()  # make the plots look pretty
        df = pd.DataFrame({'Grupa do której należą wina': clusters + 1, 'Jakość wina': y})
        df['dummy'] = 1
        ag = df.groupby(['Grupa do której należą wina', 'Jakość wina']).sum().unstack()
        ag.columns = ag.columns.droplevel()

        ag.plot(kind='bar', colormap=cm.Accent, width=1)
        plt.legend(loc='upper center')

        plt.savefig("Porównanie jakości z grupowaniem")