from typing import Any, Tuple

import pandas as pd

from data_loader import load_data
from data_visualization_v2 import (
    plot_correlation_graph,
    plot_dispersion_graphs,
    plot_histograms,
)
from settings import X_COLUMNS_NAMES, Y_COLUMN_NAME


def explore_data(data: pd.DataFrame, save_plots=False) -> Tuple[Any, int]:
    all_data_columns_names = X_COLUMNS_NAMES.copy()
    all_data_columns_names.append(Y_COLUMN_NAME)

    data = data[all_data_columns_names]
    # Generation of the plots
    if save_plots:
        plot_correlation_graph(data, filename_cor="Correlation_graph.png")
        plot_dispersion_graphs(
            data, X_COLUMNS_NAMES, Y_COLUMN_NAME, filename_disp="Dispersion_graph.png"
        )
        plot_histograms(data)

    return data


# data = load_data()
# explore_data(data)
