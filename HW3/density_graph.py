import pandas as pd
import numpy as np
import plotly.graph_objs as go
import chart_studio.plotly as py
from plotly.graph_objs import *
from matplotlib import pyplot as plt
from matplotlib import cm as cm
import seaborn as sns


def scatter_with_color_dimension_graph(features, target, layout_labels):
    """ Scatter with color dimension graph to visualize the density of the
    Given feature with target
    : param feature :
    : param target :
    : param layout_labels :
    : retrun :
    """

    trace1 = go.Scatter(
        y=features,
        mode='markers',
        marker=dict(
            size=6,
            color=target,
            colorscale='Viridis',
            showscale=True
        )
    )

    layout = go.Layout(
        title=layout_labels[2],
        xaxis=dict(title=layout_labels[0]), yaxis=dict(title=layout_labels[1])
    )
    data = [trace1]
    fig = go.Figure(data=data, layout=layout)
    fig.show()


def correlation_matrix(df, feature_names):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    corr = df.corr()
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Abalone Feature Correlation')
    ax1.set_xticklabels(feature_names, fontsize=6)
    ax1.set_yticklabels(feature_names, fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.75, .8, .85, .90, .95, 1])
    plt.show()


density_graphs = False
corr_matrix = True
DATASET_PATH = './glass_data_labeled.csv'
df = pd.read_csv(DATASET_PATH)

feature_names = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']

if density_graphs:
    for i in range(len(feature_names)):
        features = df[feature_names[i]]
        targets = df['Type']
        xlabel = 'Data Index'
        ylabel = feature_names[i] + ' Value'
        graph_title = feature_names[i] + ' -- Glass Type Density Graph'
        graph_labels = [xlabel, ylabel, graph_title]
        scatter_with_color_dimension_graph(features, targets, graph_labels)

if corr_matrix:
    # correlation_matrix(df, feature_names)
    df_features = df.loc[:, 'RI':'Fe']
    corr = df_features.corr()
    corr_matrix = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, fmt="0.2f", cmap="YlGnBu")
    fig = corr_matrix.get_figure()
    fig.savefig("correlation_matrix.png")
