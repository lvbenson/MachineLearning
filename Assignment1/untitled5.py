#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 00:01:18 2020

@author: lvbenson
"""
import sklearn
from sklearn import datasets
import openml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


dataset = openml.datasets.get_dataset(31)


print(
    f"This is dataset '{dataset.name}', the target feature is "
    f"'{dataset.default_target_attribute}'"
)
print(f"URL: {dataset.url}")
print(dataset.description[:600])

X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="dataframe", target=dataset.default_target_attribute
)


sns.set_style("darkgrid")


def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)


# We combine all the data so that we can map the different
# examples to different colors according to the classes.
#combined_data = pd.concat([X, y], axis=1)
#iris_plot = sns.pairplot(combined_data, hue="class")
#iris_plot.map_upper(hide_current_axis)
#plt.show()
    
    
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute
)
eeg = pd.DataFrame(X, columns=attribute_names)
eeg["class"] = y
print(eeg[:10])


X, y, categorical_indicator, attribute_names = dataset.get_data(
    target=dataset.default_target_attribute, dataset_format="dataframe"
)
print(X.head())
print(X.info())


eegs = eeg.sample(n=1000)
_ = pd.plotting.scatter_matrix(
    eegs.iloc[:100, :4],
    c=eegs[:100]["class"],
    figsize=(10, 10),
    marker="o",
    hist_kwds={"bins": 5},
    alpha=0.8,
    cmap="plasma",
)
    