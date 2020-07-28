# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd

df = pd.read_csv('/data/train.csv')
labels = df[['image_name', 'diagnosis']].set_index('image_name')
labels = pd.get_dummies(labels).reset_index()
labels.to_csv('labels.csv', index=False)
