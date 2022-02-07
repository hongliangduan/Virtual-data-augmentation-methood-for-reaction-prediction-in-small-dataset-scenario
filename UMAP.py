# import umap
import umap.umap_ as umap
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import matplotlib.pyplot as plt
import pandas as pd
import csv
import plotly.express as px
import numpy as np
import plotly
import plotly.graph_objs as go
# import yfinance as yf


f1 = open('UMAP2/Augment_Halogen.txt','r')
f2=open('UMAP2/Augment_Silicon.txt','r')
f3=open('UMAP2/Halogen.txt','r')
f4=open('UMAP2/Silicon.txt','r')
l1=f1.readlines()
l2=f2.readlines()
l3=f3.readlines()
l4=f4.readlines()
l=l1+l2+l3+l4
fps =[MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(x))for x in l]
reducer = umap.UMAP(random_state=100, n_neighbors=150, min_dist=1)
# reducer = umap.UMAP(random_state=100, n_neighbors=50, min_dist=1)

embedding = reducer.fit_transform(fps)

x_1 = []
y_1 = []
for row in embedding[:len(l1)]:

    x_1.append(row[0])
    y_1.append(row[1])


x_2 = []
y_2 = []
for row in embedding[len(l1):len(l1)+len(l2)]:

    x_2.append(row[0])
    y_2.append(row[1])

x_3 = []
y_3 = []
for row in embedding[len(l1)+len(l2):len(l1)+len(l2)+len(l3)]:

    x_3.append(row[0])
    y_3.append(row[1])


x_4 = []
y_4 = []
for row in embedding[len(l1)+len(l2)+len(l3):]:

    x_4.append(row[0])
    y_4.append(row[1])


trace1 = go.Scatter(x=x_1, y=y_1, name='augmented_halogen', text=l1,
                    mode='markers', marker=dict(color='#5698c3'), opacity=0.5)

trace2 = go.Scatter(x=x_2, y=y_2, name='augmented_silicon', text=l2,
                    mode='markers', marker=dict(color='#f0a1a8'), opacity=0.5)

trace3 = go.Scatter(x=x_3, y=y_3, name='halogen', text=l3,
                    mode='markers', marker=dict(color='#5698c3'), opacity=1)

trace4 = go.Scatter(x=x_4, y=y_4, name='silicon', text=l4,
                    mode='markers', marker=dict(color='#f0a1a8'), opacity=1)



# layout = go.Layout(template='plotly_white', title='HIYAMA')
layout = go.Layout(template='plotly_white')


data = [trace1, trace2, trace3, trace4]
fig = go.Figure(data, layout=layout)
fig.update_layout({'font':dict(size=25)})

plotly.offline.plot(fig, filename='hiyama.html')