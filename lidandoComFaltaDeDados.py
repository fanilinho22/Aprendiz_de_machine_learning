import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

dados = pd.read_csv(r"C:\\Users\\User\\Documents\\BDmachine\\athlete_events.csv")
dados2 = dados.dropna()

print(dados2.head())
