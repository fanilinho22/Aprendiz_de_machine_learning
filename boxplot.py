import pandas as pd
import matplotlib.pyplot as plt

dados = pd.read_csv(r"C:\\Users\\User\\Documents\\BDmachine\\athlete_events.csv")

dados.boxplot(column=["Age"])
plt.show()