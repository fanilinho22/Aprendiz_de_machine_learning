import pandas as pd
import matplotlib.pyplot as plt

dados = pd.read_csv(r"C:\\Users\\User\\Documents\\BDmachine\\athlete_events.csv")
dados.rename(columns = {"Age" : "Idade"}, inplace = True)


dados.hist(column="Idade", bins=100)
plt.show()