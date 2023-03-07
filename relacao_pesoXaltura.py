import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

dados = pd.read_csv(r"C:\\Users\\User\\Documents\\BDmachine\\athlete_events.csv")

#x = [1,2,3,4,5,6,7,8,9,10]
#y = [1,2,3,4,5,6,7,8,9,10]
#plt.scatter(x,y)
#plt.show()
#x1 = np.arange(0,1000,2)
#plt.plot(x1,x1**2)
#plt.show()

masculinos = dados.loc[dados["Sex"]=="M"]
a = masculinos["Height"]
p = masculinos["Weight"]

plt.scatter(a,p)
plt.show()