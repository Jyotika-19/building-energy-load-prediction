import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('mushrooms.csv')   #Loaded csv

null_values = pd.isnull(df).any(axis =1).sum()
print(null_values)                           #Identify if any null 
print((df == "?").sum())                     #Identify if any ? in the csv
print((df == "?").sum() * 100/ len(df))      #Percentage of ?

list_of_columns = ['odor', 'cap-shape', 'gill-color', 'veil-color', 'spore-print-color', 'cap-color', 'gill-size']

for i in list_of_columns:
  compare = pd.crosstab(df[i], df['class'])      #comparing how what odor compares to being e or p
  print(compare, "\n")
  barplot = compare.plot(kind = 'bar', stacked = True)   #Plotting the crosstab
  plt.xlabel(i)
  plt.ylabel("No. of Mushrooms")
  plt.title(f"Class vs {i}")
  plt.show()





