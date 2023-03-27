from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt 
import math
sb.set()
np.set_printoptions(suppress=True)
pd.options.display.float_format = '{:,.8f}'.format

data = pd.read_csv("movies.csv")
xvarlist = ["genre", "score", "votes", "director", "writer", "star", "budget", "runtime"]
allvarlist = xvarlist + ["gross"]
catvarlist = ["genre", "director", "writer", "star"]

cleandata = data.dropna()
cleandata = cleandata[allvarlist]

catdictofdict = {}
for var in catvarlist:
    catdictofdict[var] = dict(cleandata.groupby(var)['gross'].mean())
    cleandata[var].replace(to_replace = catdictofdict[var], inplace = True)
    
cleandata["profit"] = cleandata["gross"] - cleandata["budget"]



#to predict profit 
lm1 = ols("profit ~ genre + score + votes + director + writer + star + budget + runtime", cleandata).fit()
print("Linear model 1 summary")
print(lm1.summary())
print("p-values")
print(lm1.pvalues)
print("")
print("coefficients")
print(lm1.params)
print("")

#same as lm1 but excluding categorical variables
lm2 = ols("profit ~ score + votes + budget + runtime", cleandata).fit()
print("Linear model 2 summary")
print(lm2.summary())
print("p-values")
print(lm2.pvalues)
print("")
print("coefficients")
print(lm2.params)
print("")

#to estimate budget needed to attain a desired profit
lm3 = ols("budget ~ genre + score + votes + director + writer + star + runtime + profit", cleandata).fit()
print("Linear model 3 summary")
print(lm3.summary())
print("p-values")
print(lm3.pvalues)
print("")
print("coefficients")
print(lm3.params)
print("")
