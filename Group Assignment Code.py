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
pd.options.display.float_format = '{:,.5f}'.format

data = pd.read_csv("movies.csv")
xvarlist = ["genre", "score", "votes", "director", "writer", "star", "budget", "runtime"]
allvarlist = xvarlist + ["gross"]
catvarlist = ["genre", "director", "writer", "star"]

cleandata = data.dropna()
cleandata = cleandata[allvarlist]

catdictofdict = {}
for var in listofcatvar:
    catdictofdict[var] = dict(cleandata.groupby(var)['gross'].mean())
    cleandata[var].replace(to_replace = catdictofdict[var], inplace = True)
     
cleandata

