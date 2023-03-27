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
allvarlist = xvarlist + ["gross"] + ["profit"]
catvarlist = ["genre", "director", "writer", "star"]

data["profit"] = data["gross"] - data["budget"]
cleandata = data.dropna()
cleandata = cleandata[allvarlist]

catdictofdict = {}
for var in catvarlist:
    catdictofdict[var] = dict(cleandata.groupby(var)['gross'].mean())
    cleandata[var].replace(to_replace = catdictofdict[var], inplace = True)
    
xtraindata, xtestdata, grosstraindata, grosstestdata, profittraindata, profittestdata = train_test_split(cleandata[xvarlist], cleandata[["gross"]], cleandata[["profit"]], test_size = 0.2)
lm1data = pd.concat([xtraindata, profittraindata], axis = 1)
alltestdata = pd.concat([xtestdata, grosstestdata, profittestdata], axis = 1)


#to predict profit 
lm1 = ols("profit ~ genre + score + votes + director + writer + star + budget + runtime", lm1data).fit()
print("Linear model 1 summary")
print(lm1.summary())
print("p-values")
print(lm1.pvalues)
print("")
print("coefficients")
print(lm1.params)
print("")
print("R-squared (Goodness of Fit): ", lm1.rsquared)
print("Mean Squared Error (Prediction Accuracy): ", mean_squared_error(profittestdata, lm1.predict(exog = xtestdata)))
print("")

#same as lm1 but excluding categorical variables
lm2 = ols("profit ~ score + votes + budget + runtime", lm1data).fit()
print("Linear model 2 summary")
print(lm2.summary())
print("p-values")
print(lm2.pvalues)
print("")
print("coefficients")
print(lm2.params)
print("")
print("R-squared (Goodness of Fit): ", lm2.rsquared)
print("Mean Squared Error (Prediction Accuracy): ", mean_squared_error(profittestdata, lm2.predict(exog = xtestdata[["score", "votes", "budget", "runtime"]])))
print("")

#to estimate budget needed to attain a desired profit
lm3 = ols("budget ~ genre + score + votes + director + writer + star + runtime + profit", lm1data).fit()
print("Linear model 3 summary")
print(lm3.summary())
print("p-values")
print(lm3.pvalues)
print("")
print("coefficients")
print(lm3.params)
print("")
print("R-squared (Goodness of Fit): ", lm3.rsquared)
print("Mean Squared Error (Prediction Accuracy): ", mean_squared_error(xtestdata[["budget"]], lm3.predict(exog = alltestdata[["genre", "score", "votes", "director", "writer", "star", "runtime", "profit"]])))
print("")

#same as lm1 but using gross as dependent variable
lm4data = pd.concat([xtraindata, grosstraindata], axis = 1)
lm4 = ols("gross ~ genre + score + votes + director + writer + star + budget + runtime", lm4data).fit()
print("Linear model 4 summary")
print(lm4.summary())
print("p-values")
print(lm4.pvalues)
print("")
print("coefficients")
print(lm4.params)
print("")
print("R-squared (Goodness of Fit): ", lm4.rsquared)
print("Mean Squared Error (Prediction Accuracy): ", mean_squared_error(grosstestdata, lm4.predict(exog = xtestdata)))
print("")

grosstestpredict = pd.DataFrame(lm4.predict(exog = xtestdata))
f, axes = plt.subplots()
df = pd.concat([grosstestdata, grosstestpredict], axis = 1)
axes.scatter(grosstestdata,grosstestpredict)
axes.plot(grosstestdata,grosstestdata,'w-',linewidth=1)

profittestpredict = pd.DataFrame(lm4.predict(exog = xtestdata))
f, axes = plt.subplots()
df = pd.concat([profittestdata, profittestpredict], axis = 1)
axes.scatter(profittestdata,profittestpredict)
axes.plot(profittestdata,profittestdata,'w-',linewidth=1)
