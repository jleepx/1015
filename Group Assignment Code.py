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
from matplotlib.colors import LinearSegmentedColormap
sb.set()
np.set_printoptions(suppress=True)
pd.options.display.float_format = '{:,.8f}'.format

data = pd.read_csv("movies.csv")
xvarlist = ["genre", "score", "votes", "director", "writer", "star", "budget", "runtime"]
allvarlist = xvarlist + ["gross"] + ["profit"]
catvarlist = ["genre", "director", "writer", "star"]

data["profit"] = data["gross"] - data["budget"]
cleanoridata = data.dropna()
cleanoridata = cleanoridata[allvarlist]

cleandata = cleanoridata.copy()
catdictofdict = {}
for var in catvarlist:
    catdictofdict[var] = dict(cleandata.groupby(var)['gross'].mean())
    cleandata[var].replace(to_replace = catdictofdict[var], inplace = True)
    
f = plt.figure(figsize=(12, 12))
f.suptitle("Correlation Matrix Heatmap")
cmap = LinearSegmentedColormap.from_list('', ['blue', 'white', 'blue'])
sb.heatmap(cleandata.corr(), vmin = -1, vmax = 1, annot = True, fmt = ".2f", cmap = cmap, )


boxplot = cleanoridata[(cleanoridata["director"] == "Clint Eastwood") | (cleanoridata["director"] == "Woody Allen" )|(cleanoridata["director"] == "Steven Spielberg" )|(cleanoridata["director"] == "Ron Howard" )]
sb.catplot(data = boxplot, x = 'director', y = 'gross', kind = "box")
boxplot2 = cleanoridata[(cleanoridata["star"] == "Nicolas Cage") | (cleanoridata["star"] == "Tom Hanks" )|(cleanoridata["star"] == "Robert De Niro" )|(cleanoridata["star"] == "Bruce Willis" )]
sb.catplot(data = boxplot2, x = 'star', y = 'gross', kind = "box")
boxplot3 = cleanoridata[(cleanoridata["writer"] == "John Hughes") | (cleanoridata["writer"] == "Luc Besson" )|(cleanoridata["writer"] == "Joel Coen" )|(cleanoridata["writer"] == "Woody Allen" )]
sb.catplot(data = boxplot3, x = 'writer', y = 'gross', kind = "box")

sb.catplot(data = boxplot, x = 'director', y = 'profit', kind = "box")
sb.catplot(data = boxplot2, x = 'star', y = 'profit', kind = "box")
sb.catplot(data = boxplot3, x = 'writer', y = 'profit', kind = "box")

xtraindata, xtestdata, grosstraindata, grosstestdata, profittraindata, profittestdata = train_test_split(cleandata[xvarlist], cleandata[["gross"]], cleandata[["profit"]], test_size = 0.2)
alltraindata = pd.concat([xtraindata, grosstraindata, profittraindata], axis = 1)
alltestdata = pd.concat([xtestdata, grosstestdata, profittestdata], axis = 1)

lm = []
lmdescript = []

lmdescript.append("Linear Model 0: to predict profit using genre, score, votes, director, writer, star, budget, runtime as independent variables") 
lm.append(ols("profit ~ genre + score + votes + director + writer + star + budget + runtime", alltraindata).fit())
lmdescript.append("Linear Model 1: same as Linear Model 0 but excluding categorical variables director, writer, star")
lm.append(ols("profit ~ score + votes + budget + runtime", alltraindata).fit())
lmdescript.append("Linear Model 2: same as Linear Model 0 but excluding numerical variables")
lm.append(ols("profit ~ director + writer + star", alltraindata).fit())
lmdescript.append("Linear Model 3: same as Linear Model 0 but predicting gross revenue instead of budget")
lm.append(ols("gross ~ genre + score + votes + director + writer + star + budget + runtime", alltraindata).fit())
lmdescript.append("Linear Model 4: same as Linear Model 1 but predicting gross revenue instead of budget")
lm.append(ols("gross ~ score + votes + budget + runtime", alltraindata).fit())
lmdescript.append("Linear Model 5: same as Linear Model 2 but predicting gross revenue instead of budget")
lm.append(ols("gross ~ director + writer + star", alltraindata).fit())
lmdescript.append("Linear Model 6: to estimate budget needed to attain a desired profit using genre, score, votes, director, writer, star, runtime, profit as independent variables")
lm.append(ols("budget ~ genre + score + votes + director + writer + star + runtime + profit", alltraindata).fit())
lmdescript.append("Linear Model 7: same as Linear Model 6 but excluding categorical variables")
lm.append(ols("budget ~ genre + score + votes + runtime + profit", alltraindata).fit())
lmdescript.append("Linear Model 8: same as Linear Model 6 but excluding numerical variables except profit")
lm.append(ols("budget ~ director + writer + star + profit", alltraindata).fit())

for i in range(9):
    print(lmdescript[i])
    print("Linear model {} summary".format(i))
    print(lm[i].summary())
    print("")
    print("p-values")
    print(lm[i].pvalues)
    print("")
    print("coefficients")
    print(lm[i].params)
    print("")
    print("R-squared (Goodness of Fit): ", lm[i].rsquared)
    
    if (i < 3):
        print("Mean Squared Error (Prediction Accuracy): ", mean_squared_error(profittestdata, lm[i].predict(exog = alltestdata)))
    elif (3 <= i <= 5):
        print("Mean Squared Error (Prediction Accuracy): ", mean_squared_error(grosstestdata, lm[i].predict(exog = alltestdata)))
    else:
        print("Mean Squared Error (Prediction Accuracy): ", mean_squared_error(alltestdata[["budget"]], lm[i].predict(exog = alltestdata)))
    print("")


f, axs = plt.subplots(9,2, figsize = (16,50))

for i in range(9):
    for j in range(2):
        if (j == 0):
            exogdata = alltraindata
            s = "Train"
        else:
            exogdata = alltestdata
            s = "Test"
        
        if (i < 3):
            response = "profit"
        elif (3 <= i < 6):
            response = "gross"
        else:
            response = "budget"
            
        responsetrue = exogdata[[response]]
        responsepredict = pd.DataFrame(lm[i].predict(exog = exogdata))

        axs[i,j].plot(responsetrue,responsepredict, "xr")
        axs[i,j].plot(responsetrue,responsetrue,'-c',linewidth=1)
        axs[i,j].set(title = "Graph of Predicted {r} Against True {r} for {t} Set (Using Linear Model {k})".format(r = response, t = s, k = i), autoscale_on = True, xlabel = "True {r}".format(r = response), ylabel = "Predicted {r}".format(r = response))


h
