import ROC_analysis
import numpy as np
import pandas as pd
import random
import feature_analysis
import sm_logistic_regression
import statistical_tests
from feature_analysis import FeatureOverview, feature_list_visualizer
import sm_logistic_regression as sm_log
from ROC_analysis import roc_curve_constructor
import matplotlib.pyplot as plt
import statistical_tests as st
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.linear_model import LinearRegression
from mpl_toolkits import mplot3d


if __name__ == '__main__':
    # Read Data Frame In
    df = pd.read_csv("Results3.csv", index_col=0)

    # Remove outliers
    df["Flowrate"] = df["Flowrate"][df["Flowrate"] < 14]
    df["%DS"] = np.sqrt(4 * df["Stenosis Area"] / 4) / np.sqrt(4 * df["Mean Area"] / 4)
    df = df.dropna()

    X = df.drop(["FFR"], axis=1)
    X["velocity"] = (df['Flowrate']/(10**6))/(df['Mean Area']/(10**6))
    velocity = X['velocity']

    X['stenosis_diameter'] = (np.sqrt(4 * df["Stenosis Area"] / 4)/1000)
    stenosis_diameter = X['stenosis_diameter']

    X['Stenosis Area'] = df['Stenosis Area']/(10**6)
    stenosis_area = X['Stenosis Area']

    X['reynolds'] = velocity*stenosis_diameter*1045/0.0035
    reynolds = X['reynolds']
    print(min(reynolds))
    plt.hist(reynolds)
    plt.show()

    flow_rate = df['Flowrate']/(10**6)

    beta = X['%DS']

    X["orificeterm"] = np.power(flow_rate, 2)*(1 - np.power(beta, 4)) / (2*1045*np.power(stenosis_area, 2))
    orifice_ideal = X['orificeterm']

    aortic_pressure = 100 ##mm hg
    X['correction'] = beta/np.log(reynolds)
    discharge_coefficient = X['correction']
    # OLS correctionX['reynolds']*(-0.000304) + X['%DS'] or 0.87

    X['FFRpred'] = 1 - (1/aortic_pressure)*(orifice_ideal/np.power(discharge_coefficient, 2))
    print(X['FFRpred'])

    y = pd.DataFrame()
    y["FFR"] = df['FFR']
    y['Difference'] = y['FFR']/X['FFRpred']
    y["binary"] = df['Diagnosis']
    print(X['FFRpred'].corr(y['FFR']))
    # endregion

    negative = X[df['Diagnosis'] == 0]
    positive = X[df['Diagnosis'] == 1]
    groups = {"Positive": positive, 'Negative': negative}
    univariate = statistical_tests.statTests(groups=groups, df=X)
    univariate.mannwhitney_test(column_name='FFRpred')

    clf = sm_logistic_regression.logisticRegression(X=X[['correction', '%DS']], y=y['binary'])
    clf_fitted, clf_probs, clf_preds = clf.logistic_regression_clf()
    clf.results()
    print(clf.summary)
    print(clf.accuracy)

    clf2 = sm_logistic_regression.logisticRegression(X=X[['correction']], y=y['binary'])
    clf2_fitted, clf2_probs, clf2_preds = clf2.logistic_regression_clf()
    clf.results()
    print(clf.summary)
    print(clf.accuracy)

    plt.figure(1)
    plt.scatter(x=X['FFRpred'], y=y['FFR'])
    plt.ylabel('FFR_true')
    plt.xlabel('FFR_predicted')
    plt.show()
    plt.close(1)

    roc = roc_curve_constructor(clf=clf_fitted, X=X[['correction']], y_true=y['binary'],
                                y_predict=None, ax=None, name="Discharge Coefficient + %DS")
    roc2 = roc_curve_constructor(clf=clf2_fitted, X=X[['correction']], y_true=y['binary'],
                                 y_predict=None, ax=roc.ROC.ax_, name="Discharge Coefficient")
    plt.show()


