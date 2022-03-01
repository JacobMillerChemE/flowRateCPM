import numpy as np
import pandas as pd
from ROC_analysis import roc_curve_constructor
from sm_logistic_regression import logisticRegression
import matplotlib.pyplot as plt
import os 

def get_data_path():
    dirname = os.path.dirname(__file__)
    data_path = dirname.replace('/src/FlowRateStats', '/dataFrame/Results3.csv')
    return data_path


if __name__ == '__main__':
    path = get_data_path()
    
    # Read Data Frame In #
    df = pd.read_csv(path, index_col=0)

    # region Format input dataframe and drop outliers
    Q1, Q3 = np.percentile(df["Flowrate"], [25, 75])
    IQR = Q3 - Q1
    outlier_criteria = Q3 + 1.5*IQR
    df = df.dropna()
    df = df[df["Flowrate"] < outlier_criteria]
    # endregion

    # region FEATURE ENGINEERING
    df["%DS"] = (1 - np.sqrt(4 * df["Stenosis Area"] / 4) / np.sqrt(4 * df["Mean Area"] / 4)) * 100
    df = df.drop(["Mean Area", "length", "Nominal", "Stenosis Position", "Beta"], axis=1)
    # endregion

    # region CREATE INPUT AND TARGET ARRAYS
    X = df.drop(["FFR", "Diagnosis"], axis=1)  # Data frame with %DS and Nominal
    print(X.corr())
    y = pd.DataFrame()
    y["FFR"] = df['FFR']
    y["binary"] = df['Diagnosis']
    # endregion

    # region %DS LR
    DS_LR = logisticRegression(X=X[['%DS']], y=y['binary'])
    DS_LR_fitted, DS_LR_probs, DS_LR_pred = DS_LR.logistic_regression_clf()
    DS_LR.results()
    print(DS_LR.summary)
    print('Confusion Matrix:\n {}\n'.format(DS_LR.confusion_matrix))
    print('Accuracy: {}'.format(DS_LR.accuracy))
    # endregion

    # region FLOW RATE LR
    Q_LR = logisticRegression(X=X[['Flowrate']], y=y['binary'])
    Q_fitted, Q_probs, Q_pred = Q_LR.logistic_regression_clf()
    Q_LR.results()
    print(Q_LR.summary)
    # endregion

    # region %DS + FLOW RATE
    LR = logisticRegression(X=X[['%DS', 'Flowrate']], y=y['binary'])
    LR_fitted, LR_probs, LR_pred = LR.logistic_regression_clf()
    LR.results()
    print(LR.multicollinearity_check())
    print(LR.summary)
    print('Confusion Matrix:\n {}\n'.format(LR.confusion_matrix))
    print('Accuracy: {}'.format(LR.accuracy))
    # endregion

    # region %DS + FLOW RATE + MLA
    mla = logisticRegression(X=X[['%DS', 'Flowrate', 'Stenosis Area']], y=y['binary'])
    mla_fitted, mla_probs, mla_pred = mla.logistic_regression_clf()
    mla.results()
    print(mla.multicollinearity_check())
    print(mla.summary)
    # endregion

    # region %DS + FLOW RATE + VOLUME
    volume = logisticRegression(X=X[['%DS', 'Flowrate', 'Volume']], y=y['binary'])
    volume_fitted, volume_probs, volume_pred = volume.logistic_regression_clf()
    volume.results()
    print(volume.multicollinearity_check())
    print(volume.summary)
    print(volume.accuracy)
    # endregion

    # region ROC CURVES
    diam_stenosis = roc_curve_constructor(DS_LR_fitted, X=X[['%DS']], y_true=y['binary'], y_predict=None, ax=None,
                                          name='%DS')
    flow_rate = roc_curve_constructor(clf=Q_fitted, X=X[['Flowrate']], y_true=y['binary'], y_predict=None,
                                      ax=diam_stenosis.ROC.ax_, name='Coronary Flow Rate')
    full = roc_curve_constructor(clf=LR_fitted, X=X[['%DS', 'Flowrate']], y_true=y['binary'], y_predict=None,
                                 ax=flow_rate.ROC.ax_, name='%DS + Coronary Flow Rate')
    mla_roc = roc_curve_constructor(clf=mla_fitted, X=X[['%DS', 'Flowrate', 'Stenosis Area']], y_true=y['binary'],
                                    y_predict=None, ax=full.ROC.ax_, name='%DS + Coronary Flow Rate + MLA')
    volume_roc = roc_curve_constructor(clf=volume_fitted, X=X[['%DS', 'Flowrate', 'Volume']], y_true=y['binary'],
                                       y_predict=None, ax=mla_roc.ROC.ax_, name='%DS + Coronary Flow Rate + Volume')

    plt.plot([0, 1], [0, 1], c='black', linestyle='dashed')
    plt.show()
    # endregion
