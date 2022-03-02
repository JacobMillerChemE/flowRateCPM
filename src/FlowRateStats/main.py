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

def get_model_metrics(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    sensitivity = tp/(tp+fn)
    specificity = tn/(fp+tn)
    PPV = tp/(tp+fp)
    NPV = tn/(fn+tn)
    print(f'Sensitivity: {sensitivity}')
    print(f'Specificity: {specificity}')
    print(f'PPV: {PPV}')
    print(f'NPV: {NPV}', '\n')

def fast_regressor(X_array, y_array, multicollinearity_check):
    reg = logisticRegression(X=X_array, y=y_array)
    reg_fitted, reg_probs, reg_pred = reg.logistic_regression_clf()
    reg.results()
    if multicollinearity_check == True:
        print(reg.multicollinearity_check()) 
    print(reg.summary)
    print('Accuracy: {}'.format(reg.accuracy))
    get_model_metrics(reg.confusion_matrix)
    return reg_fitted

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

    # %DS Log Regressor
    DS_LR_fitted  = fast_regressor(X_array=X[['%DS']], y_array=y['binary'], 
            multicollinearity_check=False)

    # FLOW RATE LR
    Q_fitted = fast_regressor(X_array=X[['Flowrate']], y_array=y['binary'],
            multicollinearity_check=False)

    # %DS + FLOW RATE
    LR_fitted = fast_regressor(X_array=X[['%DS', 'Flowrate']], 
            y_array=y['binary'], multicollinearity_check=True)

    # %DS + FLOW RATE + MLA
    mla_fitted = fast_regressor(X_array=X[['%DS', 'Flowrate', 'Stenosis Area']],
            y_array=y['binary'], multicollinearity_check=True)

    # %DS + FLOW RATE + VOLUME
    volume_fitted = fast_regressor(X_array=X[['%DS', 'Flowrate', 'Volume']], 
            y_array=y['binary'], multicollinearity_check=True)

    # ROC CURVES
    diam_stenosis = roc_curve_constructor(DS_LR_fitted, X=X[['%DS']], 
            y_true=y['binary'], y_predict=None, ax=None, name='%DS') 

    flow_rate = roc_curve_constructor(clf=Q_fitted, X=X[['Flowrate']], 
            y_true=y['binary'], y_predict=None, ax=diam_stenosis.ROC.ax_, 
            name='Coronary Flow Rate')
    
    full = roc_curve_constructor(clf=LR_fitted, X=X[['%DS', 'Flowrate']], 
            y_true=y['binary'], y_predict=None, ax=flow_rate.ROC.ax_, 
            name='%DS + Coronary Flow Rate')
   
    mla_roc = roc_curve_constructor(clf=mla_fitted, X=X[['%DS', 'Flowrate', 
        'Stenosis Area']], y_true=y['binary'], y_predict=None, ax=full.ROC.ax_, 
        name='%DS + Coronary Flow Rate + MLA')
    
    volume_roc = roc_curve_constructor(clf=volume_fitted, 
            X=X[['%DS', 'Flowrate', 'Volume']], y_true=y['binary'],
            y_predict=None, ax=mla_roc.ROC.ax_, 
            name='%DS + Coronary Flow Rate + Volume')

    plt.plot([0, 1], [0, 1], c='black', linestyle='dashed')
    plt.show()
