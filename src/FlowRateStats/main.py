import numpy as np
import pandas as pd
from ROC_analysis import roc_curve_constructor
from sm_logistic_regression import logisticRegression
import matplotlib.pyplot as plt
import os
import seaborn as sns
from statsmodels.graphics.factorplots import interaction_plot

def get_data_path():
    dirname = os.path.dirname(__file__)
    data_path = dirname.replace('/src/FlowRateStats', '/dataFrame/Results4.csv')
    return data_path

def get_model_metrics(confusion_matrix):
    ''' 
        Calculates sensitivity, specificity, PPV and NPV given a 
        confusion matrix from a logistic regression
    '''
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
    '''
    Takes x, y data and whether or not there needs to be a MC check and 
    creates a LR object, prints accuracy and performance metrics,
    and returns the fitted LR
    '''

    reg = logisticRegression(X=X_array, y=y_array)
    reg_fitted, reg_probs, reg_pred = reg.logistic_regression_clf()
    reg.results()
    if multicollinearity_check == True:
        print(reg.multicollinearity_check()) 
    print(reg.summary)
    print('Accuracy: {}'.format(reg.accuracy))
    get_model_metrics(reg.confusion_matrix)
    return reg_fitted

def quartile_generator(data, feature):
    '''
    Determines quartiles and returns the conditions and values to create 
    a pandas column labeled according to the list values that is used for 
    the interaction plot
    '''
    quartiles = np.percentile(data[feature], [25, 50, 75])
    conditions = [(df[feature] <= quartiles[0]),
            (df[feature] > quartiles[0]) & (df[feature] < quartiles[2]),
            (df[feature] > quartiles[2])]
    values = ('<Q1', 'IQR', 'Q3<')
    print(f'{feature} quartiles: {quartiles}')
    return conditions, values

def velocity_calc(Q, areas):
     velocs = 1000*Q/areas
     return velocs/1000

def diam_calc(areas):
     diams = np.sqrt(4*areas/np.pi)
     return diams/1000

def reynolds_calc(Q, areas):
    velocities = velocity_calc(Q, areas)
    diameters = diam_calc(areas)
    reynolds = diameters*velocities*1045/0.0035 
    return reynolds

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

    # FEATURE ENGINEERING
    df["%DS"] = (1 - np.sqrt(4 * df["Stenosis Area"] / np.pi) / np.sqrt(4 * df["Max Area"] / np.pi)) * 100
    Re_number = pd.Series(data=reynolds_calc(df['Flowrate'], df['Mean Area']), name='Reynolds Number')
    df = df.drop(["Mean Area", "length", "Nominal", "Stenosis Position", "Beta"], axis=1)

    # Create conditiosn and values and map them to a new pandas column to get 
    # a column labeled according to the quartile bin of another column
    q_conditions, q_values = quartile_generator(df, 'Flowrate')
    ds_conditions, ds_values = quartile_generator(df, '%DS')
    df['reynolds'] = Re_number
    re_conditions, re_values = quartile_generator(df, 'reynolds')

    df['flowratelabel'] = np.select(q_conditions, q_values)
    df['%DSlabel'] = np.select(ds_conditions, ds_values)
    df['reynolds'] = np.select(re_conditions, re_values)

    # Create the interactions plot of interest
    fig1, ax = plt.subplots(figsize=(6, 6))
    fig1 = interaction_plot(
    x=df['reynolds'],
    trace=df['%DSlabel'],
    response=df['FFR'],
    colors=["blue", "red", "green"],
    ms=10,
    ax=ax,
    xlabel='Reynolds Number',
    ylabel='FFR',
    legendtitle='%DS')
    
    # CREATE INPUT AND TARGET ARRAYS FOR REGRESSION 
    X = df.drop(["FFR", "Diagnosis"], axis=1) 
    print(X.corr())
    y = pd.DataFrame()
    y["FFR"] = df['FFR']
    y["binary"] = df['Diagnosis']

    # Box Plots of high and low FFR groups
    df['FFR_Label'] = np.where(df['Diagnosis']==1, r'FFR $\leq$ 0.8', 'FFR > 0.8')  
    fig, axes = plt.subplots(figsize=(20,10), nrows=1, ncols=2)
    axes[0].tick_params(axis='both', labelsize=20)
    axes[1].tick_params(axis='both', labelsize=20)
    sns.boxplot(df['FFR_Label'], df['%DS'], ax=axes[0], showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"white", "markersize":"10"})
    sns.boxplot(df['FFR_Label'], df['Flowrate'], ax=axes[1], showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"white", "markersize":"10"})
    axes[0].set_xlabel('')
    axes[1].set_xlabel('')
    axes[0].set_ylabel('%DS', fontsize=22)
    axes[1].set_ylabel('Coronary Flow Rate (ml/s)', fontsize=22)

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
    
    volume_roc.ROC.ax_.set_xlabel('1 - Specificity', fontsize=18)
    volume_roc.ROC.ax_.set_ylabel('Sensitivity', fontsize=18)
    volume_roc.ROC.ax_.tick_params(axis='both', labelsize=18)
    plt.plot([0, 1], [0, 1], c='black', linestyle='dashed')
    plt.show()
