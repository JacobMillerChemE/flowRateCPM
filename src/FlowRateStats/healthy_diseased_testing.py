import numpy as np
import pandas as pd
from ROC_analysis import roc_curve_constructor
from sm_logistic_regression import logisticRegression
import matplotlib.pyplot as plt
import os
import seaborn as sns
from statsmodels.graphics.factorplots import interaction_plot
from statistical_tests import statTests

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

    # Making healthy and unhealthy groups
    healthy = df[df['FFR'] > 0.8]
    unhealthy = df[df['FFR'] <= 0.8]
    
    # Make dictionary with groups
    groups = {'healthy': healthy, 'unhealthy': unhealthy}

    # Find means of %DS
    print(healthy['%DS'].mean())
    print(unhealthy['%DS'].mean())
    
    # Find means of coronary flow rate
    print(healthy['Flowrate'].mean())
    print(unhealthy['Flowrate'].mean())

    # Make testing object
    testing = statTests(groups=groups, df=df)

    # Test for normality for %Ds and do appropraite test
    testing.normality_check('%DS')
    testing.t_test('%DS')

    # Test for normality for  and do appropraite test
    testing.normality_check('Flowrate')
    testing.mannwhitney_test('Flowrate')
