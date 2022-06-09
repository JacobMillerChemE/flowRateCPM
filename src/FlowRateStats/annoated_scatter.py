import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress

def get_data_path():
    dirname = os.path.dirname(__file__)
    data_path = dirname.replace('/src/FlowRateStats', '/dataFrame/Results4.csv')
    return data_path


def quartile_generator(data, feature):
    '''
    Determines quartiles and returns the conditions and values to create 
    a pandas column labeled according to the list values that is used for 
    the interaction plot
    '''
    quartiles = np.percentile(data[feature], [25, 75])
    conditions = [(df[feature] <= quartiles[0]), (df[feature] >= quartiles[1])]
    values = ('Low', 'High')
    print(f'{feature} quartiles: {quartiles}')
    return conditions, values


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
    df["%DS"] = (1 - np.sqrt(4 * df["Stenosis Area"] / np.pi) /
                 np.sqrt(4 * df["Max Area"] / np.pi)) * 100
    print(df['%DS'])
    df = df.drop(["Mean Area", "length", "Nominal", "Stenosis Position",
                 "Beta"], axis=1)

    # CREATE MINIMAL AND MODERATE DATAFRAMES
    minimal = df[df['%DS'] < 40]
    moderate = df[df['%DS'] >= 60]

    # PRINT DATA POINTS WITH LOW STENOSES
    print(df[df['%DS'] <= 45])

    # FIT LINES TO MINIMAL AND MODERATE PLOTS
    minimal_line = linregress(x=minimal['Flowrate'], y=minimal['FFR'])
    moderate_line = linregress(x=moderate['Flowrate'], y=moderate['FFR'])

    # DATA LABELING PATIENTS A AND B
    patient_A = df.loc['128 mid LAD']
    print(patient_A)
    patient_B = df.loc['086 prox LAD']
    print(patient_B)

    # SCATTER PLOT OF FFR VS CFR COLORED BY %DS
    fig, ax = plt.subplots()
    ax.set_xlabel('Coronary Flow Rate')
    ax.set_ylabel('FFR')
    ax.plot([0, df['Flowrate'].max()], [0.8, 0.8], c='k', linestyle='dashed')
    ax.plot(df['Flowrate'], minimal_line[0]*df['Flowrate']+minimal_line[1], c='b', label='< 40%')
    moderate_xvalues = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3])
    ax.plot(moderate_xvalues, moderate_line[0]*moderate_xvalues+moderate_line[1], c='r', label=r'$\geq$60%')
    ax.annotate('Patient A', xy=(patient_A['Flowrate'], patient_A['FFR']), 
            xytext=(patient_A['Flowrate']+0.25, patient_A['FFR']+0.05), 
            fontsize=12, arrowprops=dict(facecolor='green', shrink=0.05))
    ax.annotate('Patient B', xy=(patient_B['Flowrate'], patient_B['FFR']), 
            xytext=(patient_B['Flowrate']-0.25, patient_B['FFR']-0.05), 
            fontsize=12, arrowprops=dict(facecolor='green', shrink=0.05))
    scatter = ax.scatter(df['Flowrate'], df['FFR'], c=df['%DS'], cmap='seismic')
    fig.colorbar(scatter, label='%DS')
    fig.legend()
    plt.show()
