import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

def get_data_path():
    dirname = os.path.dirname(__file__)
    data_path = dirname.replace('/src/FlowRateStats', '/dataFrame/Results3.csv')
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
    df["%DS"] = (1 - np.sqrt(4 * df["Stenosis Area"] / np.pi) / np.sqrt(4 * df["Mean Area"] / np.pi)) * 100
    df = df.drop(["Mean Area", "length", "Nominal", "Stenosis Position", "Beta"], axis=1)

    # SCATTER PLOT OF FFR VS CFR COLORED BY %DS

    fig, ax = plt.subplots()
    scatter = ax.scatter(df['Flowrate'], df['FFR'], c=df['%DS'], cmap='Reds')
    ax.set_xlabel('Coronary Flow Rate')
    ax.set_ylabel('FFR')
    fig.colorbar(scatter, label='%DS')
    plt.show()
