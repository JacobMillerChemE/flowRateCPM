import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress

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
    df = df.dropna()
    df = df[(df["Flowrate"] > 1) & (df['Flowrate'] < 2)]

    # FEATURE ENGINEERING
    df["%DS"] = (1 - np.sqrt(4 * df["Stenosis Area"] / np.pi) /
                 np.sqrt(4 * df["Mean Area"] / np.pi)) * 100
    df = df.drop(["Mean Area", "length", "Nominal", "Stenosis Position",
                 "Beta"], axis=1)

    print(df.sort_values(by=['FFR']))
