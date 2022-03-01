from scipy import stats
from pingouin import ancova
from statsmodels.formula.api import glm
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

class statTests:
    def __init__(self, groups, df):
        self.df = df
        self.data = groups
        self.parametric = None
        self.keys = list(self.data.keys())

    def normality_check(self, column_name):
        for key, dataframe in self.data.items():
            self.normality_stat, self.normality_p = stats.shapiro(dataframe[column_name])
            if self.normality_p > 0.05:
                print(f"{key} is normally distributed.")
                self.parametric = True
            else:
                print(f"{key} is NOT normally distributed.")
                self.parametric = False
            print(f"Shapiro Statistic: {self.normality_stat}")
            print(f"Shapiro significance: {self.normality_p}\n")

    def levene_variance_check(self, column_name):
        levene_stat, levene_p = stats.levene(self.data[self.keys[0]][column_name], self.data[self.keys[1]][column_name])
        print(f"Levene p-value: {levene_p}")

    def mannwhitney_test(self, column_name):
        mann_stat, mann_p = stats.mannwhitneyu(self.data[self.keys[0]][column_name], self.data[self.keys[1]][column_name])
        print(f"Mann-Whitney U Test p-value for {column_name}: {mann_p}")

    def kruskal_wallis_test(self):
        krusk_stat, krusk_p = stats.kruskal()

    def t_test(self, dv_name):
        print(f"{dv_name} T-Test Results: {stats.ttest_ind(self.data[self.keys[0]][dv_name], self.data[self.keys[1]][dv_name])}")

    def ancova_test(self, df, between, covariate, dv):
        plt.scatter(self.data[self.keys[0]][covariate], self.data[self.keys[0]][dv])
        plt.scatter(self.data[self.keys[1]][covariate], self.data[self.keys[1]][dv])
        m_healthy, b_healthy = np.polyfit(self.data[self.keys[0]][covariate], self.data[self.keys[0]][dv], deg=1)
        m_sick, b_sick = np.polyfit(self.data[self.keys[1]][covariate], self.data[self.keys[1]][dv], deg=1)
        plt.plot(self.data[self.keys[0]][covariate], m_healthy * self.data[self.keys[0]][covariate] + b_healthy)
        plt.plot(self.data[self.keys[1]][covariate], m_sick * self.data[self.keys[1]][covariate] + b_sick)
        GLM = glm("Nominal ~ Diagnosis: percDS", data=self.df)
        GLM_results = GLM.fit()
        print(GLM_results.summary())
        print(ancova(data=df, dv=dv, covar=covariate, between=between))
