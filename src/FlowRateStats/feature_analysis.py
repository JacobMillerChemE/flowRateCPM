import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro, pearsonr

class FeatureOverview():
    def __init__(self, feature):
        self.data = feature
        self.descriptives = str(feature.name) + "\n" + str(feature.describe()) + "\n"
        self.name = feature.name

    def show_histogram(self):
        sns.histplot(self.data)
        plt.show()

    def show_qqplot(self):
        qqplot(self.data, line='s')
        plt.show()

    def basic_plots(self):
        fig, axs = plt.subplots(1, 2)
        qqplot(self.data, line='s', ax=axs[0])
        sns.histplot(self.data, ax=axs[1])
        fig.suptitle(f"Plots for {self.name}")
        plt.show()

    def normality_check(self):
        self.normality_stat, self.normality_p = shapiro(self.data)
        if self.normality_p > 0.05:
            print(f"{self.name} is normally distributed.")
        else:
            print(f"{self.name} is NOT normally distributed.")
        print(f"Shapiro Statistic: {self.normality_stat}")
        print(f"Shapiro significance: {self.normality_p}\n")

    def target_scatter_plot(self, targetvar):
        rpearson, pvalue = pearsonr(self.data, targetvar)
        plt.scatter(self.data, targetvar)
        plt.suptitle("{} vs {} (r = {:.3f}, p = {:.3f})".format(targetvar.name, self.name, rpearson, pvalue))
        plt.ylim(0, 1)
        plt.xlabel(self.name)
        plt.ylabel(targetvar.name)
        plt.show()


def feature_list_visualizer(feature_list, target_variable):
    for feature in feature_list.values():
        feature.basic_plots()
        feature.target_scatter_plot(target_variable)
        print(feature.descriptives)