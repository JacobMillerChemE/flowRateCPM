from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, brier_score_loss
from sklearn.calibration import CalibrationDisplay

class logisticRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def multicollinearity_check(self):
        vif_data = pd.DataFrame()
        vif_data["feature"] = self.X.columns
        vif_data["VIF"] = [variance_inflation_factor(self.X.values, i) for i in range(len(self.X.columns))]
        return vif_data

    def logistic_regression_clf(self):
        self.fitted_clf = sm.Logit(self.y, sm.add_constant(self.X)).fit()
        self.probs = self.fitted_clf.predict(sm.add_constant(self.X))
        self.predictions = list(map(round, self.fitted_clf.predict(sm.add_constant(self.X))))
        return self.fitted_clf, self.probs, self.predictions

    def log_odds_linearity_check(self, probs):
        for column in self.X:
            plt.scatter(self.X[column], probs)
            plt.xlabel(column)
            plt.ylabel("log odds")
            plt.show()

    def calibrationCheck(self):
        model_label = ""
        for i in self.X.columns:
            model_label += str(i)
            model_label += ", "
        print("\n")
        print("{} Brier Score: {}".format(model_label, brier_score_loss(y_true=self.y, y_prob=self.probs)))
        CalibrationDisplay.from_predictions(y_true=self.y, y_prob=self.probs, name="{}".format(model_label))
        plt.show()

    def results(self):
        self.summary = self.fitted_clf.summary()
        self.accuracy = accuracy_score(y_true=self.y, y_pred=self.predictions)
        self.confusion_matrix = confusion_matrix(y_true=self.y, y_pred=self.predictions)
