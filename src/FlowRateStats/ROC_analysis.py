from sklearn.metrics import RocCurveDisplay, roc_curve, auc, confusion_matrix
import statsmodels.api as sm
import numpy as np

class roc_curve_constructor():
    def __init__(self, clf, X, y_true, y_predict, ax, name):
        self.clf = clf
        self.y_true = y_true
        self.y_predict = y_predict
        self.X = X
        self.input_ax = ax
        self.name = name
        self.gmean = None
        self.best_thresh_index = None
        if self.y_predict is not None:
            self.fpr, self.tpr, self.thresholds = roc_curve(self.y_true, self.y_predict)
            self.roc_auc = auc(self.fpr, self.tpr)
        else:
            self.y_predict = self.clf.predict(sm.add_constant(self.X))
            self.fpr, self.tpr, self.thresholds = roc_curve(self.y_true, self.y_predict)
            self.roc_auc = auc(self.fpr, self.tpr)
        self.ROC = RocCurveDisplay.from_predictions(y_true=self.y_true, y_pred=self.y_predict, ax=self.input_ax,
                                                    name=self.name)


    def optimal_threshold(self):
        self.gmean = np.sqrt(self.tpr * (1 - self.fpr))
        self.best_thresh_index = np.where(self.gmean == self.gmean.max())


class ROCfromScratch():
    def __init__(self, threshold_list, y_true, classifier):
        self.thresh_list = threshold_list
        self.classifier = classifier
        self.y_true = y_true
        self.tpr_list = []
        self.fpr_list = []
        self.auc = None

    def check_metric_value(self, threshold):
        y_pred = []
        for patient in self.classifier:
            if patient > threshold:
                diagnosis = 1
            else:
                diagnosis = 0
            y_pred.append(diagnosis)
        return y_pred


    def roc_constructor(self):
        for threshold in self.thresh_list:
            y_pred = self.check_metric_value(threshold=threshold)
            tn, fp, fn, tp = confusion_matrix(y_true=self.y_true, y_pred=y_pred).ravel()
            fpr = fp/(fp+tn)
            tpr = tp/(tp+fn)
            self.tpr_list.append(tpr)
            self.fpr_list.append(fpr)
        self.auc = auc(x=self.fpr_list, y=self.tpr_list)


