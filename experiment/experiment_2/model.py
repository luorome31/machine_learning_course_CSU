import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
class BayesianClassifier:
    def __init__(self):
        self.w = None
        self.b = None


    def fit(self,X_train,y_train):
        """
        训练模型
        :param X_train: 训练数据
        :param y_train: 训练标签
        """
        class_0 = X_train[y_train == 0]
        class_1 = X_train[y_train == 1]
        mean_0 = np.mean(class_0, axis=0)
        mean_1 = np.mean(class_1, axis=0)
        cov_0 = np.cov(class_0.T)
        cov_1 = np.cov(class_1.T)
        cov = (cov_0 * len(class_0) + cov_1 * len(class_1)) / (len(class_0) + len(class_1))

        self.w = np.dot((mean_0 - mean_1), np.linalg.inv(cov))
        self.b = -0.5 * np.dot(np.dot(mean_0, np.linalg.inv(cov)), mean_0) + 0.5 * np.dot(np.dot(mean_1, np.linalg.inv(cov)), mean_1) + np.log(len(class_0) / len(class_1))

    def predict(self,X):
        """
        预测
        :param X: 测试数据
        :return: 预测结果
        """
        y_pred = np.dot(X, self.w) + self.b
        y_pred =  1 / (1 + np.exp(-y_pred))
        return np.clip(y_pred, 1e-8, 1 - 1e-8)



class BayesianClassifierEvaluation:
    def __init__(self, logits, labels, preferred_threshold=0.5):
        """
        初始化分类评估对象
        :param logits: 模型预测的概率值
        :param labels: 真实标签
        :param preferred_threshold: 默认的阈值用于分类
        """
        self.logits = logits
        self.labels = labels
        self.preferred_threshold = preferred_threshold
        self.accuracies = []
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
        self.thresholds = []
        self.conf_matrix = None
        self.fpr, self.tpr = None, None

    def calc_metrics(self):
        """
        计算不同阈值下的指标并绘制ROC曲线
        """
        # 生成一系列阈值
        self.thresholds = np.linspace(0, 1, 100)
        self.thresholds = np.insert(self.thresholds, 1, 1e-9)  # 插入一个非常小的阈值，避免所有点分类为1

        # 遍历每个阈值并计算相关指标
        for threshold in self.thresholds:
            preds = (self.logits < threshold).astype(int)
            self.accuracies.append(accuracy_score(self.labels, preds))
            self.precisions.append(precision_score(self.labels, preds, zero_division=0))
            self.recalls.append(recall_score(self.labels, preds))
            self.f1_scores.append(f1_score(self.labels, preds))

        # ROC曲线计算
        self.fpr, self.tpr, _ = roc_curve(self.labels, self.logits)

        # 使用首选阈值计算混淆矩阵
        self.p_threshold_preds = (self.logits < self.preferred_threshold).astype(int)
        self.conf_matrix = self.get_conf_matrix(self.labels, self.p_threshold_preds)

        # 从混淆矩阵获取最终的准确率、精确率、召回率、F1分数
        self.accuracy, self.precision, self.recall, self.f1_score = self.get_metrics_by_conf(self.conf_matrix)

    @staticmethod
    def get_conf_matrix(true_labels, predicted_labels):
        """
        计算混淆矩阵
        :param true_labels: 真实标签
        :param predicted_labels: 预测标签
        :return: 混淆矩阵
        """
        return confusion_matrix(true_labels, predicted_labels)

    @staticmethod
    def get_metrics_by_conf(conf_matrix):
        """
        从混淆矩阵中提取分类指标
        :param conf_matrix: 混淆矩阵
        :return: accuracy, precision, recall, f1_score
        """
        tn, fp, fn, tp = conf_matrix.ravel()
        accuracy = (tp + tn) / (tp + fn + fp + tn)
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        return accuracy, precision, recall, f1_score

    def plot_metrics_vs_threshold(self):
        """
        绘制各个分类指标随阈值变化的曲线
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.thresholds, self.accuracies, label="Accuracy", marker='o')
        plt.plot(self.thresholds, self.precisions, label="Precision", marker='o')
        plt.plot(self.thresholds, self.recalls, label="Recall", marker='o')
        plt.plot(self.thresholds, self.f1_scores, label="F1 Score", marker='o')
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("Metrics vs Threshold")
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()

    def plot_roc_curve(self):
        """
        绘制ROC曲线
        """
        plt.figure(figsize=(6, 6))
        plt.plot(self.fpr, self.tpr, color='blue', label="ROC Curve")
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label="Random Guess")
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title("ROC Curve")
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()

    def plot_confusion_matrix(self):
        """
        绘制混淆矩阵
        """
        plt.figure(figsize=(6, 4))
        sns.heatmap(self.conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix at Threshold {self.preferred_threshold}")
        plt.show()

    @staticmethod
    def get_metrics_str(accuracy, precision, recall, f1_score):
        """
        格式化输出分类指标
        """
        return (f'Accuracy: {accuracy:.3f}     Precision: {precision:.3f}\n'
                f'Recall: {recall:.3f}       F1 Score: {f1_score:.3f}')