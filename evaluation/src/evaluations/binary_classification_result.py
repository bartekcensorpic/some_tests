from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np


class BinaryClassificationResult:
    def __init__(
        self,
        precision,
        recall,
        thresholds,
        f1s,
        ths,
        p_r_t_axis,
        cc_axis,
        cf_axis,
        f1_ths_ax,
    ):

        self._cf_axis = cf_axis
        self._cc_axis = cc_axis
        self._p_r_t_axis = p_r_t_axis
        self._f1_ths_ax = f1_ths_ax
        self._ths = ths
        self._f1s = f1s
        self._thresholds = thresholds
        self._recall = recall
        self._precision = precision

    def get_precision_recall_thresholds(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns precision, recall and thresholds

        :return: Tuple of 3 list: precision, recall, thresholds
        """
        return (self._precision, self._recall, self._thresholds)

    def get_precision_recall_threshold_plot(self) -> plt.axis:
        """
        Returns plot of precision, recall and thresholds

        :return: matplotlib.axis
        """
        return self._p_r_t_axis

    def get_calibration_curve(self) -> plt.axis:
        """
        Returns calibration curve

        :return: matpllib.axis
        """

        return self._cc_axis

    def get_confusion_matrix(self) -> plt.axis:
        """
        Returns confusion matrix

        :return: matpllib.axis
        """

        return self._cf_axis

    def get_f1_thresholds_and_plot(self) -> Tuple[np.ndarray, np.ndarray, plt.axis]:
        """
        Return plot of F1 scores vs Thresholds

        :return: Tuple of
        """
        return self._f1s, self._ths, self._f1_ths_ax
