# coding:utf-8
import os

from PyQt5.QtGui import QIcon, QTransform, QFont
from PyQt5.QtWidgets import  QFrame, QStackedWidget, QGridLayout,\
    QGraphicsPixmapItem, QFileDialog
from qfluentwidgets import FluentIcon as FIF, PushButton, TableWidget, TitleLabel,LineEdit,LargeTitleLabel, Dialog
from qfluentwidgets import (NavigationInterface, NavigationItemPosition,
                            isDarkTheme)
from qfluentwidgets import (SpinBox,  DoubleSpinBox)
from qframelesswindow import FramelessWindow, StandardTitleBar
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,QLineEdit, QSizePolicy
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, QObject, QUrl, QTimer

import pandas as pd

from PyQt5.QtWidgets import  QTableWidgetItem
from PyQt5.QtCore import Qt
from graphviz import Digraph
import pyqtgraph as pg

y_min_global = None
y_max_global = None
x_min_global = None
x_max_global = None
# 初始化数据
network_global = None
input_size_global = 1
output_size_global = 1
hidden_size_global = 3
epochs_global = 5000
learning_rate_global = 0.015
test_losses_global = None
x_train_global = None
y_train_global = None
x_test_global = None
y_test_global = None
recall_losses_global = None
train_losses_global = None
output_test_global = None


# 定义神经网络结构
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 初始化权重和偏置
        np.random.seed(42)
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.biases_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.biases_output = np.zeros((1, output_size))

    # 定义激活函数（sigmoid）
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # 定义激活函数的导数
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # 前向传播
    def forward_propagation(self, inputs):
        hidden_output = self.sigmoid(np.dot(inputs, self.weights_input_hidden) + self.biases_hidden)
        output = self.sigmoid(np.dot(hidden_output, self.weights_hidden_output) + self.biases_output)
        return output, hidden_output

    # 定义损失函数（均方误差）
    def mse_loss(self, predicted, target):
        return np.mean((predicted - target) ** 2)

    # 定义损失函数的导数
    def mse_loss_derivative(self, predicted, target):
        return 2 * (predicted - target)

    # 反向传播
    def backward_propagation(self, inputs, hidden_output, output, target):
        output_error = self.mse_loss_derivative(output, target)
        output_delta = output_error * self.sigmoid_derivative(output)

        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(hidden_output)

        return output_delta, hidden_delta

    # 更新权重和偏置
    def update_weights(self, inputs, hidden_output, output_delta, hidden_delta, learning_rate):
        self.weights_hidden_output -= np.dot(hidden_output.T, output_delta) * learning_rate
        self.biases_output -= np.sum(output_delta, axis=0, keepdims=True) * learning_rate

        self.weights_input_hidden -= np.dot(inputs.T, hidden_delta) * learning_rate
        self.biases_hidden -= np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def visualize_network(self):
        graph = Digraph()
        for i in range(self.input_size):
            input_node_name = f'Input {i + 1}'
            graph.node(input_node_name)

        for i in range(self.hidden_size):
            hidden_node_name = f'Hidden {i + 1}'
            graph.node(hidden_node_name)
            for j in range(self.input_size):
                input_node_name = f'Input {j + 1}'
                graph.edge(input_node_name, hidden_node_name)

        for i in range(self.output_size):
            output_node_name = f'Output {i + 1}'
            graph.node(output_node_name)
            for j in range(self.hidden_size):
                hidden_node_name = f'Hidden {j + 1}'
                graph.edge(hidden_node_name, output_node_name)

        # 设置节点内部的标签位置为默认位置
        graph.attr('node', labelloc='c')

        return graph



def normalize_input_data(input_data):
    # 对输入特征进行归一化处理
    x_min = np.min(input_data, axis=0)
    x_max = np.max(input_data, axis=0)
    x_normalized = (input_data - x_min) / (x_max - x_min)
    return x_normalized, x_min, x_max


def normalize_output_data(output_data):
    # 对目标变量进行归一化处理
    y_min = np.min(output_data)
    y_max = np.max(output_data)
    y_normalized = (output_data - y_min) / (y_max - y_min)
    return y_normalized, y_min, y_max


def inverse_normalize_data(data_normalized, min_val, max_val):
    # 还原归一化后的数据到原始范围
    data = data_normalized * (max_val - min_val) + min_val
    return data


def inverse_normalize_output(output_normalized, y_min, y_max):
    # 反归一化输出数据
    output = inverse_normalize_data(output_normalized, y_min, y_max)
    return output


def inverse_normalize_input(input_normalized, x_min, x_max):
    # 反归一化输入数据
    input_data = inverse_normalize_data(input_normalized, x_min, x_max)
    return input_data

def read_data_from_csv_train(file_path, input_size, output_size):
    # 跳过表头
    data = pd.read_csv(file_path, skiprows=1)

    # 提取输入特征
    x_train = data.iloc[:, :input_size].values
    # 提取目标变量
    y_train = data.iloc[:, -output_size:].values

    # 计算最大值和最小值
    x_min = np.min(x_train, axis=0)
    x_max = np.max(x_train, axis=0)
    y_min = np.min(y_train)
    y_max = np.max(y_train)

    # 归一化输入特征
    x_train_normalized = (x_train - x_min) / (x_max - x_min)

    # 归一化目标变量
    y_train_normalized = (y_train - y_min) / (y_max - y_min)

    return x_train_normalized, y_train_normalized, x_min, x_max, y_min, y_max

# 生成测试数据
def read_data_from_csv_test(file_path, input_size, output_size, x_min, x_max, y_min, y_max):
    # 跳过表头
    data = pd.read_csv(file_path, skiprows=1)

    # 提取输入特征
    x_test = data.iloc[:, :input_size].values
    # 提取目标变量
    y_test = data.iloc[:, -output_size:].values

    # 归一化输入特征
    x_test_normalized = (x_test - x_min) / (x_max - x_min)

    # 归一化目标变量
    y_test_normalized = (y_test - y_min) / (y_max - y_min)

    return x_test_normalized, y_test_normalized
# 无归一化
def read_data_from_csv(file_path, input_size, output_size):
    # 跳过表头
    data = pd.read_csv(file_path, skiprows=1)

    # 提取输入特征
    x_train = data.iloc[:, :input_size].values
    # 提取目标变量
    y_train = data.iloc[:, -output_size:].values

    return x_train, y_train

def train_neural_network(network, x_train, y_train, learning_rate, epochs):

    losses = []
    for epoch in range(epochs):
        # 前向传播
        output, hidden_output = network.forward_propagation(x_train)

        # 计算损失
        loss = network.mse_loss(output, y_train)
        # 打印损失，保留五位小数
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.5f}')
        losses.append(loss)

        # 反向传播
        output_delta, hidden_delta = network.backward_propagation(x_train, hidden_output, output, y_train)

        # 更新权重和偏置
        network.update_weights(x_train, hidden_output, output_delta, hidden_delta, learning_rate)
    return losses


def show_dataset(self, file_path):
    try:
        data = pd.read_csv(file_path)
        # 检查数据是否为空
        if data.empty:
            self.msg = Dialog("Empty Dataset","数据集为空，请检查文件内容。")
            self.msg.cancelButton.hide()
            self.msg.buttonLayout.insertStretch(1)
            self.msg.exec_()
            return None
    except FileNotFoundError:
        self.msg = Dialog("File Not Found", "文件未找到，请检查文件路径是否正确。")
        self.msg.cancelButton.hide()
        self.msg.buttonLayout.insertStretch(1)
        self.msg.exec_()
        return None
    except Exception as e:
        self.msg = Dialog("Error", f"发生错误：{str(e)}")
        self.msg.cancelButton.hide()
        self.msg.buttonLayout.insertStretch(1)
        self.msg.exec_()
        return None
    return data

 # 可视化神经网络结构


class Widget(QFrame):

    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        self.label = QLabel(text, self)
        self.label.setAlignment(Qt.AlignCenter)
        self.hBoxLayout = QHBoxLayout(self)
        self.hBoxLayout.addWidget(self.label, 1, Qt.AlignCenter)
        self.setObjectName(text.replace(' ', '-'))


class data_traintable(QWidget):

    def __init__(self, text: str, parent=None, start_widget_instance=None):
        super().__init__()
        self.file_path = 'training_data.csv'
        self.data_widget_instance = start_widget_instance
        self.data_widget_instance.communicate.updateData.connect(self.update)

        self.setObjectName(text.replace(' ', '-'))

        self.mainLayout = QVBoxLayout()

        self.titleLayout = QHBoxLayout()
        # 输出标签train
        self.label = TitleLabel("train")
        self.label.setAlignment(Qt.AlignCenter)

        self.titleLayout.addWidget(self.label, alignment=Qt.AlignCenter)

        self.tableLayout = QHBoxLayout()
        self.tableView = TableWidget()
        self.tableView.setBorderVisible(True)
        self.tableView.setBorderRadius(8)
        self.tableView.setWordWrap(False)
        self.tableView.setMinimumWidth(800)
        self.tableView.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tableLayout.addWidget(self.tableView, alignment=Qt.AlignHCenter)

        self.mainLayout.addLayout(self.titleLayout)
        self.mainLayout.addLayout(self.tableLayout)

        self.setLayout(self.mainLayout)

    def update(self, file_path):
        self.file_path = file_path
        self.load_data(file_path)

    def load_data(self, file_path):
        data = show_dataset(self, file_path)
        if data is not None:
            self.tableView.clear()  # 清空表格
            self.tableView.setRowCount(len(data))
            self.tableView.setColumnCount(len(data.columns))
            self.tableView.setHorizontalHeaderLabels(data.columns)
            for i in range(len(data)):
                for j in range(len(data.columns)):
                    item = QTableWidgetItem(str(data.iloc[i, j]))
                    self.tableView.setItem(i, j, item)
                    # self.tableView.resizeColumnsToContents()  # 调整列宽度为内容的最佳宽度
            self.tableView.resizeColumnsToContents()  # 调整列宽度为内容的最佳宽度
            self.set_max_column_width(200)  # 设置列的最大宽度为200像素


    def set_max_column_width(self, max_width):
        for column in range(self.tableView.columnCount()):
            width = self.tableView.columnWidth(column)
            if width > max_width:
                self.tableView.setColumnWidth(column, max_width)

class data_testtable(QWidget):

    def __init__(self, text: str, parent=None, start_widget_instance=None):
        super().__init__()
        self.file_path = 'training_data_test.csv'
        self.data_widget_instance = start_widget_instance
        self.data_widget_instance.communicate2.updateData.connect(self.update)

        self.setObjectName(text.replace(' ', '-'))

        self.mainLayout = QVBoxLayout()

        self.titleLayout = QHBoxLayout()
        # 输出标签train
        self.label = TitleLabel("test")
        self.label.setAlignment(Qt.AlignCenter)

        self.titleLayout.addWidget(self.label, alignment=Qt.AlignCenter)

        self.tableLayout = QHBoxLayout()
        self.tableView = TableWidget()
        self.tableView.setBorderVisible(True)
        self.tableView.setBorderRadius(8)
        self.tableView.setWordWrap(False)
        self.tableView.setMinimumWidth(800)
        self.tableView.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tableLayout.addWidget(self.tableView, alignment=Qt.AlignHCenter)

        self.mainLayout.addLayout(self.titleLayout)
        self.mainLayout.addLayout(self.tableLayout)

        self.setLayout(self.mainLayout)

    def update(self, file_path):
        self.file_path = file_path
        self.load_data(file_path)

    def load_data(self, file_path):
        data = show_dataset(self, file_path)
        if data is not None:
            self.tableView.clear()  # 清空表格
            self.tableView.setRowCount(len(data))
            self.tableView.setColumnCount(len(data.columns))
            self.tableView.setHorizontalHeaderLabels(data.columns)
            for i in range(len(data)):
                for j in range(len(data.columns)):
                    item = QTableWidgetItem(str(data.iloc[i, j]))
                    self.tableView.setItem(i, j, item)
            self.tableView.resizeColumnsToContents()

class SettingCommunicate(QObject):
    updateSettings = pyqtSignal(int, int, int, int, float)

class dataCommunicate(QObject):
    updateData = pyqtSignal(str)

class setting_widget(QFrame):
    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        self.setObjectName(text.replace(' ', '-'))
        # 添加输入框
        self.inputSizeEdit = SpinBox()
        self.outputSizeEdit = SpinBox()
        self.hiddenSizeEdit = SpinBox()
        self.epochsEdit = SpinBox()
        self.learningRateEdit = DoubleSpinBox()
        self.learningRateEdit.setDecimals(3)  # 设置显示小数位数为3位
        self.communicate = SettingCommunicate()


        # 设置范围
        self.inputSizeEdit.setRange(1, 100)
        self.outputSizeEdit.setRange(1, 100)
        self.hiddenSizeEdit.setRange(1, 100)
        self.epochsEdit.setRange(1, 10000)
        self.epochsEdit.setSingleStep(50)
        self.learningRateEdit.setMinimum(0.001)
        self.learningRateEdit.setMaximum(1.0)
        self.learningRateEdit.setSingleStep(0.001)

        # 设置默认值
        self.inputSizeEdit.setValue(1)
        self.outputSizeEdit.setValue(1)
        self.hiddenSizeEdit.setValue(3)
        self.epochsEdit.setValue(5000)
        self.learningRateEdit.setValue(0.015)

        # 添加标签
        self.inputSizeLabel = QLabel("Input Size:", self)
        # 监听数值改变信号
        self.inputSizeEdit.valueChanged.connect(lambda value: print("input当前值：", value))
        print(self.inputSizeEdit.value())
        self.inputSizeEdit.setValue(input_size_global)
        self.outputSizeLabel = QLabel("Output Size:", self)
        # 监听数值改变信号
        self.outputSizeEdit.valueChanged.connect(lambda value: print("output当前值：", value))
        self.outputSizeEdit.setValue(output_size_global)
        print(self.outputSizeEdit.value())


        self.hiddenSizeLabel = QLabel("Hidden Size:", self)
        # 监听数值改变信号
        self.hiddenSizeEdit.valueChanged.connect(lambda value: print("hidden当前值：", value))
        self.hiddenSizeEdit.setValue(hidden_size_global)
        print(self.hiddenSizeEdit.value())

        self.epochsLabel = QLabel("Epochs:", self)
        # 监听数值改变信号
        self.epochsEdit.valueChanged.connect(lambda value: print("epochs当前值：", value))
        self.epochsEdit.setValue(epochs_global)
        print(self.epochsEdit.value())

        self.learningRateLabel = QLabel("Learning Rate:", self)
        # 监听数值改变信号
        self.learningRateEdit.valueChanged.connect(lambda value: print("learningrate当前值：", value))
        self.learningRateEdit.setValue(learning_rate_global)
        print("{:.3f}".format(self.learningRateEdit.value()))

        # 添加确认按钮
        self.confirmButton = PushButton("确认")
        self.confirmButton.clicked.connect(self.confirmSettings)

        # 添加布局
        self.gridLayout = QGridLayout()
        self.gridLayout.setContentsMargins(100, 50, 100, 50)
        self.gridLayout.addWidget(self.inputSizeLabel, 0, 0)
        self.gridLayout.addWidget(self.inputSizeEdit, 0, 1)
        self.gridLayout.addWidget(self.outputSizeLabel, 1, 0)
        self.gridLayout.addWidget(self.outputSizeEdit, 1, 1)
        self.gridLayout.addWidget(self.hiddenSizeLabel, 2, 0)
        self.gridLayout.addWidget(self.hiddenSizeEdit, 2, 1)
        self.gridLayout.addWidget(self.epochsLabel, 3, 0)
        self.gridLayout.addWidget(self.epochsEdit, 3, 1)
        self.gridLayout.addWidget(self.learningRateLabel, 4, 0)
        self.gridLayout.addWidget(self.learningRateEdit, 4, 1)
        self.gridLayout.addWidget(self.confirmButton, 5, 1)  # 将确认按钮放在第五行第二列
        self.gridLayout.setSpacing(10)  # 设置控件之间的间距


        self.setLayout(self.gridLayout)



    def confirmSettings(self):
        # 获取当前输入框的值
        input_size1 = self.inputSizeEdit.value()
        output_size1 = self.outputSizeEdit.value()
        hidden_size1 = self.hiddenSizeEdit.value()
        epochs1 = self.epochsEdit.value()
        learning_rate1 = self.learningRateEdit.value()

        # 更新全局变量值
        global input_size_global, output_size_global, hidden_size_global, epochs_global, learning_rate_global
        input_size_global = input_size1
        output_size_global = output_size1
        hidden_size_global = hidden_size1
        epochs_global = epochs1
        learning_rate_global = learning_rate1

        # 发送信号
        self.communicate.updateSettings.emit(input_size_global, output_size_global, hidden_size_global, epochs_global,
                                             learning_rate_global)

# 引入弹窗类
class CustomMessageBox(QFrame):
    onUrlEntered = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("打开 URL")
        self.setMinimumWidth(350)

        self.titleLabel = QLabel("打开 URL")
        self.urlLineEdit = QLineEdit()
        self.urlLineEdit.setPlaceholderText("输入文件、流或者播放列表的 URL")
        self.urlLineEdit.setClearButtonEnabled(True)
        self.urlLineEdit.textChanged.connect(self._validateUrl)

        self.openButton = QPushButton("打开")
        self.openButton.clicked.connect(self._openUrl)
        self.openButton.setDisabled(True)

        self.cancelButton = QPushButton("取消")
        self.cancelButton.clicked.connect(self.close)

        layout = QVBoxLayout()
        layout.addWidget(self.titleLabel)
        layout.addWidget(self.urlLineEdit)
        layout.addWidget(self.openButton)
        layout.addWidget(self.cancelButton)

        self.setLayout(layout)

    def _validateUrl(self, text):
        self.openButton.setEnabled(QUrl(text).isValid())

    def _openUrl(self):
        url = self.urlLineEdit.text()
        self.onUrlEntered.emit(url)
        self.close()


class start_widget(QFrame):
    onTrainDataChanged = pyqtSignal(dict)
    def __init__(self, text: str, parent=None, setting_widget_instance=None):
        super().__init__(parent=parent)
        self.setting_widget_instance = setting_widget_instance
        self.setting_widget_instance.communicate.updateSettings.connect(self.updateLabels)
        self.setObjectName(text.replace(' ', '-'))
        self.communicate = dataCommunicate()
        self.communicate2  = dataCommunicate()
        # 添加标签，占据单独一行
        self.label = LargeTitleLabel("神经网络")
        #蓝色
        self.label.setStyleSheet("color: #0078d4;")
        #换个字体，圆润一点的
        self.label.setFont(QFont("Microsoft YaHei UI", 36, QFont.Bold))
        self.label.setAlignment(Qt.AlignCenter)

        self.inputSizeLabel = LineEdit()
        self.inputSizeLabel.setPlaceholderText("输入:")
        self.inputSizeLabel.setReadOnly(True)

        self.inputSizeEdit = LineEdit()
        self.inputSizeEdit.setPlaceholderText(str(input_size_global))
        self.inputSizeEdit.setReadOnly(True)

        self.outputSizeLabel = LineEdit()
        self.outputSizeLabel.setPlaceholderText("输出:")
        self.outputSizeLabel.setReadOnly(True)

        self.outputSizeEdit = LineEdit()
        self.outputSizeEdit.setPlaceholderText(str(output_size_global))
        self.outputSizeEdit.setReadOnly(True)

        self.hiddenSizeLabel = LineEdit()
        self.hiddenSizeLabel.setPlaceholderText("隐藏层:")
        self.hiddenSizeLabel.setReadOnly(True)

        self.hiddenSizeEdit = LineEdit()
        self.hiddenSizeEdit.setPlaceholderText(str(hidden_size_global))
        self.hiddenSizeEdit.setReadOnly(True)

        self.epochsLabel =  LineEdit()
        self.epochsLabel.setPlaceholderText("迭代次数:")
        self.epochsLabel.setReadOnly(True)

        self.epochsEdit = LineEdit()
        self.epochsEdit.setPlaceholderText(str(epochs_global))
        self.epochsEdit.setReadOnly(True)

        self.learningRateLabel = LineEdit()
        self.learningRateLabel.setPlaceholderText("学习率:")
        self.learningRateLabel.setReadOnly(True)

        self.learningRateEdit = LineEdit()
        self.learningRateEdit.setPlaceholderText(str(learning_rate_global))
        self.learningRateEdit.setReadOnly(True)

        self.trainStatusLabel = LineEdit()
        self.trainStatusLabel.setPlaceholderText("训练状态:")
        self.trainStatusLabel.setReadOnly(True)

        self.trainStatusEdit = LineEdit()
        self.trainStatusEdit.setPlaceholderText("未开始")
        # 红色
        self.trainStatusEdit.setStyleSheet("color: red;")
        self.trainStatusEdit.setReadOnly(True)

        self.trainButton = PushButton("开始训练")
        self.trainButton.clicked.connect(self.setstatus)
        # self.trainButton.clicked.connect(self.train_and_plot)


        self.predictButton = PushButton("预测")
        self.predictButton.clicked.connect(self.predict_output)


        self.manualInputLineEdit = LineEdit()
        self.manualInputLineEdit.setPlaceholderText("请输入数据")
        self.manualInputLineEdit.setFixedWidth(200)

        self.predictionResultLabel = LineEdit()
        self.predictionResultLabel.setPlaceholderText("预测结果:")
        self.predictionResultLabel.setReadOnly(True)

        self.predictionResultEdit = LineEdit()
        self.predictionResultEdit.setPlaceholderText("未预测")
        # 蓝色
        self.predictionResultEdit.setStyleSheet("color: blue;")
        self.predictionResultEdit.setReadOnly(True)

        # 文件输入
        self.url_input_1 = LineEdit()
        self.url_input_1.setPlaceholderText("选择训练数据集")
        self.url_input_1.setFixedWidth(200)
        self.url_input_1.setReadOnly(True)
        self.selectFileButton_1 = PushButton("训练文件")
        self.selectFileButton_1.clicked.connect(self.select_file_1)
        self.url_input_2 = LineEdit()
        self.url_input_2.setPlaceholderText("选择测试数据集")
        self.url_input_2.setFixedWidth(200)
        self.url_input_2.setReadOnly(True)
        self.selectFileButton_2 = PushButton("测试文件")
        self.selectFileButton_2.clicked.connect(self.select_file_2)

        # 添加按钮
        self.button = PushButton("加载数据")
        self.button.clicked.connect(self.load_data_and_show_table)

        # Separate layout for the 神经网络 label
        label_layout = QHBoxLayout()
        label_layout.addWidget(self.label)

        # Rest of your code remains the same...
        top_layout = QVBoxLayout()
        # top_layout.setContentsMargins(10, 80, 10, 80)  # 设置内边距，这里只在左右设置，上下不设置
        # 随文本长度自适应
        self.label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        widgets_row1 = [self.inputSizeLabel, self.inputSizeEdit,
                        self.outputSizeLabel, self.outputSizeEdit,
                        self.hiddenSizeLabel, self.hiddenSizeEdit]

        widgets_row2 = [self.epochsLabel, self.epochsEdit,
                        self.learningRateLabel, self.learningRateEdit]

        # 创建两个水平布局来分别放置 widgets_row1 和 widgets_row2
        h_layout1 = QHBoxLayout()
        for widget in widgets_row1:
            h_layout1.addWidget(widget)

        h_layout2 = QHBoxLayout()
        for widget in widgets_row2:
            h_layout2.addWidget(widget)

        # 将两个水平布局添加到垂直布局中
        top_layout.addLayout(h_layout1)
        top_layout.addLayout(h_layout2)

        self.settingframe = QFrame()
        self.settingframe.setStyleSheet(
            "QFrame { background-color: #f0f0f0; border: 2px solid #aaa; border-radius: 5px; box-shadow: 5px 5px 5px rgba(0, 0, 0, 0.1); }")  # 设置边框样式、背景颜色和阴影效果
        self.settingframe.setLayout(top_layout)

        status_frame = QFrame()
        status_frame.setStyleSheet(
            "QFrame { background-color: #f0f0f0; border: 2px solid #aaa; border-radius: 5px; box-shadow: 5px 5px 5px rgba(0, 0, 0, 0.1); }")
        status_layout = QHBoxLayout(status_frame)
        status_layout.addWidget(self.trainStatusLabel)
        status_layout.addWidget(self.trainStatusEdit)

        action_frame = QFrame()
        action_frame.setStyleSheet(
            "QFrame { background-color: #f0f0f0; border: 2px solid #aaa; border-radius: 5px; box-shadow: 5px 5px 5px rgba(0, 0, 0, 0.1); }")
        action_layout = QHBoxLayout(action_frame)
        action_layout.addWidget(self.trainButton)

        action_layout.addWidget(self.url_input_1)
        action_layout.addWidget(self.selectFileButton_1)
        action_layout.addWidget(self.url_input_2)
        action_layout.addWidget(self.selectFileButton_2)
        action_layout.addWidget(self.button)

        predict_frame = QFrame()
        predict_frame.setStyleSheet(
            "QFrame { background-color: #f0f0f0; border: 2px solid #aaa; border-radius: 5px; box-shadow: 5px 5px 5px rgba(0, 0, 0, 0.1); }")
        # 创建两个水平布局
        h_layout1 = QHBoxLayout()
        h_layout1.addWidget(self.manualInputLineEdit)
        h_layout1.addWidget(self.predictButton)
        h_layout1.setStretchFactor(self.manualInputLineEdit, 1)  # 设置数据输入框的拉伸因子为1，让其占据更多的空间
        h_layout1.setStretchFactor(self.predictButton, 0)  # 设置预测按钮的拉伸因子为0，不让其拉伸


        h_layout2 = QHBoxLayout()
        self.predictionResultLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # 设置大小策略为 Expanding
        h_layout2.addWidget(self.predictionResultLabel)
        h_layout2.addWidget(self.predictionResultEdit)

        # 创建一个垂直布局
        predict_layout = QVBoxLayout(predict_frame)
        predict_layout.addLayout(h_layout1)
        predict_layout.addLayout(h_layout2)

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.addLayout(label_layout)
        main_layout.addWidget(self.settingframe)
        main_layout.addWidget(status_frame)
        main_layout.addWidget(action_frame)
        main_layout.addWidget(predict_frame)

        self.setLayout(main_layout)

    def updateLabels(self, input_size, output_size, hidden_size, epochs, learning_rate):
        self.inputSizeEdit.setPlaceholderText(str(input_size))
        self.outputSizeEdit.setPlaceholderText(str(output_size))
        self.hiddenSizeEdit.setPlaceholderText(str(hidden_size))
        self.epochsEdit.setPlaceholderText(str(epochs))
        self.learningRateEdit.setPlaceholderText(str(learning_rate))

    def select_file_1(self):
        # 选择文件逻辑
        file_name_train, _ = QFileDialog.getOpenFileName(self, "选择文件")
        if file_name_train:
            self.url_input_1.setText(file_name_train)

    def select_file_2(self):
        # 选择文件逻辑
        file_name_test, _ = QFileDialog.getOpenFileName(self, "选择文件")
        if file_name_test:
            self.url_input_2.setText(file_name_test)

    def load_data_and_show_table(self):
        file_path = self.url_input_1.text()  # 假设你从输入框获取文件路径
        print(file_path)
        self.communicate.updateData.emit(file_path)
        file_path2 = self.url_input_2.text()  # 假设你从输入框获取文件路径
        print(file_path2)
        self.communicate2.updateData.emit(file_path2)

    def setstatus(self):
        self.trainStatusEdit.setPlaceholderText("训练中")
        #红色的字体
        self.trainStatusEdit.setStyleSheet("color: blue;")
        QTimer.singleShot(500, self.train_and_plot)


    def train_and_plot(self):
        # 获取输入的参数

        global y_min_global, y_max_global, x_min_global, x_max_global, input_size_global, output_size_global, hidden_size_global, epochs_global, learning_rate_global, network_global, test_losses_global, recall_losses_global, train_losses_global, x_test_global, y_test_global, x_train_global, y_train_global
        input_size = input_size_global
        output_size = output_size_global
        hidden_size = hidden_size_global
        epochs = epochs_global
        learning_rate = learning_rate_global

        self.network = network_global
        self.x_test = x_test_global
        self.y_test = y_test_global
        self.x_train = x_train_global
        self.y_train = y_train_global
        # 检测输入的参数是否合法
        if input_size <= 0 or hidden_size <= 0 or output_size <= 0:
            print("Invalid input size, hidden size or output size.")
            # 弹窗提示
            self.msg = Dialog("提示", "参数不合法，请重新输入")
            self.msg.cancelButton.hide()
            self.msg.buttonLayout.insertStretch(1)
            self.msg.exec_()
            self.trainStatusEdit.setPlaceholderText("训练失败")
            #红色
            self.trainStatusEdit.setStyleSheet("color: red;")
            return False

        # 创建神经网络实例
        self.network = NeuralNetwork(input_size, hidden_size, output_size)

        # 生成训练数据
        file_path = self.url_input_1.text()
        # 判断路径是否合法
        if not os.path.exists(file_path):
            print("Invalid file path.")
            # 弹窗提示
            self.msg = Dialog("提示", "训练数据文件路径无效")
            self.msg.cancelButton.hide()
            self.msg.buttonLayout.insertStretch(1)
            self.msg.exec_()
            # 状态显示
            self.trainStatusEdit.setPlaceholderText("训练失败")
            # 红色
            self.trainStatusEdit.setStyleSheet("color: red;")
            return False
        filetest_path = self.url_input_2.text()
        # 判断路径是否合法
        if not os.path.exists(filetest_path):
            print("Invalid file path.")
            # 弹窗提示
            self.msg = Dialog("提示", "测试数据文件路径无效")
            self.msg.cancelButton.hide()
            self.msg.buttonLayout.insertStretch(1)
            self.msg.exec_()

            #状态显示
            self.trainStatusEdit.setPlaceholderText("训练失败")
            #红色
            self.trainStatusEdit.setStyleSheet("color: red;")
            return False
        self.x_train, self.y_train, x_min_global, x_max_global, y_min_global, y_max_global = read_data_from_csv_train(
            file_path, input_size, output_size)
        # self.x_train, self.y_train = read_data_from_csv(file_path, input_size, output_size)
        self.x_test, self.y_test = read_data_from_csv_test(filetest_path, input_size, output_size, x_min_global,
                                                           x_max_global, y_min_global, y_max_global)
        # self.x_test, self.y_test = read_data_from_csv(filetest_path, input_size, output_size)
        # 打印全局最大最小值
        print("x_min_global:", x_min_global)
        print("x_max_global:", x_max_global)
        print("y_min_global:", y_min_global)
        print("y_max_global:", y_max_global)
        # 训练神经网络
        self.train_losses = train_neural_network(self.network, self.x_train, self.y_train, learning_rate, epochs)
        self.recall_losses = train_neural_network(self.network, self.x_train, self.y_train, learning_rate, epochs)
        self.test_losses = train_neural_network(self.network, self.x_test, self.y_test, learning_rate, epochs)
        # 保存全局变量
        network_global = self.network
        test_losses_global = self.test_losses
        recall_losses_global = self.recall_losses
        train_losses_global = self.train_losses
        x_test_global = self.x_test
        y_test_global = self.y_test
        x_train_global = self.x_train
        y_train_global = self.y_train

        res = {
            'network': self.network,
            'test_losses': self.test_losses,
            'recall_losses': self.recall_losses,
            'train_losses': self.train_losses,
            'x_test': self.x_test,
            'y_test': self.y_test,
            'x_train': self.x_train,
            'y_train': self.y_train
        }
        self.onTrainDataChanged.emit(res)

        # 训练完成
        self.trainStatusEdit.setPlaceholderText("训练完成")
        #绿色
        self.trainStatusEdit.setStyleSheet("color: green;")
        # self.spinner.stopAnimation()  # 停止环形进度条动画

        # 绘制训练曲线
        # self.plot_training_curve()

    def predict_output(self):
        global y_min_global, y_max_global, x_min_global, x_max_global, network_global, input_size_global, output_size_global, hidden_size_global, epochs_global, learning_rate_global
        self.network = network_global
        # 如果没有训练神经网络，则提示训练神经网络
        if self.network is None:
            # 如果没有训练神经网络，则直接输出请训练神经网络
            print("Please train the neural network first.")
            # gui界面弹窗提示
            self.msg_train = Dialog("提示", "请先训练神经网络")
            self.msg_train.cancelButton.hide()
            self.msg_train.buttonLayout.insertStretch(1)

            self.msg_train.exec_()
            return
        # 如果没有输入数据，则提示输入数据
        if self.manualInputLineEdit.text() == '':
            print("Please input data.")
            # gui界面弹窗提示
            self.msg_input = Dialog("提示", "请输入数据")
            self.msg_input.cancelButton.hide()
            self.msg_input.buttonLayout.insertStretch(1)
            self.msg_input.exec_()
            return
        # 输入的数据个数小于输入层的个数
        if len(self.manualInputLineEdit.text().split()) < input_size_global:
            print("Invalid input size.")
            # gui界面弹窗提示
            self.msg_input_size = Dialog("提示", "参数过少")
            self.msg_input_size.cancelButton.hide()
            self.msg_input_size.buttonLayout.insertStretch(1)
            self.msg_input_size.exec_()
            return

        if len(self.manualInputLineEdit.text().split()) > input_size_global:
            print("Invalid input size.")
            # gui界面弹窗提示
            self.msg_input_size = Dialog("提示", "参数过多")
            self.msg_input_size.cancelButton.hide()
            self.msg_input_size.buttonLayout.insertStretch(1)
            self.msg_input_size.exec_()
            return
        # 检查是否是数字
        try:
            # 获取手动输入的数据
            input_data_str = self.manualInputLineEdit.text()
            input_data = np.array([float(x) for x in input_data_str.split()])
        except ValueError:
            print("Invalid input data.")
            # gui界面弹窗提示
            self.msg_input_data = Dialog("提示", "无效输入数据")
            self.msg_input_data.cancelButton.hide()
            self.msg_input_data.buttonLayout.insertStretch(1)
            self.msg_input_data.exec_()
            return
        # 检查输入数据的范围是否在全局范围内
        if np.any(input_data > x_max_global):
            print("Invalid input data range.")
            # gui界面弹窗提示
            self.msg_input_range = Dialog("提示", "输入数据过大")
            self.msg_input_range.cancelButton.hide()
            self.msg_input_range.buttonLayout.insertStretch(1)
            self.msg_input_range.exec_()
            return
        if np.any(input_data < x_min_global):
            print("Invalid input data range.")
            # gui界面弹窗提示
            self.msg_input_range = Dialog("提示", "输入数据过小")
            self.msg_input_range.cancelButton.hide()
            self.msg_input_range.buttonLayout.insertStretch(1)
            self.msg_input_range.exec_()
            return

        # 对输入数据进行归一化处理
        input_data = (input_data - x_min_global) / (x_max_global - x_min_global)
        print("Normalized Input:", input_data)
        # 进行预测
        output, hidden_output = self.network.forward_propagation(input_data.reshape(1, -1))
        print("Normalized Output:", output)
        # # 反归一化输出数据
        output = inverse_normalize_output(output, y_min_global, y_max_global)
        # 显示预测结果
        self.predictionResultEdit.setPlaceholderText(str(output))

        # 输出正向预测的函数
        print("Forward Propagation Function:")
        print(
            f'hidden_output = sigmoid({np.dot(input_data, self.network.weights_input_hidden) + self.network.biases_hidden})')
        print(
            f'output = sigmoid({np.dot(hidden_output, self.network.weights_hidden_output) + self.network.biases_output})')


class plot_widget(QFrame):
    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        self.setObjectName(text.replace(' ', '-'))
        self.network = None
        self.test_losses = None
        self.recall_losses = None
        self.train_losses = None
        self.x_test = None
        self.y_test = None
        self.x_train = None
        self.y_train = None
        self.output_test = None
        # 初始化数据
        self.setup_ui()

    def handleTrainDataChanged(self, data):
        # 处理接收到的数据，并更新绘图
        self.network = data['network']
        self.test_losses = data['test_losses']
        self.recall_losses = data['recall_losses']
        self.train_losses = data['train_losses']
        self.x_test = data['x_test']
        self.y_test = data['y_test']
        self.x_train = data['x_train']
        self.y_train = data['y_train']
        self.update()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # 创建一个网格布局
        grid_layout = QGridLayout()
        self.update_button = PushButton('更新绘图')
        self.update_button.clicked.connect(self.plot_button_clicked)
        layout.addWidget(self.update_button)

        # 为每个绘图创建一个小部件
        self.training_widget = pg.PlotWidget()
        self.training_widget.setBackground('w')
        self.recall_widget = pg.PlotWidget()
        self.recall_widget.setBackground('w')
        self.generalization_widget = pg.PlotWidget()
        self.generalization_widget.setBackground('w')
        self.network_widget = pg.PlotWidget()  # 使用PlotWidget来显示网络结构图像
        self.network_widget.setBackground('w')
        # self.network_widget = QLabel()  # 使用QLabel来显示网络结构图像
        # self.network_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)  # 设置图像居中显示

        # 将小部件添加到网格布局中
        grid_layout.addWidget(self.training_widget, 0, 0)
        grid_layout.addWidget(self.recall_widget, 0, 1)
        grid_layout.addWidget(self.generalization_widget, 1, 0)
        grid_layout.addWidget(self.network_widget, 1, 1)

        # 将网格布局设置为主布局
        layout.addLayout(grid_layout)
        # 添加按钮


    def plot_training_curve(self):
        if self.train_losses is None:
            # 如果没有训练数据，则提示训练神经网络,弹窗
            self.msg = Dialog("提示", "请先训练神经网络")
            self.msg.cancelButton.hide()
            self.msg.buttonLayout.insertStretch(1)
            self.msg.exec_()
            print("请先训练神经网络")
            return

        # Clear previous plot
        self.training_widget.clear()

        # Plot training curve
        self.training_widget.plot(self.train_losses, pen='b')
        self.training_widget.setLabel('left', 'Loss')
        self.training_widget.setLabel('bottom', 'Epochs')
        self.training_widget.setTitle('Training Curve')
        self.training_widget.setBackground('w')

    def plot_recall_curve(self):
        if self.output_test is None:
            self.output_test, _ = self.network.forward_propagation(self.x_test)

        # Clear previous plot
        self.recall_widget.clear()

        # Plot recall curve
        self.recall_widget.plot(self.recall_losses, pen='b')
        self.recall_widget.setLabel('left', 'Loss')
        self.recall_widget.setLabel('bottom', 'Epochs')
        self.recall_widget.setTitle('Recall Curve')
        self.recall_widget.setBackground('w')

    def plot_generalization_curve(self):
        if self.output_test is None:
            self.output_test, _ = self.network.forward_propagation(self.x_test)

        # Clear previous plot
        self.generalization_widget.clear()

        # Plot generalization curve
        self.generalization_widget.plot(self.test_losses, pen='b')
        self.generalization_widget.setLabel('left', 'Loss')
        self.generalization_widget.setLabel('bottom', 'Epochs')
        self.generalization_widget.setTitle('Generalization Curve')
        self.generalization_widget.setBackground('w')

    def plot_network_structure(self):
        if self.network is None:
            print("请先训练神经网络")
            return

        # Clear previous plot
        self.network_widget.clear()

        # Plot network structure
        if self.network:
            network_graph = self.network.visualize_network()
            network_graph.attr('node', style='filled', shape='box', fillcolor='lightblue')
            network_graph.attr('edge', arrowsize='0.5')
            network_graph.attr('graph', fontsize='10')  # 设置字体大小
            network_graph.render('network', format='png', cleanup=True)
            network_image = QPixmap('network.png')  # 加载png图像

            # 旋转180度
            transform = QTransform().rotate(180)
            network_image = network_image.transformed(transform)

            # 水平镜像翻转
            network_image = network_image.transformed(QTransform().scale(-1, 1))

            # 创建 QGraphicsPixmapItem
            network_item = QGraphicsPixmapItem(network_image)

            # 添加 QGraphicsPixmapItem 到 network_widget
            self.network_widget.addItem(network_item)

    def plot_button_clicked(self):
        self.plot_training_curve()
        self.plot_recall_curve()  # Do not need these lines anymore
        self.plot_generalization_curve()  # Do not need these lines anymore
        self.plot_network_structure()  # Do not need these lines anymore

class Window(FramelessWindow):

    def __init__(self):
        super().__init__()
        self.setTitleBar(StandardTitleBar(self))

        self.hBoxLayout = QHBoxLayout(self)
        self.navigationInterface = NavigationInterface(self, showMenuButton=True)
        self.stackWidget = QStackedWidget(self)

        self.settingWidget = setting_widget('设置界面', self)
        self.startWidget = start_widget('首页', self, self.settingWidget)
        self.trainTable = data_traintable('训练数据', self,self.startWidget)
        self.testTable = data_testtable('测试数据', self,self.startWidget)
        self.plotWidget = plot_widget('画图界面', self)


        # 创建子界面
        self.settingInterface = self.settingWidget
        self.searchInterface = self.startWidget
        self.musicInterface1 = self.trainTable
        self.musicInterface2 = self.testTable

        self.albumInterface = self.plotWidget

        self.startWidget.onTrainDataChanged.connect(self.plotWidget.handleTrainDataChanged)

        # 初始化布局
        self.initLayout()

        # 添加项目到导航界面
        self.initNavigation()

        self.initWindow()

    def initLayout(self):
        self.hBoxLayout.setSpacing(0)
        self.hBoxLayout.setContentsMargins(0, self.titleBar.height(), 0, 0)
        self.hBoxLayout.addWidget(self.navigationInterface)
        self.hBoxLayout.addWidget(self.stackWidget)
        self.hBoxLayout.setStretchFactor(self.stackWidget, 1)

    def initNavigation(self):
        # 启用亚克力效果
        # self.navigationInterface.setAcrylicEnabled(True)

        self.addSubInterface(self.searchInterface, FIF.ROBOT, '首页')
        self.addSubInterface(self.musicInterface1, FIF.FOLDER, '训练数据')
        self.addSubInterface(self.musicInterface2, FIF.FOLDER, '测试数据')

        self.navigationInterface.addSeparator()

        self.addSubInterface(self.albumInterface, FIF.EDIT, '画图', NavigationItemPosition.SCROLL)

        self.addSubInterface(self.settingInterface, FIF.SETTING, '设置', NavigationItemPosition.BOTTOM)

        self.stackWidget.currentChanged.connect(self.onCurrentInterfaceChanged)
        self.stackWidget.setCurrentIndex(1)

    def initWindow(self):
        self.resize(900, 700)
        self.setWindowIcon(QIcon('resource/logo.png'))
        self.setWindowTitle('神经网络')
        self.titleBar.setAttribute(Qt.WA_StyledBackground)

        desktop = QApplication.desktop().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w//2 - self.width()//2, h//2 - self.height()//2)

        self.setQss()

    def addSubInterface(self, interface, icon, text: str, position=NavigationItemPosition.TOP, parent=None):
        """ 添加子界面 """
        self.stackWidget.addWidget(interface)
        self.navigationInterface.addItem(
            routeKey=interface.objectName(),
            icon=icon,
            text=text,
            onClick=lambda: self.switchTo(interface),
            position=position,
            tooltip=text,
            parentRouteKey=parent.objectName() if parent else None
        )

    def setQss(self):
        color = 'dark' if isDarkTheme() else 'light'
        with open(f'resource/{color}/demo.qss', encoding='utf-8') as f:
            self.setStyleSheet(f.read())

    def switchTo(self, widget):
        self.stackWidget.setCurrentWidget(widget)

    def onCurrentInterfaceChanged(self, index):
        widget = self.stackWidget.widget(index)
        self.navigationInterface.setCurrentItem(widget.objectName())

        # !IMPORTANT: This line of code needs to be uncommented if the return button is enabled
        # qrouter.push(self.stackWidget, widget.objectName())

    def onCurrentInterfaceChanged(self, index):
        widget = self.stackWidget.widget(index)
        self.navigationInterface.setCurrentItem(widget.objectName())





if __name__ == '__main__':
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    w = Window()
    w.show()
    app.exec_()
