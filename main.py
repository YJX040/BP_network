# coding:utf-8
import os

from PyQt5.QtGui import QIcon, QTransform, QFont, QPalette, QBrush
from PyQt5.QtWidgets import  QFrame, QStackedWidget, QGridLayout,\
    QGraphicsPixmapItem, QFileDialog
from qfluentwidgets import FluentIcon as FIF, PushButton, TableWidget,LineEdit, LargeTitleLabel, Dialog, \
    BodyLabel
from qfluentwidgets import (NavigationInterface, NavigationItemPosition,isDarkTheme)
from qfluentwidgets import (SpinBox,  DoubleSpinBox)
from qframelesswindow import FramelessWindow, StandardTitleBar
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSizePolicy
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, QObject,  QTimer
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


def inverse_normalize_data(data_normalized, min_val, max_val):
    # 还原归一化后的数据到原始范围
    data = data_normalized * (max_val - min_val) + min_val
    return data


def inverse_normalize_output(output_normalized, y_min, y_max):
    # 反归一化输出数据
    output = inverse_normalize_data(output_normalized, y_min, y_max)
    return output

def read_data_from_csv_train(file_path, input_size, output_size):
    # 跳过表头
    data = pd.read_csv(file_path, skiprows=1)
    # 检查输入特征和输出特征的数量是否符合预期
    if (len(data.columns) != (input_size + output_size)):
        flag = False
        return flag,None,None,None,None,None,None
    else:
        flag = True
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

        return flag,x_train_normalized, y_train_normalized, x_min, x_max, y_min, y_max

# 生成测试数据
def read_data_from_csv_test(file_path, input_size, output_size, x_min, x_max, y_min, y_max):
    # 跳过表头
    data = pd.read_csv(file_path, skiprows=1)
    # 检查输入特征和输出特征的数量是否符合预期
    if (len(data.columns) != input_size + output_size) :
        flag = False
        # print("flag",flag)
        return flag , None,None
    else:
        flag = True
        # print("flag",flag)
        # 提取输入特征
        x_test = data.iloc[:, :input_size].values
        # 提取目标变量
        y_test = data.iloc[:, -output_size:].values

        # 归一化输入特征
        x_test_normalized = (x_test - x_min) / (x_max - x_min)

        # 归一化目标变量
        y_test_normalized = (y_test - y_min) / (y_max - y_min)

        return flag,x_test_normalized, y_test_normalized

def train_neural_network(network, x_train, y_train, learning_rate, epochs):

    losses = []
    for epoch in range(epochs):
        # 前向传播
        output, hidden_output = network.forward_propagation(x_train)

        # 计算损失
        loss = network.mse_loss(output, y_train)
        # 打印损失，保留五位小数
        # if (epoch + 1) % 1000 == 0:
        #     print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.5f}')
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

class Widget(QFrame):

    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        self.label = QLabel(text, self)
        self.label.setAlignment(Qt.AlignCenter)
        self.hBoxLayout = QHBoxLayout(self)
        self.hBoxLayout.addWidget(self.label, 1, Qt.AlignCenter)
        self.setObjectName(text.replace(' ', '-'))

# 训练数据展示
class data_traintable(QWidget):

    def __init__(self, text: str, parent=None, start_widget_instance=None):
        super().__init__()
        self.file_path = 'training_data.csv'
        # 接受start_widget_instance传来的filepath
        self.data_widget_instance = start_widget_instance
        self.data_widget_instance.communicate.updateData.connect(self.update)
        # 设置对象名称
        self.setObjectName(text.replace(' ', '-'))
        # 添加标签
        self.label = BodyLabel("train")
        #颜色
        self.label.setStyleSheet("color: #0078d4;font-size: 30px;font-weight: bold;")
        self.label.setAlignment(Qt.AlignCenter)
        # 创建垂直布局
        self.mainLayout = QVBoxLayout()
        # 创建水平布局
        self.titleLayout = QHBoxLayout()
        self.titleLayout.addWidget(self.label)
        # 创建水平布局
        self.tableLayout = QHBoxLayout()
        self.tableView = TableWidget()
        self.tableView.setBorderVisible(True)
        self.tableView.setBorderRadius(8)
        self.tableView.setWordWrap(False)
        desktop = QApplication.desktop().availableGeometry()
        w= desktop.width()
        self.tableView.setMinimumWidth(int(w * 0.7))
        self.tableView.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tableLayout.addWidget(self.tableView, alignment=Qt.AlignHCenter)
        # 添加布局
        self.mainLayout.addLayout(self.titleLayout)
        self.mainLayout.addLayout(self.tableLayout)
        # 设置布局
        self.setLayout(self.mainLayout)
    # 更新数据
    def update(self, file_path):
        self.file_path = file_path
        self.load_data(file_path)
    # 加载数据
    def load_data(self, file_path):
        data = show_dataset(self, file_path)
        # 检查数据是否为空
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

    # 设置列的最大宽度
    def set_max_column_width(self, max_width):
        for column in range(self.tableView.columnCount()):
            width = self.tableView.columnWidth(column)
            if width > max_width:
                self.tableView.setColumnWidth(column, max_width)
# 测试数据展示
class data_testtable(QWidget):

    def __init__(self, text: str, parent=None, start_widget_instance=None):
        super().__init__()
        self.file_path = 'training_data_test.csv'
        # 接受start_widget_instance传来的filepath
        self.data_widget_instance = start_widget_instance
        self.data_widget_instance.communicate2.updateData.connect(self.update)
        # 设置对象名称
        self.setObjectName(text.replace(' ', '-'))
        self.label = LargeTitleLabel("test")
        #颜色
        self.label.setStyleSheet("color: #0078d4;font-size: 30px;font-weight: bold;")
        self.label.setAlignment(Qt.AlignCenter)
        # 创建垂直布局
        self.mainLayout = QVBoxLayout()
        # 创建水平布局
        self.titleLayout = QHBoxLayout()
        self.titleLayout.addWidget(self.label)
        # 创建水平布局
        self.tableLayout = QHBoxLayout()
        self.tableView = TableWidget()
        self.tableView.setBorderVisible(True)
        self.tableView.setBorderRadius(8)
        self.tableView.setWordWrap(False)
        desktop = QApplication.desktop().availableGeometry()
        w= desktop.width()
        self.tableView.setMinimumWidth(int(w * 0.7))

        self.tableView.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tableLayout.addWidget(self.tableView, alignment=Qt.AlignHCenter)
        # 添加布局
        self.mainLayout.addLayout(self.titleLayout)
        self.mainLayout.addLayout(self.tableLayout)
        # 设置布局
        self.setLayout(self.mainLayout)
    # 更新数据
    def update(self, file_path):
        self.file_path = file_path
        self.load_data(file_path)
    # 加载数据
    def load_data(self, file_path):
        data = show_dataset(self, file_path)
        # 检查数据是否为空
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
# 设置信息传递
class SettingCommunicate(QObject):
    updateSettings = pyqtSignal(int, int, int, int, float)
# 数据信息传递
class dataCommunicate(QObject):
    updateData = pyqtSignal(str)
# 设置页面
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
        # 设置步长
        self.epochsEdit.setSingleStep(50)
        self.learningRateEdit.setMinimum(0.001)
        self.learningRateEdit.setMaximum(1.0)
        # 设置步长
        self.learningRateEdit.setSingleStep(0.001)

        # 设置默认值
        self.inputSizeEdit.setValue(1)
        self.outputSizeEdit.setValue(1)
        self.hiddenSizeEdit.setValue(3)
        self.epochsEdit.setValue(5000)
        self.learningRateEdit.setValue(0.015)

        # 添加标签
        self.inputSizeLabel = QLabel("Input Size:", self)
        #蓝色
        self.inputSizeLabel.setStyleSheet("color: #0078d4;")

        # 监听数值改变信号
        self.inputSizeEdit.valueChanged.connect(lambda value: print("input当前值：", value))
        # print(self.inputSizeEdit.value())
        # 监听数值改变信号
        self.inputSizeEdit.setValue(input_size_global)

        self.outputSizeLabel = QLabel("Output Size:", self)
        # 蓝色
        self.outputSizeLabel.setStyleSheet("color: #0078d4;")
        # 监听数值改变信号
        self.outputSizeEdit.valueChanged.connect(lambda value: print("output当前值：", value))

        self.outputSizeEdit.setValue(output_size_global)
        # print(self.outputSizeEdit.value())


        self.hiddenSizeLabel = QLabel("Hidden Size:", self)
        # 监听数值改变信号
        self.hiddenSizeEdit.valueChanged.connect(lambda value: print("hidden当前值：", value))
        self.hiddenSizeEdit.setValue(hidden_size_global)
        # print(self.hiddenSizeEdit.value())

        self.epochsLabel = QLabel("Epochs:", self)
        # 监听数值改变信号
        self.epochsEdit.valueChanged.connect(lambda value: print("epochs当前值：", value))
        self.epochsEdit.setValue(epochs_global)
        # print(self.epochsEdit.value())

        self.learningRateLabel = QLabel("Learning Rate:", self)
        # 监听数值改变信号
        self.learningRateEdit.valueChanged.connect(lambda value: print("learningrate当前值：", value))
        self.learningRateEdit.setValue(learning_rate_global)
        # print("{:.3f}".format(self.learningRateEdit.value()))
        # 添加跳转首页按钮
        self.homeButton = PushButton("返回首页")
        self.homeButton.setStyleSheet("background-color: white;border-radius: 5px; color:#0078d4;min-width: 100px;min-height: 30px;font-size: 16px;border: 1px solid #0078d4;")

        # 添加确认按钮
        self.confirmButton = PushButton("确认")
        self.confirmButton.setStyleSheet("background-color: white;border-radius: 5px; color:#0078d4;min-width: 100px;min-height: 30px;font-size: 16px;border: 1px solid #0078d4;")
        self.confirmButton.clicked.connect(self.confirmSettings)

        # 添加布局
        self.gridLayout = QGridLayout()
        # 设置边距
        self.gridLayout.setContentsMargins(100, 50, 100, 50)
        # 设置控件之间的间距
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
        self.gridLayout.addWidget(self.homeButton, 5, 0)
        self.gridLayout.addWidget(self.confirmButton, 5, 1)
        self.gridLayout.setSpacing(10)  # 设置控件之间的间距

        # 设置布局
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
        # 跳出提示框
        self.msg = Dialog("提示", "参数设置成功")
        self.msg.cancelButton.hide()
        self.msg.buttonLayout.insertStretch(1)
        self.msg.exec_()

        # 发送信号
        self.communicate.updateSettings.emit(input_size_global, output_size_global, hidden_size_global, epochs_global,
                                             learning_rate_global)
# 首页
class start_widget(QFrame):
    onTrainDataChanged = pyqtSignal(dict)
    def __init__(self, text: str, parent=None, setting_widget_instance=None):
        super().__init__(parent=parent)
        self.setting_widget_instance = setting_widget_instance
        self.setting_widget_instance.communicate.updateSettings.connect(self.updateLabels)
        self.setObjectName(text.replace(' ', '-'))
        # 数据接受
        self.communicate = dataCommunicate()
        self.communicate2  = dataCommunicate()

        # 添加标签，占据单独一行
        self.label = LargeTitleLabel("BP 神 经 网 络")
        #蓝色
        self.label.setStyleSheet("color: #0078d4;")
        #换个字体，圆润一点的
        self.label.setFont(QFont("Microsoft YaHei UI", 30, QFont.Bold))
        self.label.setAlignment(Qt.AlignCenter)

        self.labelmessage = BodyLabel("当前参数设置")
        #字体
        self.labelmessage.setFont(QFont("Microsoft YaHei UI", 20, QFont.Bold))
        #颜色
        self.labelmessage.setStyleSheet("color: #0078d4;")
        self.labelmessage.setAlignment(Qt.AlignCenter)

        # 添加标签
        self.inputSizeLabel = BodyLabel("输 入 :")
        #字体
        self.inputSizeLabel.setFont(QFont("Microsoft YaHei UI", 13))
        #颜色浅蓝
        self.inputSizeLabel.setStyleSheet("color: #0078d4;")
        # 添加标签
        self.inputSizeEdit = BodyLabel(str(input_size_global))
        #字体
        self.inputSizeEdit.setFont(QFont("Microsoft YaHei UI", 13))
        #颜色浅蓝
        self.inputSizeEdit.setStyleSheet("color: #0078d4;")

        # 添加标签
        self.outputSizeLabel = BodyLabel("输 出 :")
        #字体
        self.outputSizeLabel.setFont(QFont("Microsoft YaHei UI", 13))
        #颜色浅蓝
        self.outputSizeLabel.setStyleSheet("color: #0078d4;")
        # 添加标签
        self.outputSizeEdit = BodyLabel(str(output_size_global))
        #字体
        self.outputSizeEdit.setFont(QFont("Microsoft YaHei UI", 13))
        #颜色浅蓝
        self.outputSizeEdit.setStyleSheet("color: #0078d4;")

        # 添加标签
        self.hiddenSizeLabel = BodyLabel("隐 藏 层 :")
        #字体
        self.hiddenSizeLabel.setFont(QFont("Microsoft YaHei UI", 13))
        #颜色浅蓝
        self.hiddenSizeLabel.setStyleSheet("color: #0078d4;")
        # 添加标签
        self.hiddenSizeEdit = BodyLabel(str(hidden_size_global))
        #字体
        self.hiddenSizeEdit.setFont(QFont("Microsoft YaHei UI", 13))
        #颜色浅蓝
        self.hiddenSizeEdit.setStyleSheet("color: #0078d4;")


        # 添加标签
        self.epochsLabel = BodyLabel("迭 代 次 数 :")
        #字体
        self.epochsLabel.setFont(QFont("Microsoft YaHei UI", 13))
        #颜色浅蓝
        self.epochsLabel.setStyleSheet("color: #0078d4;")
        # 添加标签
        self.epochsEdit = BodyLabel(str(epochs_global))
        #字体
        self.epochsEdit.setFont(QFont("Microsoft YaHei UI", 13))
        #颜色浅蓝
        self.epochsEdit.setStyleSheet("color: #0078d4;")


        # 添加标签
        self.learningRateLabel = BodyLabel("学 习 率 :")
        #字体
        self.learningRateLabel.setFont(QFont("Microsoft YaHei UI", 13))
        #颜色浅蓝
        self.learningRateLabel.setStyleSheet("color: #0078d4;")
        # 添加标签
        self.learningRateEdit = BodyLabel(str(learning_rate_global))
        #字体
        self.learningRateEdit.setFont(QFont("Microsoft YaHei UI", 13))
        #颜色浅蓝
        self.learningRateEdit.setStyleSheet("color: #0078d4;")

        # 添加标签
        self.trainStatusLabel = BodyLabel("训 练 状 态 :")
        #字体
        self.trainStatusLabel.setFont(QFont("Microsoft YaHei UI", 16, QFont.Bold))
        #颜色浅蓝
        self.trainStatusLabel.setStyleSheet("color: #0078d4;")
        # 添加标签
        self.trainStatusEdit = BodyLabel("未 开 始")
        #字体
        self.trainStatusEdit.setFont(QFont("Microsoft YaHei UI", 16, QFont.Bold))
        #颜色红色
        self.trainStatusEdit.setStyleSheet("color: red;")
        # 添加按钮
        self.trainButton = PushButton("开 始 训 练")
        #背景蓝色
        self.trainButton.setStyleSheet("background-color:white;border-radius: 5px; color:#0078d4;min-width: 100px;min-height: 30px;font-size: 16px;border: 1px solid #0078d4;")
        #弹出确认弹窗，确认后开始训练
        self.trainButton.clicked.connect(self.setstatus)
        # self.trainButton.clicked.connect(self.train_and_plot)


        self.predictButton = PushButton("预 测")
        self.predictButton.setStyleSheet("background-color: white;border-radius: 5px; color:#0078d4;min-width: 100px;min-height: 30px;font-size: 16px;border: 1px solid #0078d4;")
        self.predictButton.clicked.connect(self.predict_output)

        self.predictionDataLabel = BodyLabel("预 测 数 据:")
        # 字体
        self.predictionDataLabel.setFont(QFont("Microsoft YaHei UI", 12))
        # 颜色浅蓝
        self.predictionDataLabel.setStyleSheet("color: #0078d4;")
        self.manualInputLineEdit = LineEdit()
        self.manualInputLineEdit.setPlaceholderText("请输入数据")
        self.manualInputLineEdit.setStyleSheet("background-color:white; border-radius: 5px; color:#0078d4;")

        self.predictionResultLabel = BodyLabel("预 测 结 果:")
        #字体
        self.predictionResultLabel.setFont(QFont("Microsoft YaHei UI", 12))
        #颜色浅蓝
        self.predictionResultLabel.setStyleSheet("color: #0078d4;")

        self.predictionResultEdit = LineEdit()
        self.predictionResultEdit.setPlaceholderText("无")
        self.predictionResultEdit.setStyleSheet("background-color: white; border-radius: 5px; color:#0078d4;")
        self.predictionResultEdit.setReadOnly(True)
        self.predictionResultEdit.setDisabled(True)


        # 文件输入
        self.url_input_1 = LineEdit()
        self.url_input_1.setPlaceholderText("训练数据路径")
        #居中
        self.url_input_1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.url_input_1.setReadOnly(True)
        self.url_input_1.setStyleSheet("background-color: white;border-radius: 5px;color:#0078d4;")
        self.url_input_1.setDisabled(True)

        self.selectFileButton_1 = PushButton("添 加 训 练 文 件")


        #蓝色字体
        self.selectFileButton_1.setStyleSheet("background-color:white; border-radius: 5px; color:#0078d4;min-width: 100px;min-height: 30px;font-size: 16px;border: 1px solid #0078d4;")
        self.selectFileButton_1.clicked.connect(self.select_file_1)

        self.url_input_2 = LineEdit()
        self.url_input_2.setPlaceholderText("测试数据路径")
        #居中
        self.url_input_2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.url_input_2.setReadOnly(True)
        self.url_input_2.setStyleSheet("background-color:white;border-radius: 5px;color:#0078d4;")
        self.url_input_2.setDisabled(True)


        self.selectFileButton_2 = PushButton("添 加 测 试 文 件")
        #蓝色字体
        self.selectFileButton_2.setStyleSheet("background-color:white; border-radius: 5px; color:#0078d4;min-width: 100px;min-height: 30px;font-size: 16px;border: 1px solid #0078d4;")
        self.selectFileButton_2.clicked.connect(self.select_file_2)

        # 添加按钮
        self.button = PushButton("加 载 数 据")
        self.button.setStyleSheet("background-color: white;border-radius: 5px; color:#0078d4;min-width: 100px;min-height: 30px;font-size: 16px;border: 1px solid #0078d4;")
        self.button.clicked.connect(self.load_data_and_show_table)

        # Separate layout for the 神经网络 label
        label_layout = QHBoxLayout()
        label_layout.addWidget(self.label)
        self.label.setMaximumHeight(100)

        # Rest of your code remains the same...
        top_layout = QVBoxLayout()
        top_layout.addWidget(self.labelmessage)
        # 随文本长度自适应
        self.label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        widgets_row1 = [self.inputSizeLabel,self.inputSizeEdit,
                        self.outputSizeLabel, self.outputSizeEdit,
                        self.hiddenSizeLabel, self.hiddenSizeEdit]

        widgets_row2 = [self.epochsLabel, self.epochsEdit,
                        self.learningRateLabel, self.learningRateEdit]

        # 创建两个水平布局来分别放置 widgets_row1 和 widgets_row2
        h_layout1 = QVBoxLayout()  # 创建水平布局来容纳第一行的标签和编辑框
        for label, edit in zip(widgets_row1[::2], widgets_row1[1::2]):  # 使用zip函数一次迭代标签和编辑框
            widget_layout = QHBoxLayout()  # 创建一个垂直布局来容纳每对标签和编辑框
            widget_layout.addWidget(label)
            widget_layout.addWidget(edit)
            h_layout1.addLayout(widget_layout)  # 将每对标签和编辑框的垂直布局添加到水平布局中

        h_layout2 = QVBoxLayout()  # 创建水平布局来容纳第二行的标签和编辑框
        for label, edit in zip(widgets_row2[::2], widgets_row2[1::2]):  # 使用zip函数一次迭代标签和编辑框
            widget_layout = QHBoxLayout()  # 创建一个垂直布局来容纳每对标签和编辑框
            widget_layout.addWidget(label)
            widget_layout.addWidget(edit)
            h_layout2.addLayout(widget_layout)  # 将每对标签和编辑框的垂直布局添加到水平布局中

        status_layout = QHBoxLayout()
        status_layout.addWidget(self.trainStatusLabel)
        status_layout.addWidget(self.trainStatusEdit)
        # 将两个水平布局添加到垂直布局中

        top_layout.addLayout(h_layout1)
        top_layout.addLayout(h_layout2)
        top_layout.addLayout(status_layout)

        self.settingframe = QFrame()
        self.settingframe.setStyleSheet(
            # "background-color: transparent; border-radius: 5px;}"
            "QFrame { background-color: white; border-radius: 5px; }")
        self.settingframe.setLayout(top_layout)

        action_frame = QFrame()
        action_frame.setStyleSheet(
            # "QFrame { background-color:  transparent; border-radius: 5px;  }")
            "QFrame { background-color: white; border-radius: 5px;  }")
        action_layout1 = QHBoxLayout()
        action_layout1.addWidget(self.url_input_1)

        action_layout1.addWidget(self.url_input_2)

        action_layout2 = QHBoxLayout()
        action_layout2.addWidget(self.selectFileButton_1)
        action_layout2.addWidget(self.selectFileButton_2)

        action_layout3 = QHBoxLayout()
        action_layout3.addWidget(self.button)
        action_layout3.addWidget(self.trainButton)


        action_layout = QVBoxLayout(action_frame)
        action_layout.addLayout(action_layout1)
        action_layout.addLayout(action_layout2)
        action_layout.addLayout(action_layout3)



        predict_frame = QFrame()
        predict_frame.setStyleSheet(
            # "QFrame { background-color: transparent; border-radius: 5px;  }")
            "QFrame { background-color: white; border-radius: 5px;}")
        # 创建两个水平布局
        h_layout1_1 = QHBoxLayout()
        h_layout1_1.addWidget(self.predictionDataLabel)
        h_layout1_1.addWidget(self.manualInputLineEdit)
        h_layout1_2 = QHBoxLayout()
        h_layout1_2.addWidget(self.predictionResultLabel)
        h_layout1_2.addWidget(self.predictionResultEdit)



        h_layout2 = QHBoxLayout()
        h_layout2.addWidget(self.predictButton)

        # 创建一个垂直布局
        predict_layout = QVBoxLayout(predict_frame)
        predict_layout.addLayout(h_layout1_1)
        predict_layout.addLayout(h_layout1_2)
        predict_layout.addLayout(h_layout2)

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.addLayout(label_layout)
        main_layout.addWidget(self.settingframe)
        # main_layout.addWidget(status_frame)
        main_layout.addWidget(action_frame)
        main_layout.addWidget(predict_frame)

        self.setLayout(main_layout)

        # # 背景图片设置
        # self.setAutoFillBackground(True)
        # self.updateBackground()


    def updateBackground(self):
        # 背景图片设置,没找到合适的图
        desktop = QApplication.desktop().availableGeometry()
        w = desktop.width()
        h = desktop.height()

        self.bg = QPixmap('resource/change1_1.jpg').scaled(int(w*0.8),int(h*0.8), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(self.bg))
        self.setPalette(palette)
        # print(self.bg.size())

    def updateLabels(self, input_size, output_size, hidden_size, epochs, learning_rate):
        self.inputSizeEdit.setText(str(input_size))
        self.outputSizeEdit.setText(str(output_size))
        self.hiddenSizeEdit.setText(str(hidden_size))
        self.epochsEdit.setText(str(epochs))
        self.learningRateEdit.setText(str(learning_rate))
        self.trainStatusEdit.setText("等 待 训 练")
        self.trainStatusEdit.setStyleSheet("color: blue;")

    def select_file_1(self):
        # 选择文件逻辑
        file_name_train, _ = QFileDialog.getOpenFileName(self, "训练文件")
        if file_name_train:
            self.url_input_1.setText(file_name_train)

    def select_file_2(self):
        # 选择文件逻辑
        file_name_test, _ = QFileDialog.getOpenFileName(self, "测试文件")
        if file_name_test:
            self.url_input_2.setText(file_name_test)

    def load_data_and_show_table(self):
        file_path = self.url_input_1.text()  # 假设你从输入框获取文件路径
        # print(file_path)
        self.communicate.updateData.emit(file_path)
        file_path2 = self.url_input_2.text()  # 假设你从输入框获取文件路径
        # print(file_path2)
        self.communicate2.updateData.emit(file_path2)
        #加载成功弹出提示框
        if file_path and file_path2:
            self.msg = Dialog("提示", "数据加载成功")
            self.msg.cancelButton.hide()
            self.msg.buttonLayout.insertStretch(1)
            self.msg.exec_()

    # 预测
    def setstatus(self):
        #进行确认
        self.msg = Dialog("提示", "确认开始训练？")
        self.msg.buttonLayout.insertStretch(1)
        self.msg.exec_()
        if self.msg.result() == 1:
            self.trainStatusEdit.setText("训 练 中")
            #红色的字体
            self.trainStatusEdit.setStyleSheet("color: blue;")
            QTimer.singleShot(500, self.train_and_plot)
        else:
            pass

    # 训练
    def train_and_plot(self):
        # 获取输入的参数
        # 全局变量
        global y_min_global, y_max_global, x_min_global, x_max_global, input_size_global, output_size_global, hidden_size_global, epochs_global, learning_rate_global, network_global, test_losses_global, recall_losses_global, train_losses_global, x_test_global, y_test_global, x_train_global, y_train_global
        input_size = input_size_global
        output_size = output_size_global
        hidden_size = hidden_size_global
        epochs = epochs_global
        learning_rate = learning_rate_global
        # 全局变量
        self.network = network_global
        self.x_test = x_test_global
        self.y_test = y_test_global
        self.x_train = x_train_global
        self.y_train = y_train_global
        # 检测输入的参数是否合法
        if input_size <= 0 or hidden_size <= 0 or output_size <= 0:
            # print("Invalid input size, hidden size or output size.")
            # 弹窗提示
            self.msg = Dialog("提示", "参数不合法，请重新输入")
            self.msg.cancelButton.hide()
            self.msg.buttonLayout.insertStretch(1)
            self.msg.exec_()
            self.trainStatusEdit.setText("训 练 失 败")
            #红色
            self.trainStatusEdit.setStyleSheet("color: red;")
            return False

        # 创建神经网络实例
        self.network = NeuralNetwork(input_size, hidden_size, output_size)

        # 生成训练数据
        file_path = self.url_input_1.text()
        # 判断路径是否合法
        if not os.path.exists(file_path):
            # print("Invalid file path.")
            # 弹窗提示
            self.msg = Dialog("提示", "训练数据文件路径无效")
            self.msg.cancelButton.hide()
            self.msg.buttonLayout.insertStretch(1)
            self.msg.exec_()
            # 状态显示
            self.trainStatusEdit.setText("训 练 失 败")
            # 红色
            self.trainStatusEdit.setStyleSheet("color: red;")
            return False
        filetest_path = self.url_input_2.text()
        # 判断路径是否合法
        if not os.path.exists(filetest_path):
            # print("Invalid file path.")
            # 弹窗提示
            self.msg = Dialog("提示", "测试数据文件路径无效")
            self.msg.cancelButton.hide()
            self.msg.buttonLayout.insertStretch(1)
            self.msg.exec_()

            #状态显示
            self.trainStatusEdit.setText("训 练 失 败")
            #红色
            self.trainStatusEdit.setStyleSheet("color: red;")
            return False
        flag_train,self.x_train, self.y_train, x_min_global, x_max_global, y_min_global, y_max_global = read_data_from_csv_train(
            file_path, input_size, output_size)
        # print(input_size, output_size, hidden_size, epochs, learning_rate)
        # print(flag_train)
        if flag_train == False:
            # print("Invalid training data.")
            # 弹窗提示
            self.msg = Dialog("提示", "训练数据与设置参数不匹配")
            self.msg.cancelButton.hide()
            self.msg.buttonLayout.insertStretch(1)
            self.msg.exec_()
            #状态显示
            self.trainStatusEdit.setText("训 练 失 败")
            #红色
            self.trainStatusEdit.setStyleSheet("color: red;")
            return False

        flag_test,self.x_test, self.y_test = read_data_from_csv_test(filetest_path, input_size, output_size, x_min_global,
                                                           x_max_global, y_min_global, y_max_global)
        # print(input_size, output_size, hidden_size, epochs, learning_rate)
        # print(flag_test)
        if flag_test == False:
            # print("Invalid test data.")
            # 弹窗提示
            self.msg = Dialog("提示", "测试数据与设置参数不匹配")
            self.msg.cancelButton.hide()
            self.msg.buttonLayout.insertStretch(1)
            self.msg.exec_()
            #状态显示
            self.trainStatusEdit.setText("训 练 失 败")
            #红色
            self.trainStatusEdit.setStyleSheet("color: red;")
            return False
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
        self.trainStatusEdit.setText("训 练 完 成")
        #绿色
        self.trainStatusEdit.setStyleSheet("color: green;")

    # 弹窗提示
    def predict_output(self):
        global y_min_global, y_max_global, x_min_global, x_max_global, network_global, input_size_global, output_size_global, hidden_size_global, epochs_global, learning_rate_global
        self.network = network_global
        # 如果没有训练神经网络，则提示训练神经网络
        if self.network is None:
            # 如果没有训练神经网络，则直接输出请训练神经网络
            # print("Please train the neural network first.")
            # gui界面弹窗提示
            self.msg_train = Dialog("提示", "请先训练神经网络")
            self.msg_train.cancelButton.hide()
            self.msg_train.buttonLayout.insertStretch(1)

            self.msg_train.exec_()
            return
        # 如果没有输入数据，则提示输入数据
        if self.manualInputLineEdit.text() == '':
            # print("Please input data.")
            # gui界面弹窗提示
            self.msg_input = Dialog("提示", "请输入数据")
            self.msg_input.cancelButton.hide()
            self.msg_input.buttonLayout.insertStretch(1)
            self.msg_input.exec_()
            return
        # 输入的数据个数小于输入层的个数
        if len(self.manualInputLineEdit.text().split()) < input_size_global:
            # print("Invalid input size.")
            # gui界面弹窗提示
            self.msg_input_size = Dialog("提示", "参数过少")
            self.msg_input_size.cancelButton.hide()
            self.msg_input_size.buttonLayout.insertStretch(1)
            self.msg_input_size.exec_()
            return

        if len(self.manualInputLineEdit.text().split()) > input_size_global:
            # print("Invalid input size.")
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
            # print("Invalid input data.")
            # gui界面弹窗提示
            self.msg_input_data = Dialog("提示", "无效输入数据")
            self.msg_input_data.cancelButton.hide()
            self.msg_input_data.buttonLayout.insertStretch(1)
            self.msg_input_data.exec_()
            return
        # 检查输入数据的范围是否在全局范围内
        if np.any(input_data > x_max_global):
            # print("Invalid input data range.")
            # gui界面弹窗提示
            self.msg_input_range = Dialog("提示", "输入数据过大")
            self.msg_input_range.cancelButton.hide()
            self.msg_input_range.buttonLayout.insertStretch(1)
            self.msg_input_range.exec_()
            return
        if np.any(input_data < x_min_global):
            # print("Invalid input data range.")
            # gui界面弹窗提示
            self.msg_input_range = Dialog("提示", "输入数据过小")
            self.msg_input_range.cancelButton.hide()
            self.msg_input_range.buttonLayout.insertStretch(1)
            self.msg_input_range.exec_()
            return

        # 对输入数据进行归一化处理
        input_data = (input_data - x_min_global) / (x_max_global - x_min_global)
        # 进行预测
        output, hidden_output = self.network.forward_propagation(input_data.reshape(1, -1))

        # # 反归一化输出数据
        output = inverse_normalize_output(output, y_min_global, y_max_global)
        # 显示预测结果
        self.predictionResultEdit.setPlaceholderText(str(output))

# 画图
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
        self.update_button = PushButton('更 新 绘 图')
        self.update_button.setStyleSheet("background-color: white;border-radius: 5px; color:#0078d4;min-width: 100px;min-height: 30px;font-size: 16px;border: 1px solid #0078d4;")

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
            # print("请先训练神经网络")
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
        self.plot_recall_curve()
        self.plot_generalization_curve()
        self.plot_network_structure()
#总页面
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
        self.settingWidget.homeButton.clicked.connect(self.goToStartPage)
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
        self.navigationInterface.setAcrylicEnabled(True)
        desktop = QApplication.desktop().availableGeometry()
        w= desktop.width()
        self.navigationInterface.setExpandWidth(int(w * 0.1))

        self.addSubInterface(self.searchInterface, FIF.HOME, '首页')
        self.addSubInterface(self.musicInterface1, FIF.DOCUMENT, '训练数据')
        self.addSubInterface(self.musicInterface2, FIF.DOCUMENT, '测试数据')

        self.navigationInterface.addSeparator()

        self.addSubInterface(self.albumInterface, FIF.EDIT, '画图', NavigationItemPosition.SCROLL)

        self.addSubInterface(self.settingInterface, FIF.SETTING, '设置', NavigationItemPosition.BOTTOM)

        self.stackWidget.currentChanged.connect(self.onCurrentInterfaceChanged)
        self.stackWidget.setCurrentIndex(0)

    def initWindow(self):
        #设置成和分辨率有关的

        desktop = QApplication.desktop().availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.resize(int(w * 0.7), int(h * 0.8))
        self.move(w//2 - self.width()//2, h//2 - self.height()//2)

        # self.resize(900, 700)
        self.setWindowIcon(QIcon('resource/logo.png'))
        #打印窗口大小
        # print(self.size())
        self.setWindowTitle('神经网络')
        # 蓝色
        self.titleBar.setStyleSheet("color: #0078d4;font-weight: bold;")
        self.titleBar.setAttribute(Qt.WA_StyledBackground)


        #打印窗口位置
        # print(self.pos())
        self.setQss()

        # self.setQss()

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

    def goToStartPage(self):
        index = self.stackWidget.indexOf(self.startWidget)  # 获取 start_widget 的索引
        if index != -1:  # 如果页面存在
            self.stackWidget.setCurrentIndex(index)  # 切换到 start_widget 页面


if __name__ == '__main__':
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    w = Window()
    # 打印一下窗口的大小
    # print(w.size())
    w.show()
    app.exec_()
