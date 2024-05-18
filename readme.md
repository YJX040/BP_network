# BP神经网络

本项目是一个基于 PyQt5 和 pyqtgraph 的神经网络可视化应用。该应用提供了训练数据管理、数据可视化和网络结构展示等功能，旨在帮助用户更直观地理解神经网络的训练过程和结构。

## 功能特点

- **训练数据管理**：用户可以导入、展示和管理训练数据，包括训练集和测试集，确保数据的完整性和准确性。
- **数据可视化**：应用提供了多种数据可视化功能，包括训练曲线、回忆曲线和泛化曲线的绘制。用户可以通过这些曲线直观地分析神经网络的训练效果和泛化能力。
- **网络结构展示**：用户可以查看训练后的神经网络结构图，从而更直观地了解网络的层次结构和连接关系。

## 使用说明

### 环境要求

- Python 3.x
- PyQt5：用于创建用户界面和交互式应用程序。
- qfluentwidgets：提供了丰富的界面组件，使界面设计更加美观和易用。
- qframelesswindow：用于创建无边框窗口和自定义标题栏。
- numpy：用于处理数值数据和数组操作。
- pandas：用于数据处理和分析，包括读取和操作数据。
- graphviz：用于绘制图形和网络结构。
- pyqtgraph：提供了丰富的绘图功能，包括绘制曲线和图表等。

### 安装依赖

确保已安装 Python 3.x，并执行以下命令安装 PyQt5 和 pyqtgraph以及其他依赖项：

```
pip install PyQt5 qfluentwidgets qframelesswindow numpy pandas graphviz pyqtgraph
```

### 运行应用

在终端中进入应用所在目录，并执行以下命令启动应用：

* 首页用pyqt写的简易界面

```
python demo.py
```

* 稍微优化后的

```
python main.py
```


### 功能介绍

1. **首页**

   首页展示了基本的界面信息，包括设置界面、训练数据、测试数据等选项卡，用户可以根据需要切换不同功能界面。

   <img src="https://mine-picgo.oss-cn-beijing.aliyuncs.com/imgtest/image-20240518144300865.png" alt="image-20240518144300865" style="zoom: 25%;" />

2. **设置界面**

   设置界面用于配置应用的参数和选项，例如主题设置等。

   <img src="https://mine-picgo.oss-cn-beijing.aliyuncs.com/imgtest/image-20240518144352803.png" alt="image-20240518144352803" style="zoom:25%;" />

3. **训练数据**

   在训练数据选项卡中，用户可以管理训练数据，包括数据导入和展示。

   <img src="https://mine-picgo.oss-cn-beijing.aliyuncs.com/imgtest/image-20240518144327034.png" alt="image-20240518144327034" style="zoom: 25%;" />

4. **测试数据**

   在测试数据选项卡中，用户可以管理测试数据，包括数据导入和展示。

   <img src="https://mine-picgo.oss-cn-beijing.aliyuncs.com/imgtest/image-20240518144406091.png" alt="image-20240518144406091" style="zoom: 25%;" />

5. **画图界面**

   画图界面提供了数据的可视化功能，包括训练曲线、回忆曲线、泛化曲线等绘制，用户可以直观地分析数据。

   <img src="https://mine-picgo.oss-cn-beijing.aliyuncs.com/imgtest/image-20240518144458529.png" alt="image-20240518144458529" style="zoom:25%;" />

6. **神经网络**

   神经网络选项卡展示了训练后的神经网络结构图，用户可以查看网络的层次结构。

## 注意事项

- 请确保数据格式的正确性和一致性，以保证应用的正常运行和数据的准确性。
- pyinstaller打包需要注意版本，新版本打包去除终端时可能会出现检测到病毒等问题，卸载pyinstaller选择旧版本即可
