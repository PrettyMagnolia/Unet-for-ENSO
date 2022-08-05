# U-Net网络处理ENSO偏差订正和降尺度

## 环境配置：
* Python==3.7.13
* torch==1.12.0
* torchvision==0.13.0

## 文件结构：
```
  ├── loss: 存放训练时的损失值  
  ├── my_module: 存放训练好的模型
  ├── outcome: 存放验证模型时输出的图像
  ├── net2.py: U-net网络模型代码
  ├── net2_test.py: 验证模型代码
  ├── net2_train.py: 训练模型代码
  ├── net2_utils.py: 需要的工具函数  
  └── Unet_data.py: 自定义dataset读取数据集
```

## 数据集下载地址：
* 官网地址： [https://drive.grand-challenge.org/](https://drive.grand-challenge.org/)
* 百度云链接： [https://pan.baidu.com/s/1Tjkrx2B9FgoJk0KviA-rDw](https://pan.baidu.com/s/1Tjkrx2B9FgoJk0KviA-rDw)  密码: 8no8


## 训练及测试方法
* 提前准备好数据集并放入 `dataset` 文件夹中
* 运行 `net2_train.py` 训练，训练出的模型将保存至 `my_module` 文件夹，训练过程中的损失值以及损失值变化曲线将保存至 `loss` 文件夹

​		注意修改模型保存的名称，防止覆盖已有模型

```python
# 模型保存路径
torch.save(net, r'./my_module/module_2.pth')
```

* 运行 `net2_test.py` 测试，测试得到的图像将会保存至 `outcome` 文件夹中

  修改模型路径以应用对应的模型

```python
# 模型路径
PATH = r'./my_module/module_1.pth'
```

​		修改验证的序号以验证不同月份的数据（0-203）

```python
# 选择验证的序号
my_choice = 200
```

