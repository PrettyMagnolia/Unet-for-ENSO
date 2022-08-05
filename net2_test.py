import torch
import numpy as np
from net2_utils import *

# 确定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 模型路径
PATH = r'./my_module/module_1.pth'
module = torch.load(PATH)
module.eval()

# 读取image第一个图层
image_1 = np.load('./dataset/input_observe.npy')
image_1 = np.nan_to_num(image_1, nan=0)
# 读取image第二个图层
image_2 = np.load('./dataset/input_pred.npy')
image_2 = np.nan_to_num(image_2, nan=0)
# 读取mask
mask = np.load('./dataset/output.npy')
mask = np.nan_to_num(mask, nan=0)

# 选择验证的序号
my_choice = 200

output = module(make_test_data(image_1[my_choice], image_2[my_choice], device))
output_ = F.interpolate(output, scale_factor=4, mode='nearest')
output_ = output_.squeeze(0).squeeze(0).cpu().detach().numpy()

make_plot(720, 1440, image_2[my_choice], "原预测图像")
make_plot(720, 1440, output_, "经过模型校正的预测图象")
make_plot(720, 1440, mask[my_choice], "原观测图像")
