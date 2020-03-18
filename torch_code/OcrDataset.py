import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
# from utils import image_processing
import os
from torch.utils.data import image_processing

class OcrDataset(torch.utils.data.Dataset):
    def __init__(self, txt_path, image_dir, resize_height=40, resize_width=40, repeat= 1):
        '''
        :param filename: 数据文件TXT：格式：imge_name.jpg label1_id labe2_id
        :param image_dir: 图片路径：image_dir+imge_name.jpg构成图片的完整路径
        :param resize_height 为None时，不进行缩放
        :param resize_width  为None时，不进行缩放，
                              PS：当参数resize_height或resize_width其中一个为None时，可实现等比例缩放
        :param repeat: 所有样本数据重复次数，默认循环一次，当repeat为None时，表示无限循环<sys.maxsize
        '''
        self.image_label_list = self.read_file(txt_path)
        self.image_dir = image_dir
        self.len = len(self.image_label_list)
        self.repeat = repeat
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.toTensor = transforms.ToTensor()

    def __getitem__(self, index): # 必须重写，返回训练数据，图片和label
        pass
    def __len__(self): # 必须重写，返回数据长度
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = len(self.image_label_list) * self.repeat
        return data_len

    def read_file(self, txt_path):
        image_label_list = []
        with open(txt_path, 'r')as f:
            lines = f.readlines()
            for line in lines:
                content = line.rstrip().split(' ')
                image_name = content[0]
                image_label = content[1]
                image_label_list.append((image_name, image_label))
        return image_label_list

    def load_data(self, path, resize_height, resize_width, normalization):
        '''
        加载数据
        :param path:
        :param resize_height:
        :param resize_width:
        :param normalization: 是否归一化
        :return:
        '''
        image = image_processing.read_image(path, resize_height, resize_width, normalization)
        return image

    def data_preproccess(self, data):
        '''
        数据预处理
        :param data:
        :return:
        '''
        data = self.toTensor(data)
        return data



if __name__ == '__main__':
    train_filename = "./train.txt"
    # test_filename="../dataset/test.txt"
    image_dir = './images'

    epoch_num = 2  # 总样本循环次数
    batch_size = 16  # 训练时的一组数据的大小
    train_data_nums = 10
    max_iterate = int((train_data_nums + batch_size - 1) / batch_size * epoch_num)  # 总迭代次数

    train_data = OcrDataset(txt_path=train_filename, image_dir=image_dir, repeat=1)
    # test_data = TorchDataset(filename=test_filename, image_dir=image_dir,repeat=1)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(dataset=test_data, batch_size=batch_size,shuffle=False)

    # [1]使用epoch方法迭代，TorchDataset的参数repeat=1
    for epoch in range(epoch_num):
        for batch_image, batch_label in train_loader:
            image = batch_image[0, :]
            image = image.numpy()  # image=np.array(image)
            image = image.transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]

            print("batch_image.shape:{},batch_label:{}".format(batch_image.shape, batch_label))

