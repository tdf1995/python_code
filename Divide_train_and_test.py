'''
这个脚本用来分割数据集
产生训练和测试两个数据集
比例可调
'''
import  os,random,shutil
from os import walk
import glob

train_percent = 0.18 # 训练集的比例
root_path = r'\\192.168.1.251\1项目-2细胞\0血液细胞项目\图片汇总\新样机采集图片\3-已审核图片\1-临时训练汇总数据\1112_冯石佳整理_40倍_100倍_可用训练数据汇总_单核_淋巴已复审\1112_冯石佳整理_40倍_100倍_可用训练数据汇总\x40/'
train_path = r'\\192.168.1.251\1项目-2细胞\0血液细胞项目\图片汇总\新样机采集图片\3-已审核图片\1-临时训练汇总数据\1112_冯石佳整理_40倍_100倍_可用训练数据汇总_单核_淋巴已复审\1112_冯石佳整理_40倍_100倍_可用训练数据汇总\x401/'
if __name__ == '__main__':
    for (_,dirnames,_)in walk(root_path):
        for dirname in dirnames:
            path = os.path.join(root_path, dirname)
            pic_list = glob.glob(path+'/*.jpg')
            dirpath = path
            # pathdir = os.listdir(dirpath)
            num = len(pic_list)
            train_num = int(num*train_percent)
            trainset = random.sample(pic_list, train_num)
            for trainpic in trainset:
                if not os.path.exists(os.path.join(train_path, dirname)):
                    os.makedirs(os.path.join(train_path, dirname))
                shutil.move(trainpic, os.path.join(train_path,dirname, os.path.basename(trainpic)))
                # xml_path = trainpic[:-4]+'.xml'
                # new_xml_path = os.path.join(train_path,dirname, os.path.basename(trainpic))[:-4]+'.xml'
                # shutil.copy(xml_path, new_xml_path)
# if __name__ == '__main__':
#     for (_,dirnames,_)in walk(root_path):
#         for dirname in dirnames:
#             if dirname == 'JPEGImages':
#                 path = os.path.join(root_path, dirname)
#
#                 dirpath = path
#                 pathdir = os.listdir(dirpath)
#                 num = len(pathdir)
#                 train_num = int(num*train_percent)
#                 trainset = random.sample(pathdir, train_num)
#                 for trainpic in trainset:
#                     if not os.path.exists(os.path.join(train_path, dirname)):
#                         os.makedirs(os.path.join(train_path, dirname))
#                     shutil.move(os.path.join(dirpath,trainpic), os.path.join(train_path,dirname, trainpic))
#                     shutil.move(os.path.join(root_path,'Annotations',trainpic[:-4]+'.xml'), os.path.join(train_path,'Annotations',trainpic[:-4]+'.xml'))
#             else:
#                 continue