'''
这个脚本用来分割数据集
产生训练和测试两个数据集
比例可调
'''
import  os,random,shutil
from os import walk
import glob

test_percent = 0.1 # 训练集的比例
root_path = r'F:\models-master\cell\分类\train_set'
test_path = r'F:\models-master\cell\分类\test_set'
if __name__ == '__main__':
    for (_,dirnames,_)in walk(root_path):
        for dirname in dirnames:
            path = os.path.join(root_path, dirname)
            pic_list = glob.glob(path+'/*.jpg')
            dirpath = path
            # pathdir = os.listdir(dirpath)
            num = len(pic_list)
            test_num = int(num*test_percent)
            testset = random.sample(pic_list, test_num)
            for trainpic in testset:
                if not os.path.exists(os.path.join(test_path, dirname)):
                    os.makedirs(os.path.join(test_path, dirname))
                shutil.move(trainpic, os.path.join(test_path,dirname, os.path.basename(trainpic)))
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