import os
root_path = r'C:\Users\tdf\Desktop\已标注'
target_path = r'\\192.168.1.251\ssd-研发部\项目工作目录\菜品识别项目\蒸浏记\原始数据合并\20190903_第一批_训练集_未分割'

root_files = os.listdir(root_path)
for root_file in root_files:
    target_files = os.listdir(target_path)
    if root_file in target_files:
        os.remove(os.path.join(target_path, root_file))