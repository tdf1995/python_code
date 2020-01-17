'''
将字典写入xml
'''
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

def read_xml(in_path):
    """读取并解析xml文件
       in_path: xml路径
       return: tree"""
    tree = ET.parse(in_path)
    return tree

def creat_dict(root):
    """xml生成为dict：，
    将tree中个节点添加到list中，将list转换为字典dict_init
    叠加生成多层字典dict_new"""
    dict_new = {}
    for key, valu in enumerate(root):
        dict_init = {}
        list_init = []
        for item in valu:
            list_init.append([item.tag, item.text])
            for lists in list_init:
                dict_init[lists[0]] = lists[1]
        dict_new[key] = dict_init
    return dict_new

def dict_to_xml(input_dict,root_tag,node_tag):
    """ 定义根节点root_tag，定义第二层节点node_tag
    第三层中将字典中键值对对应参数名和值
       return: xml的tree结构 """
    root_name = ET.Element(root_tag)
    for (k, v) in input_dict.items():
        node_name = ET.SubElement(root_name, node_tag)
        for (key, val) in sorted(v.items(), key=lambda e:e[0], reverse=True):
            key = ET.SubElement(node_name, key)
            key.text = val
    return root_name

def out_xml(root):
    """格式化root转换为xml文件"""
    rough_string = ET.tostring(root, 'utf-8')
    reared_content = minidom.parseString(rough_string)
    with open(out_file, 'w+') as fs:
        reared_content.writexml(fs, addindent=" ", newl="\n", encoding="utf-8")
    return True

if __name__ == '__main__':
    in_files = r"D:\baspool_read.xml"
    out_file = r"D:\baspool_out.xml"
    tree = read_xml(in_files)
    node_new = creat_dict(tree.getroot())  # 将xml转换为dict
    root = dict_to_xml(node_new, "baspools", "bas")  # 将dict转换为xml
    out_xml(root)     # 输出xml到out_files
