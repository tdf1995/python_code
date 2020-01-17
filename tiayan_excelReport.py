import xlwt
import xlrd
import re

error_report = open('C:/Users/Administrator/Desktop/error report.txt',encoding='UTF-8')

lines = error_report.readlines()

report = xlwt.Workbook(encoding = 'utf-8', style_compression=0)
sheet = report.add_sheet('tiaoyan_report')
sheet.write(0,0,'统计项')
sheet.write(1,0,'分类总数')
sheet.write(2,0,'误分类数')
sheet.write(3,0,'识别准确率')
sheet.write(4,0,'误识别率前三')
i = 0
for line in lines:
    if line == '\n':
        continue

    a = re.split('[的:：: :\n:]',line)#a[0]:类别，a[2]:误识别率,a[4]:错误数，a[6]：总数
    if not a[1] =='误识别率为':
        continue
    i = i + 1
    sheet.write(0,i,a[0])
    sheet.write(1,i,a[6])
    sheet.write(2,i,a[4])
    sheet.write(3,i,a[2])
    sheet.write(4,i,a[9])
report.save('error_report.xls')