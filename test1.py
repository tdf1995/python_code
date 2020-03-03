import pytesseract
import cv2
from PIL import Image
image = cv2.imread(r"C:\Users\tdf\Desktop\1.png")
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
_, image = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2, 2))
image = cv2.erode(image,kernel)
cv2.imshow('1',image)
cv2.waitKey(0)

text = pytesseract.image_to_string(image)
print(text)