import mypreprocessing_factory
import numpy as np
'''
'Resize','Resize_And_Padding','random_crop','bmp_to_jpg'
'''
preprocessing_name = 'Resize'
root_path = r'E:\ocr\dataset\png'
output_height = 150
output_width = 375
if __name__ =='__main__':
    image_preprocessing_fn = mypreprocessing_factory.get_mypreprocessing(
        preprocessing_name
    )
    image_preprocessing_fn(root_path,output_height,output_width)