import Resize
import Resize_And_Padding
import random_crop
import bmp_to_jpg

def get_mypreprocessing(name):

    mypreprocessing_fn_map = {
        'Resize': Resize,
        'Resize_And_Padding': Resize_And_Padding,
        'random_crop': random_crop,
        'bmp_to_jpg':bmp_to_jpg
    }

    if name not in mypreprocessing_fn_map:
        raise ValueError('Preprocessing name [%s] was not recognized' % name)

    def mypreprocessing_fn(path, output_height, output_width):
        return mypreprocessing_fn_map[name].preprocess(
            path, output_height, output_width
        )
    return mypreprocessing_fn