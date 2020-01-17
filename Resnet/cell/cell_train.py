import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from keras.optimizers import SGD,Adam
from keras import Input
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense,GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from classification_models.keras import Classifiers

def get_model(TRAIN_NUM,TRAIN_SIZE):

    SeResNet50, preprocess_input = Classifiers.get('seresnet50')
    base_model = SeResNet50((TRAIN_SIZE, TRAIN_SIZE, 3),weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    prediction = Dense(TRAIN_NUM, activation='softmax', name='predict')(x)
    model = Model(inputs=base_model.input,
                  outputs=prediction)
    return model

TRAIN_SIZE = 300
TRAIN_NUM = 8
batch_size = 8

model = get_model(TRAIN_NUM,TRAIN_SIZE)
model.summary()

log_dir = 'logs'
train_dir = r'E:\细胞\x100\test1211\pic_all\x100'
val_dir = r'D:\python code\Resnet\cell\prediction'

train_gen = ImageDataGenerator(
                               rescale=1. / 255,
                               rotation_range=180,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               brightness_range=[0.7, 1.3],
                               zoom_range=0.1,
                               horizontal_flip=True,
                               vertical_flip=True,
                               validation_split=0.2
                               )

train_generator = train_gen.flow_from_directory(
    train_dir,
    target_size=(TRAIN_SIZE, TRAIN_SIZE),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_gen.flow_from_directory(
    train_dir,
    target_size=(TRAIN_SIZE, TRAIN_SIZE),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

model.compile(
    optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1.0, decay=1e-08,amsgrad=False),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

logging = TensorBoard(log_dir=log_dir)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)
checkpoint = ModelCheckpoint(log_dir + '/cell_9class_weight.h5', verbose=1, period=1,save_weights_only= True)

model.fit_generator(train_generator,
                    steps_per_epoch=train_generator.samples//batch_size, 
                    validation_data=val_generator,
                    validation_steps=val_generator.samples//batch_size,  
                    epochs=10,
                    initial_epoch=0,
                    callbacks=[logging, checkpoint, early_stopping, reduce_lr]
                    )

model.save('logs/cell_9class.h5')