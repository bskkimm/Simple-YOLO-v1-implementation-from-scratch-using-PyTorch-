import os

DATA_PATH = 'data'


BATCH_SIZE = 48 # originally 64, but reduced to fit in memory
TRAIN_SCHEDULER = {'schedule1': {'epochs1': 75, 'lr1': 1E-3},
                   'schedule2': {'epochs2': 30, 'lr2': 1E-4},
                   'schedule3': {'epochs3': 30, 'lr3': 1E-5}
                  }



LEARNING_RATE = 1E-4 # Initial learning rate

EPSILON = 1E-6
IMAGE_SIZE = (448, 448)


S = 7       # Divide each image into a SxS grid
B = 2       # Number of bounding boxes to predict
C = 20      # Number of classes in the dataset

classes = {
  "horse": 0,
  "person": 1,
  "bottle": 2,
  "dog": 3,
  "tvmonitor": 4,
  "car": 5,
  "aeroplane": 6,
  "bicycle": 7,
  "boat": 8,
  "chair": 9,
  "diningtable": 10,
  "pottedplant": 11,
  "train": 12,
  "cat": 13,
  "sofa": 14,
  "bird": 15,
  "sheep": 16,
  "motorbike": 17,
  "bus": 18,
  "cow": 19
}