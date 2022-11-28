import cv2
import time
import threading
import numpy as np

from utils.coco_utils import COCO_test_helper
from rknn_excute import RKNN_model_container
from yolovx_post_process import IMG_SIZE
from yolovx_post_process import yolov5_post_process

model_path = "./yoloxm_tk2_RK3588_i8.rknn"
CLASSES = (
    "person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
    "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
    "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife ", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet ",
    "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ", "oven ", "toaster", "sink",
    "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush ")


def draw(image, boxes, scores, classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    height, width = image.shape[:2]
    cw = width/IMG_SIZE[0]
    ch = height/IMG_SIZE[1]
    if boxes is not None:
        for box, score, cl in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            x1 = int(x1*cw)
            y1 = int(y1*ch)
            x2 = int(x2*cw)
            y2 = int(y2*ch)

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score), (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


model = RKNN_model_container(model_path)
co_helper = COCO_test_helper(enable_letter_box=True)

list_before = []
list_after = []


class rknnThread (threading.Thread):
    def run(self):
        while True:
            if len(list_before) == 0:
                time.sleep(1/1000)
            else:
                img_src = list_before[0]
                del list_before[0]

                # Due to rga init with (0,0,0), we using pad_color (0,0,0) instead of (114, 114, 114)
                start_time = time.time()
                resized = cv2.resize(img_src.copy(), dsize=IMG_SIZE)
                img_4_rknn = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                print("CVTColor耗时: {:.3f}秒".format(time.time() - start_time))

                # rknn run
                start_time = time.time()
                outputs = model.run([img_4_rknn])
                print("RKNN耗时: {:.3f}秒".format(time.time() - start_time))

                # proprocess result
                start_time = time.time()
                anchors = [[[1.0, 1.0]]]*3
                outputs = [output.reshape([len(anchors[0]), -1]+list(output.shape[-2:])) for output in outputs]
                outputs = [np.transpose(output, (2, 3, 0, 1)) for output in outputs]
                boxes, classes, scores = yolov5_post_process(outputs, anchors)
                print("POST耗时: {:.3f}秒".format(time.time() - start_time))

                draw(img_src, boxes, scores, classes)
                list_after.append(img_src)


if __name__ == '__main__':

    rknnThread(group=None).start()

    vcap = cv2.VideoCapture("http://172.18.1.22:3310/live_stream/2/stream.flv?profile_id=0")

    index_count = 0
    while (1):
        et, img_src = vcap.read()
        img_src_resized = cv2.resize(img_src, dsize=(960, 540))

        index_count = index_count+1
        if index_count == 25:
            index_count = 0
            list_before.append(img_src_resized.copy())

        if len(list_after) != 0:
            boxes_after_rknn = list_after[0]
            del list_after[0]
            cv2.imshow("RKNN", boxes_after_rknn)
            cv2.waitKey(1)

        cv2.imshow("LIVE", img_src_resized)
        cv2.waitKey(1)
