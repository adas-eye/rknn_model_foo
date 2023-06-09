import cv2
import time
import threading
import numpy as np

from utils.coco_utils import COCO_test_helper
from rknn_excute import RKNN_model_container
from yolox_post_process import IMG_SIZE
from yolox_post_process import yolox_post_process
from yolox_post_process import draw_box

model_path = "./models/yoloxs_tk2_RK3588_i8.rknn"
rendering_size = (960, 540)


model = RKNN_model_container(model_path)
co_helper = COCO_test_helper(enable_letter_box=True)

list_before = []
list_after = []


class rknn_thread (threading.Thread):
    def run(self):
        while True:
            if len(list_before) == 0:
                time.sleep(1/1000)
            else:
                start_time = time.time()

                img_src = list_before[0]
                del list_before[0]

                # Due to rga init with (0,0,0), we using pad_color (0,0,0) instead of (114, 114, 114)
                img_4_rknn = cv2.cvtColor(cv2.resize(img_src.copy(), dsize=IMG_SIZE), cv2.COLOR_BGR2RGB)

                # rknn run
                outputs = model.run([img_4_rknn])

                # proprocess result
                anchors = [[[1.0, 1.0]]]*3
                outputs = [output.reshape([len(anchors[0]), -1]+list(output.shape[-2:])) for output in outputs]
                outputs = [np.transpose(output, (2, 3, 0, 1)) for output in outputs]
                boxes, classes, scores = yolox_post_process(outputs, anchors)

                # draw box
                draw_box(img_src, boxes, scores, classes)
                list_after.append(img_src)
                print("RKNN耗时: {:.3f}秒".format(time.time() - start_time))


class rendering_thread (threading.Thread):
    def __init__(self):
        super(rendering_thread, self).__init__()

    def run(self):
        while True:
            if len(list_after) == 0:
                time.sleep(1/1000)
            else:
                boxes_after_rknn = list_after[0]
                del list_after[0]
                cv2.imshow("RKNN", boxes_after_rknn)
                cv2.waitKey(1)


if __name__ == '__main__':
    vcap = cv2.VideoCapture("http://172.18.1.22:3310/live_stream/3/stream.flv")

    rknn_thread().start()
    rendering_thread().start()

    index_count = 0
    while (1):
        et, img = vcap.read()
        if img is not None:
            if len(list_before) < 250:
                list_before.append(cv2.resize(img, dsize=rendering_size))
