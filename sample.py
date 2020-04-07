#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import copy
import json

import cv2 as cv
import tensorflow as tf


def graph_load(path):
    config = tf.compat.v1.ConfigProto(
        gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))

    with tf.compat.v1.Graph().as_default() as net_graph:
        graph_data = tf.gfile.FastGFile(path, 'rb').read()
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(graph_data)
        tf.import_graph_def(graph_def, name='')

    sess = tf.compat.v1.Session(graph=net_graph, config=config)
    sess.graph.as_default()

    return sess


def session_run(sess, inp):
    out = sess.run(
        [
            sess.graph.get_tensor_by_name('num_detections:0'),
            sess.graph.get_tensor_by_name('detection_scores:0'),
            sess.graph.get_tensor_by_name('detection_boxes:0'),
            sess.graph.get_tensor_by_name('detection_classes:0')
        ],
        feed_dict={
            'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)
        },
    )
    return out


def main():
    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    # 手検出モデルロード #######################################################
    sess = graph_load('model/frozen_inference_graph.pb')
    label = open('model/labels.json', 'r')
    label = json.load(label)

    fps = 10

    while True:
        start_time = time.time()

        # カメラキャプチャ #####################################################
        ret, frame = cap.read()
        if not ret:
            continue
        frame_width, frame_height = frame.shape[1], frame.shape[0]
        debug_image = copy.deepcopy(frame)

        # 手検出実施 ###########################################################
        inp = cv.resize(frame, (512, 512))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        out = session_run(sess, inp)

        num_detections = int(out[0][0])
        for i in range(num_detections):
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            class_id = int(out[3][0][i])

            if score < 0.8:
                continue

            # 手検出結果可視化 #################################################
            x1, y1 = int(bbox[1] * frame_width), int(bbox[0] * frame_height)
            x2, y2 = int(bbox[3] * frame_width), int(bbox[2] * frame_height)

            cv.putText(debug_image,
                       label[class_id - 1] + ":" + '{:.3f}'.format(score),
                       (x1, y1 - 15), cv.FONT_HERSHEY_SIMPLEX, 1.0,
                       (0, 255, 0), 2, cv.LINE_AA)
            cv.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # FPS調整 #############################################################
        elapsed_time = time.time() - start_time
        sleep_time = max(0, ((1.0 / fps) - elapsed_time))
        time.sleep(sleep_time)

        # 画面反映 #############################################################
        cv.putText(
            debug_image,
            "Elapsed Time:" + '{:.1f}'.format(elapsed_time * 1000) + "ms",
            (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
        cv.imshow(' ', debug_image)
        cv.moveWindow(' ', 100, 100)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
