import cv2
import time
import numpy as np
import face_recognition
import random
import colorsys


def draw_bbox(image, bboxes, show_label=True):  # classes=read_class_names(cfg.YOLO.CLASSES)
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """
    classes = ['face']
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        x, y, width, height = coor
        c1, c2 = (x, y), (x+width, y+height)
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

    return image


video_path = r"/home/ubuntu/Desktop/Trash/youtube1.mp4"
cv_video_name = 'video'

input_size = 416

vid = cv2.VideoCapture(video_path)

fps = vid.get(cv2.CAP_PROP_FPS)
width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

out = cv2.VideoWriter('first_yt.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, (int(width), int(height)))
max_idx = 3000
idx =0
while True:
    idx +=1
    print(f'{idx} out of {max_idx}')

    return_value, frame = vid.read()
    if return_value:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame,(input_size, input_size))
    else:
        print('else finished')
        break

    if idx > max_idx:
        print('finished')
        break

    prev_time = time.time()

    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    faces = detector.detect_faces(frame)
    for face in faces:
        print(face)

    bboxes = [face['box'] + [face['confidence']] + [0] for face in faces]

    curr_time = time.time()
    exec_time = curr_time - prev_time

    image = draw_bbox(frame, bboxes)

    result = np.asarray(image)
    info = "time: %.2f ms" % (1000 * exec_time)
    print(info)
    # cv2.namedWindow(cv_video_name, cv2.WINDOW_AUTOSIZE)
    # result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #
    # cv2.imshow(cv_video_name, result)
    save_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    out.write(save_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()
print('destroyed')
