from blocks import blocks
from model.yolov3 import YoloV3, Default_YoloV3
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import time

def load_class_names(class_file):
    with open(class_file, 'r') as fin:
        class_names = fin.read().splitlines()
    return class_names

def load_model(log=False):
    if log:
        print("loading model...")
    t1 = time.time()
    model = YoloV3("yolo/config/yolov3_608.cfg", "yolo/config/yolov3_608.weights")
    t2 = time.time()

    if log:
        execution_time_ms = (t2 - t1) / 1000
        print(f"Model created!\nExecution Time: {execution_time_ms}ms")
    return model

def predict(image_file, model, input_size, log=False):
    t1 = time.time()
    class_names = load_class_names("yolo/config/class_names.cfg")
    image_raw = tf.image.decode_image(open(image_file, 'rb').read(), channels=3)
    image_raw_shape = image_raw.get_shape()
    image = tf.expand_dims(image_raw, 0)
    image = tf.image.resize(image, input_size)
    image = image / 255

    boxes, scores, classes, valid_detections = model.predict(image)

    image = draw_outputs(image_raw.numpy(), (boxes, scores, classes, valid_detections), class_names, image_raw_shape, image_file)
    t2 = time.time()
    if log:
        execution_time_ms = (t2 - t1) / 1000
        print(f"Prediction Made!\nExecution Time: {execution_time_ms}ms")
    return image

def draw_outputs(img, outputs, class_names, image_raw_shape, image_file):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])

    image = Image.open(image_file)

    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        text_pos = (int(x1y1[0] + image_raw_shape[1] * 0.01), int(x1y1[1] + image_raw_shape[0] * 0.02))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        box_points = (x1y1, x2y2)

        image = draw_image(image, box_points, class_names, int(classes[i]), objectness[i])

    return image

def draw_image(image, box_points, labels, label_index, confidence, width = 2):
    # compute color
    num_labels = len(labels)
    hue = int(360 / num_labels * (label_index + 1))
    color = f"hsl({hue}, 70%, 60%)"

    # unpack box points
    (box_x1, box_y1), (box_x2, box_y2) = box_points
    
    # compute font size
    font_size = 10 * width

    # get text dimensions
    label = labels[label_index]
    confidence = round(confidence * 100, 2)
    text = f"{label}|{confidence}%"
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
    text_width, text_height = font.getsize(text)

    # compute margin
    margin = 1 * width

    # compute label box points
    label_box_x1, label_box_y1 = box_points[0]
    label_box_x2 = label_box_x1 + text_width + 2 * margin
    label_box_y2 = label_box_y1 + text_height + 2 * margin
    label_box_points = [(label_box_x1, label_box_y1), (label_box_x2, label_box_y2)]

    # compute text points (top left only)
    text_x = label_box_x1 + margin
    text_y = label_box_y1 + margin
    text_points = (text_x, text_y)

    # draw bounding box
    draw = ImageDraw.Draw(image)
    draw.rectangle(box_points, outline = color, width = width)

    # draw label box
    draw.rectangle(label_box_points, fill = color, width = width)

    # draw text
    draw.text(text_points, text, font = font)

    return image

def main():
    model = YoloV3("config/yolov3_608.cfg", "../yolov3_608.weights")
    # model.save("config/tensorflow_default_model.h5")

    # model = Default_YoloV3()
    predict(model, "polar.jpg", (608, 608))

if __name__ == "__main__":
    main()