import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import blocks.blocks as b

# constants
_NUM_CLASSES = 80
_ANCHORS = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)], np.float32) / 416
_MASKS = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
_MAX_BOXES = 100
_IOU_THRESHOLD = 0.5
_SCORE_THRESHOLD = 0.5

def Default_YoloV3():
	return tf.saved_model.load("yolo/config")

def YoloV3(config_file, weights_file, training = False, log = False):
	# doesn't work please fix Dx
	# model = Default_YoloV3()
	model = create_yolov3_model(config_file, log, training = False)
	load_weights(model, weights_file, config_file, log)
	return model

def parse_config(config_file):
	blocks = []
	block_parameters = {}
	block_type = None
	with open(config_file, "r") as fin:
		for line in fin:
			# line is blank or a comment
			if line[0] == "#" or line == "\n":
				continue

			line = line.rstrip("\n")
			if line[0] == "[":
				if len(block_parameters) > 0:
					block_parameters["type"] = block_type
					blocks.append(block_parameters)
					block_parameters = {}
				block_type = line[1:-1]
			else:
				key, val = line.split("=")
				block_parameters[key.strip()] = val.strip()
		block_parameters["type"] = block_type
		blocks.append(block_parameters)
	return blocks

def create_yolov3_model(config_file, log, training = False, anchors = _ANCHORS, masks = _MASKS):
	blocks = parse_config(config_file)
	network_parameters = blocks[0]

	input_shape = (
		int(network_parameters["height"]),
		int(network_parameters["width"]),
		int(network_parameters["channels"])
	)
	
	x = inputs = tf.keras.layers.Input(shape = input_shape, name='input')

	outputs = []
	block_outputs = []

	for block_number, block in enumerate(blocks[1:]):
		block_type = block["type"]

		if block_type == "convolutional":
			filters = int(block["filters"])
			kernel_size = int(block["size"])
			strides = int(block["stride"])
			batch_norm = "batch_normalize" in block
			x = b.convolutional(x, filters, kernel_size, strides, batch_norm, block_number)
		
		elif block_type == "upsample":
			size = int(block["stride"])
			x = b.upsample(x, size, block_number)

		elif block_type == "shortcut":
			skip = int(block["from"])
			prev = block_outputs[block_number + skip]
			x = b.shortcut(x, prev, block_number)

		elif block_type == "route":
			layer_numbers = list(map(int, block["layers"].split(", ")))
			layers = []
			for layer_number in layer_numbers:
				if layer_number > 0:
					layers.append(block_outputs[layer_number])
				else:
					layers.append(block_outputs[block_number + layer_number])
			x = b.route(layers, block_number)

		elif block_type == "yolo":
			num_anchors = len(masks)
			num_classes = int(block["classes"])
			x = b.yolo(x, num_anchors, num_classes, block_number)
			outputs.append(x)

		block_outputs.append(x)

	if training:
		model = Model(inputs, outputs, name = "YoloV3")
		if log: 
			model.summary()
		return model

	# post-process outputs

	boxes0 = tf.keras.layers.Lambda(lambda x: generate_boxes(x, anchors[masks[0]], num_classes),
		name = f"GenerateBoxes_0")(outputs[0])
	
	boxes1 = tf.keras.layers.Lambda(lambda x: generate_boxes(x, anchors[masks[1]], num_classes),
		name = f"GenerateBoxes_1")(outputs[1])

	boxes2 = tf.keras.layers.Lambda(lambda x: generate_boxes(x, anchors[masks[2]], num_classes),
		name = f"GenerateBoxes_2")(outputs[2])
	
	outputs = tf.keras.layers.Lambda(lambda x: non_max_suppression(x),
					name='NonMaxSuppression')((boxes0[:3], boxes1[:3], boxes2[:3]))

	model = Model(inputs, outputs, name = "YoloV3")
	if log:
		model.summary()
	return model


def load_weights(model, weights_file, config_file, log):

	with open(weights_file, "rb") as fin:
		
		# skip first 5 header values
		np.fromfile(fin, dtype=np.int32, count=5)

		blocks = parse_config(config_file)

		for block_number, block in enumerate(blocks[1:]):
			block_type = block["type"]

			if (block_type == "convolutional"):
				conv_layer = model.get_layer(f"Convolutional_{block_number}")

				if log:
					print(conv_layer.name, conv_layer)

				num_filters = conv_layer.filters
				kernel_size = conv_layer.kernel_size[0]
				num_input_channels = conv_layer.input_shape[-1]
				has_batch_norm = "batch_normalize" in block

				if has_batch_norm:
					batch_norm_layer = model.get_layer(f"BatchNormalize_{block_number}")

					if log:
						print(batch_norm_layer.name, batch_norm_layer)

					batch_norm_weights = np.fromfile(fin, dtype=np.float32, count = 4 * num_filters)

					# darknet batch norm weights order: [beta, gamma, mean, variance]
					# tensorflow batch norm weights order: [gamma, beta, mean, variance]
					batch_norm_weights = batch_norm_weights.reshape((4, num_filters))[[1, 0, 2, 3]]
				else:
					conv_bias = np.fromfile(fin, dtype=np.float32, count=num_filters)
				
				# darknet conv weights shape: (num_output_channels, num_input_channels, height, width)
				conv_shape = (num_filters, num_input_channels, kernel_size, kernel_size)
				conv_weights = np.fromfile(fin, dtype=np.float32, count=np.product(conv_shape))

				# tensorflow conv weights shape: (height, width, num_input_channels, num_output_channels)
				conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

				if has_batch_norm:
					batch_norm_layer.set_weights(batch_norm_weights)
					conv_layer.set_weights([conv_weights])
				else:
					conv_layer.set_weights([conv_weights, conv_bias])

def generate_boxes(outputs, anchors, num_classes):
    # outputs: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...num_classes))
    grid_size = tf.shape(outputs)[1]
    box_xy, box_wh, objectness, class_probs = tf.split(
        outputs, (2, 2, 1, num_classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    outputs_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
        tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, outputs_box

def non_max_suppression(outputs, max_boxes = _MAX_BOXES, iou_threshold = _IOU_THRESHOLD, score_threshold = _SCORE_THRESHOLD):
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(
            scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class = max_boxes,
        max_total_size = max_boxes,
        iou_threshold = iou_threshold,
        score_threshold = score_threshold
    )

    return boxes, scores, classes, valid_detections
