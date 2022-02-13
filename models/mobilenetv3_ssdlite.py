import keras.layers as KL
import keras.backend as K

import numpy as np
from keras import layers
from keras.models import Model
from keras.layers import Input, Lambda, Activation, Conv2D, DepthwiseConv2D, Reshape, Concatenate, BatchNormalization, ReLU
from layers.AnchorBoxesLayer import AnchorBoxes
from layers.DecodeDetectionsLayer import DecodeDetections
from layers.DecodeDetectionsFastLayer import DecodeDetectionsFast



def predict_block(inputs, out_channel, sym, id):
    name = 'ssd_' + sym + '{}'.format(id)
    x = DepthwiseConv2D(kernel_size=3, strides=1,
                           activation=None, use_bias=False, padding='same', name=name + '_dw_conv')(inputs)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=name + '_dw_bn')(x)
    x = ReLU(6., name=name + '_dw_relu')(x)

    x = Conv2D(out_channel, kernel_size=1, padding='same', use_bias=False,
                  activation=None, name=name + 'conv2')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=name + 'conv2_bn')(x)
    return x



def correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.

    # Returns
        A tuple.
    """
    img_dim = 2 if K.image_data_format() == 'channels_first' else 1
    input_size = K.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


def relu(x):
    return layers.ReLU()(x)


def hard_sigmoid(x):
    return layers.ReLU(6.)(Lambda(lambda x: x + 3.)(x))


def hard_swish(x):
    # TODO: 
    return Lambda(lambda x: x * 1. / 6.)(layers.Multiply()([x, hard_sigmoid(x)]))


def _depth(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _se_block(inputs, filters, se_ratio, prefix):
    x = layers.AveragePooling2D(
        pool_size=K.int_shape(inputs)[1:3], name=prefix + 'squeeze_excite/AvgPool')(inputs)
    x = layers.Conv2D(
        _depth(filters * se_ratio),
        kernel_size=1,
        padding='same',
        name=prefix + 'squeeze_excite/Conv')(x)
    x = layers.ReLU(name=prefix + 'squeeze_excite/Relu')(x)
    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding='same',
        name=prefix + 'squeeze_excite/Conv_1')(x)
    x = hard_sigmoid(x)
    x = layers.Multiply(name=prefix + 'squeeze_excite/Mul')([inputs, x])
    return x


def _inverted_res_block(x, expansion, filters, kernel_size, stride, se_ratio,
                        activation, block_id):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    shortcut = x
    prefix = 'expanded_conv/'
    infilters = K.int_shape(x)[channel_axis]
    if block_id:
        # Expand
        prefix = 'expanded_conv_{}/'.format(block_id)
        x = layers.Conv2D(
            _depth(infilters * expansion),
            kernel_size=1,
            padding='same',
            use_bias=False,
            name=prefix + 'expand')(
            x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + 'expand/BatchNorm')(
            x)
        x = activation(x)

    if stride == 2:
        x = layers.ZeroPadding2D(
            padding=correct_pad(x, kernel_size),
            name=prefix + 'depthwise/pad')(
            x)
    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=stride,
        padding='same' if stride == 1 else 'valid',
        use_bias=False,
        name=prefix + 'depthwise')(
        x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'depthwise/BatchNorm')(
        x)
    x = activation(x)

    if se_ratio:
        x = _se_block(x, _depth(infilters * expansion), se_ratio, prefix)

    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding='same',
        use_bias=False,
        name=prefix + 'project')(
        x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'project/BatchNorm')(
        x)

    if stride == 1 and infilters == filters:
        x = layers.Add(name=prefix + 'Add')([shortcut, x])
    return x


def _followed_down_sample_block(inputs, conv_out_channel, sep_out_channel, id):
    name = 'ssd_{}'.format(id)
    x = KL.Conv2D(conv_out_channel, kernel_size=1, padding='same', use_bias=False,
                  activation=None, name=name + '_conv')(inputs)
    x = KL.BatchNormalization(epsilon=1e-3, momentum=0.999, name=name + '_conv_bn')(x)
    x = KL.ReLU(6., name=name + '_conv_relu')(x)

    x = KL.ZeroPadding2D(padding=correct_pad(x, 3), name=name + '_dw_pad')(x)
    x = KL.DepthwiseConv2D(kernel_size=3, strides=2,
                           activation=None, use_bias=False, padding='valid', name=name + '_dw_conv')(x)
    x = KL.BatchNormalization(epsilon=1e-3, momentum=0.999, name=name + '_dw_bn')(x)
    x = KL.ReLU(6., name=name + '_dw_relu')(x)

    x = KL.Conv2D(sep_out_channel, kernel_size=1, padding='same', use_bias=False,
                  activation=None, name=name + '_conv2')(x)
    x = KL.BatchNormalization(epsilon=1e-3, momentum=0.999, name=name + '_conv2_bn')(x)
    x = KL.ReLU(6., name=name + '_conv2_relu')(x)
    return x


def mobilenet_v3_extractor(inputs, model_type='large'):
    alpha = 1.0
    kernel = 5
    activation = hard_swish
    se_ratio = 0.25
    input_shape = (None, None, 3)

    if K.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]
    if rows and cols and (rows < 32 or cols < 32):
        raise ValueError(f'Input size must be at least 32x32; Received `input_shape={input_shape}`')

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    x = layers.ZeroPadding2D(padding=correct_pad(inputs, 3), name='bbn_stage1_block1_pad')(inputs)
    x = layers.Conv2D(
        16,
        kernel_size=3,
        strides=(2, 2),
        padding='same',
        use_bias=False,
        name='Conv')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3,
        momentum=0.999, name='Conv/BatchNorm')(x)
    x = activation(x)

    def depth(d):
        return _depth(d * alpha)
    
    if model_type == 'Small':
        last_point_ch = 1024
        x = _inverted_res_block(x, 1, depth(16), 3, 2, se_ratio, relu, 0)
        x = _inverted_res_block(x, 72. / 16, depth(24), 3, 2, None, relu, 1)
        x = _inverted_res_block(x, 88. / 24, depth(24), 3, 1, None, relu, 2)
        x = _inverted_res_block(x, 4, depth(40), kernel, 2, se_ratio, activation, 3)
        x = _inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 4)
        x = _inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 5)
        x = _inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 6)
        x = _inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 7)
        link1 = x = _inverted_res_block(x, 6, depth(96), kernel, 2, se_ratio, activation, 8)
        x = _inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation, 9)
        x = _inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation, 10)
    else:
        last_point_ch = 1280
        x = _inverted_res_block(x, 1, depth(16), 3, 1, None, relu, 0)
        x = _inverted_res_block(x, 4, depth(24), 3, 2, None, relu, 1)
        x = _inverted_res_block(x, 3, depth(24), 3, 1, None, relu, 2)
        x = _inverted_res_block(x, 3, depth(40), kernel, 2, se_ratio, relu, 3)
        x = _inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, relu, 4)
        x = _inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, relu, 5)
        x = _inverted_res_block(x, 6, depth(80), 3, 2, None, activation, 6)
        x = _inverted_res_block(x, 2.5, depth(80), 3, 1, None, activation, 7)
        x = _inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 8)
        x = _inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 9)
        x = _inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 10)
        x = _inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 11)
        link1 = x = _inverted_res_block(x, 6, depth(160), kernel, 2, se_ratio, activation, 12)
        x = _inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio, activation, 13)
        x = _inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio, activation, 14)

    last_conv_ch = _depth(K.int_shape(x)[channel_axis] * 6)
    if alpha > 1.0:
        last_point_ch = _depth(last_point_ch * alpha)
    x = layers.Conv2D(
        last_conv_ch,
        kernel_size=1,
        padding='same',
        use_bias=False,
        name='Conv_1')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3,
        momentum=0.999, name='Conv_1/BatchNorm')(x)
    
    link2 = x = activation(x)
    link3 = x = _followed_down_sample_block(x, 256, 512, 3)
    link4 = x = _followed_down_sample_block(x, 128, 256, 4)
    link5 = x = _followed_down_sample_block(x, 128, 256, 5)
    link6 = x = _followed_down_sample_block(x, 64, 128, 6)
    links = [link1, link2, link3, link4, link5, link6]
    return links


def mobilenet_v3_ssdlite(image_size,
            n_classes,
            mode='training',
            l2_regularization=0.0005,
            min_scale=None,
            max_scale=None,
            scales=None,
            aspect_ratios_global=None,
            aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                     [1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5]],
            two_boxes_for_ar1=True,
            steps=[8, 16, 32, 64, 100, 300],
            offsets=None,
            clip_boxes=False,
            variances=[0.1, 0.1, 0.2, 0.2],
            coords='centroids',
            normalize_coords=True,
            subtract_mean=[123, 117, 104],
            divide_by_stddev=None,
            swap_channels=[2, 1, 0],
            confidence_thresh=0.01,
            iou_threshold=0.45,
            top_k=200,
            nms_max_output_size=400,
            return_predictor_sizes=False,
            model_type='Large'):

    n_predictor_layers = 6  # The number of predictor conv layers in the network is 6 for the original SSD300.
    n_classes += 1  # Account for the background class.
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    ############################################################################
    # Get a few exceptions out of the way.
    ############################################################################

    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError(
            "`aspect_ratios_global` and `aspect_ratios_per_layer` \
            cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError(
                "It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, \
                but len(aspect_ratios_per_layer) == {}.".format(
                    n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers + 1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(
                n_predictor_layers + 1, len(scales)))
    # If no explicit list of scaling factors was passed,
    # compute the list of scaling factors from `min_scale` and `max_scale`
    else:
        scales = np.linspace(min_scale, max_scale, n_predictor_layers + 1)

    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")

    ############################################################################
    # Compute the anchor box parameters.
    ############################################################################

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1)  # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))
    # If only a global aspect ratio list was passed,
    # then the number of boxes is the same for each predictor layer
    else:
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    ############################################################################
    # Define functions for the Lambda layers below.
    ############################################################################

    def identity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    def input_channel_swap(tensor):
        if len(swap_channels) == 3:
            return K.stack(
                [tensor[..., swap_channels[0]], tensor[..., swap_channels[1]], tensor[..., swap_channels[2]]], axis=-1)
        elif len(swap_channels) == 4:
            return K.stack([tensor[..., swap_channels[0]], tensor[..., swap_channels[1]], tensor[..., swap_channels[2]],
                            tensor[..., swap_channels[3]]], axis=-1)

    ############################################################################
    # Build the network.
    ############################################################################

    x = Input(shape=(img_height, img_width, img_channels))

    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(x)

    if subtract_mean is not None:
        x1 = Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels),
                    name='input_mean_normalization')(x1)
    if divide_by_stddev is not None:
        x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels),
                    name='input_stddev_normalization')(x1)
    if swap_channels:
        x1 = Lambda(input_channel_swap, output_shape=(img_height, img_width, img_channels), 
                    name='input_channel_swap')(x1)

    links = mobilenet_v3_extractor(x1, model_type)

    link1_cls = predict_block(links[0], n_boxes[0] * n_classes, 'cls', 1)
    link2_cls = predict_block(links[1], n_boxes[1] * n_classes, 'cls', 2)
    link3_cls = predict_block(links[2], n_boxes[2] * n_classes, 'cls', 3)
    link4_cls = predict_block(links[3], n_boxes[3] * n_classes, 'cls', 4)
    link5_cls = predict_block(links[4], n_boxes[4] * n_classes, 'cls', 5)
    link6_cls = predict_block(links[5], n_boxes[5] * n_classes, 'cls', 6)

    link1_box = predict_block(links[0], n_boxes[0] * 4, 'box', 1)
    link2_box = predict_block(links[1], n_boxes[1] * 4, 'box', 2)
    link3_box = predict_block(links[2], n_boxes[2] * 4, 'box', 3)
    link4_box = predict_block(links[3], n_boxes[3] * 4, 'box', 4)
    link5_box = predict_block(links[4], n_boxes[4] * 4, 'box', 5)
    link6_box = predict_block(links[5], n_boxes[5] * 4, 'box', 6)

    priorbox1 = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1],
                                             aspect_ratios=aspect_ratios[0],
                                             two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0],
                                             this_offsets=offsets[0], clip_boxes=clip_boxes,
                                             variances=variances, coords=coords, normalize_coords=normalize_coords,
                                             name='ssd_priorbox_1')(link1_box)
    priorbox2 = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2],
                                    aspect_ratios=aspect_ratios[1],
                                    two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1],
                                    clip_boxes=clip_boxes,
                                    variances=variances, coords=coords, normalize_coords=normalize_coords,
                                    name='ssd_priorbox_2')(link2_box)
    priorbox3 = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3],
                                        aspect_ratios=aspect_ratios[2],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2],
                                        this_offsets=offsets[2], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords,
                                        name='ssd_priorbox_3')(link3_box)
    priorbox4 = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4],
                                        aspect_ratios=aspect_ratios[3],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3],
                                        this_offsets=offsets[3], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords,
                                        name='ssd_priorbox_4')(link4_box)
    priorbox5 = AnchorBoxes(img_height, img_width, this_scale=scales[4], next_scale=scales[5],
                                        aspect_ratios=aspect_ratios[4],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[4],
                                        this_offsets=offsets[4], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords,
                                        name='ssd_priorbox_5')(link5_box)
    priorbox6 = AnchorBoxes(img_height, img_width, this_scale=scales[5], next_scale=scales[6],
                                        aspect_ratios=aspect_ratios[5],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[5],
                                        this_offsets=offsets[5], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords,
                                        name='ssd_priorbox_6')(link6_box)

    # Reshape
    cls1_reshape = Reshape((-1, n_classes), name='ssd_cls1_reshape')(link1_cls)
    cls2_reshape = Reshape((-1, n_classes), name='ssd_cls2_reshape')(link2_cls)
    cls3_reshape = Reshape((-1, n_classes), name='ssd_cls3_reshape')(link3_cls)
    cls4_reshape = Reshape((-1, n_classes), name='ssd_cls4_reshape')(link4_cls)
    cls5_reshape = Reshape((-1, n_classes), name='ssd_cls5_reshape')(link5_cls)
    cls6_reshape = Reshape((-1, n_classes), name='ssd_cls6_reshape')(link6_cls)

    box1_reshape = Reshape((-1, 4), name='ssd_box1_reshape')(link1_box)
    box2_reshape = Reshape((-1, 4), name='ssd_box2_reshape')(link2_box)
    box3_reshape = Reshape((-1, 4), name='ssd_box3_reshape')(link3_box)
    box4_reshape = Reshape((-1, 4), name='ssd_box4_reshape')(link4_box)
    box5_reshape = Reshape((-1, 4), name='ssd_box5_reshape')(link5_box)
    box6_reshape = Reshape((-1, 4), name='ssd_box6_reshape')(link6_box)

    priorbox1_reshape = Reshape((-1, 8), name='ssd_priorbox1_reshape')(priorbox1)
    priorbox2_reshape = Reshape((-1, 8), name='ssd_priorbox2_reshape')(priorbox2)
    priorbox3_reshape = Reshape((-1, 8), name='ssd_priorbox3_reshape')(priorbox3)
    priorbox4_reshape = Reshape((-1, 8), name='ssd_priorbox4_reshape')(priorbox4)
    priorbox5_reshape = Reshape((-1, 8), name='ssd_priorbox5_reshape')(priorbox5)
    priorbox6_reshape = Reshape((-1, 8), name='ssd_priorbox6_reshape')(priorbox6)

    cls = Concatenate(axis=1, name='ssd_cls')(
        [cls1_reshape, cls2_reshape, cls3_reshape, cls4_reshape, cls5_reshape, cls6_reshape])

    box = Concatenate(axis=1, name='ssd_box')(
        [box1_reshape, box2_reshape, box3_reshape, box4_reshape, box5_reshape, box6_reshape])

    priorbox = Concatenate(axis=1, name='ssd_priorbox')(
        [priorbox1_reshape, priorbox2_reshape, priorbox3_reshape,
         priorbox4_reshape, priorbox5_reshape, priorbox6_reshape])

    cls = Activation('softmax', name='ssd_mbox_conf_softmax')(cls)

    predictions = Concatenate(axis=2, name='ssd_predictions')([cls, box, priorbox])

    if mode == 'training':
        model = Model(inputs=x, outputs=predictions)
    elif mode == 'inference':
        decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
                                               iou_threshold=iou_threshold,
                                               top_k=top_k,
                                               nms_max_output_size=nms_max_output_size,
                                               coords=coords,
                                               normalize_coords=normalize_coords,
                                               img_height=img_height,
                                               img_width=img_width,
                                               name='ssd_decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    elif mode == 'inference_fast':
        decoded_predictions = DecodeDetectionsFast(confidence_thresh=confidence_thresh,
                                                   iou_threshold=iou_threshold,
                                                   top_k=top_k,
                                                   nms_max_output_size=nms_max_output_size,
                                                   coords=coords,
                                                   normalize_coords=normalize_coords,
                                                   img_height=img_height,
                                                   img_width=img_width,
                                                   name='ssd_decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    else:
        raise ValueError(
            "`mode` must be one of 'training', 'inference' or 'inference_fast', but received '{}'.".format(mode))

    if return_predictor_sizes:
        predictor_sizes = np.array(
            [cls[0]._keras_shape[1:3], cls[1]._keras_shape[1:3], cls[2]._keras_shape[1:3],
             cls[3]._keras_shape[1:3], cls[4]._keras_shape[1:3], cls[5]._keras_shape[1:3]])
        return model, predictor_sizes
    else:
        return model

