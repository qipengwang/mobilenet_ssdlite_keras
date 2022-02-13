# # Copyright 2020 The TensorFlow Authors. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ==============================================================================
# # pylint: disable=invalid-name
# # pylint: disable=missing-function-docstring
# """MobileNet v3 models for Keras."""

# from keras import backend
# from keras import models
# from keras.applications import imagenet_utils
# from keras import layers
# from keras.utils import data_utils
# from keras.utils import layer_utils
# from tensorflow.python.platform import tf_logging as logging
# from tensorflow.python.util.tf_export import keras_export



# def MobileNetV3(model_type='large', input_shape=None):
#     alpha = 1.0
#     # If input_shape is None and input_tensor is None using standard shape
#     if input_shape is None:
#         input_shape = (None, None, 3)

#     if backend.image_data_format() == 'channels_last':
#         row_axis, col_axis = (0, 1)
#     else:
#         row_axis, col_axis = (1, 2)
#     rows = input_shape[row_axis]
#     cols = input_shape[col_axis]
#     if rows and cols and (rows < 32 or cols < 32):
#         raise ValueError(f'Input size must be at least 32x32; Received `input_shape={input_shape}`')


#     img_input = layers.Input(shape=input_shape)

#     channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

#     kernel = 5
#     activation = hard_swish
#     se_ratio = 0.25

#     x = layers.Conv2D(
#         16,
#         kernel_size=3,
#         strides=(2, 2),
#         padding='same',
#         use_bias=False,
#         name='Conv')(img_input)
#     x = layers.BatchNormalization(
#         axis=channel_axis, epsilon=1e-3,
#         momentum=0.999, name='Conv/BatchNorm')(x)
#     x = activation(x)

#     # x = stack_fn(x, kernel, activation, se_ratio)
#     def depth(d):
#         return _depth(d * alpha)
    
#     if model_type == 'small':
#         last_point_ch = 1024
#         x = _inverted_res_block(x, 1, depth(16), 3, 2, se_ratio, relu, 0)
#         x = _inverted_res_block(x, 72. / 16, depth(24), 3, 2, None, relu, 1)
#         x = _inverted_res_block(x, 88. / 24, depth(24), 3, 1, None, relu, 2)
#         x = _inverted_res_block(x, 4, depth(40), kernel, 2, se_ratio, activation, 3)
#         x = _inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 4)
#         x = _inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 5)
#         x = _inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 6)
#         x = _inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 7)
#         x = _inverted_res_block(x, 6, depth(96), kernel, 2, se_ratio, activation, 8)
#         x = _inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation, 9)
#         x = _inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation, 10)
#     else:
#         last_point_ch = 1280
#         x = _inverted_res_block(x, 1, depth(16), 3, 1, None, relu, 0)
#         x = _inverted_res_block(x, 4, depth(24), 3, 2, None, relu, 1)
#         x = _inverted_res_block(x, 3, depth(24), 3, 1, None, relu, 2)
#         x = _inverted_res_block(x, 3, depth(40), kernel, 2, se_ratio, relu, 3)
#         x = _inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, relu, 4)
#         x = _inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, relu, 5)
#         x = _inverted_res_block(x, 6, depth(80), 3, 2, None, activation, 6)
#         x = _inverted_res_block(x, 2.5, depth(80), 3, 1, None, activation, 7)
#         x = _inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 8)
#         x = _inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 9)
#         x = _inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 10)
#         x = _inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 11)
#         x = _inverted_res_block(x, 6, depth(160), kernel, 2, se_ratio, activation, 12)
#         x = _inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio, activation, 13)
#         x = _inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio, activation, 14)
    
#     link3 = x = _followed_down_sample_block(x, 256, 512, 3)

#     link4 = x = _followed_down_sample_block(x, 128, 256, 4)

#     link5 = x = _followed_down_sample_block(x, 128, 256, 5)

#     link6 = x = _followed_down_sample_block(x, 64, 128, 6)

#     links = [link1, link2, link3, link4, link5, link6]

#     last_conv_ch = _depth(backend.int_shape(x)[channel_axis] * 6)
#     if alpha > 1.0:
#         last_point_ch = _depth(last_point_ch * alpha)
#     x = layers.Conv2D(
#         last_conv_ch,
#         kernel_size=1,
#         padding='same',
#         use_bias=False,
#         name='Conv_1')(x)
#     x = layers.BatchNormalization(
#         axis=channel_axis, epsilon=1e-3,
#         momentum=0.999, name='Conv_1/BatchNorm')(x)
#     x = activation(x)

#     inputs = img_input

#     # Create model.
#     model = models.Model(inputs, x, name='MobilenetV3' + model_type)

#     return model


# def relu(x):
#     return layers.ReLU()(x)


# def hard_sigmoid(x):
#     return layers.ReLU(6.)(x + 3.) * (1. / 6.)


# def hard_swish(x):
#     return layers.Multiply()([x, hard_sigmoid(x)])



# def _depth(v, divisor=8, min_value=None):
#     if min_value is None:
#         min_value = divisor
#     new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
#     # Make sure that round down does not go down by more than 10%.
#     if new_v < 0.9 * v:
#         new_v += divisor
#     return new_v


# def _se_block(inputs, filters, se_ratio, prefix):
#     x = layers.GlobalAveragePooling2D(
#         keepdims=True, name=prefix + 'squeeze_excite/AvgPool')(
#         inputs)
#     x = layers.Conv2D(
#         _depth(filters * se_ratio),
#         kernel_size=1,
#         padding='same',
#         name=prefix + 'squeeze_excite/Conv')(
#         x)
#     x = layers.ReLU(name=prefix + 'squeeze_excite/Relu')(x)
#     x = layers.Conv2D(
#         filters,
#         kernel_size=1,
#         padding='same',
#         name=prefix + 'squeeze_excite/Conv_1')(
#         x)
#     x = hard_sigmoid(x)
#     x = layers.Multiply(name=prefix + 'squeeze_excite/Mul')([inputs, x])
#     return x


# def _inverted_res_block(x, expansion, filters, kernel_size, stride, se_ratio,
#                         activation, block_id):
#     channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
#     shortcut = x
#     prefix = 'expanded_conv/'
#     infilters = backend.int_shape(x)[channel_axis]
#     if block_id:
#         # Expand
#         prefix = 'expanded_conv_{}/'.format(block_id)
#         x = layers.Conv2D(
#             _depth(infilters * expansion),
#             kernel_size=1,
#             padding='same',
#             use_bias=False,
#             name=prefix + 'expand')(
#             x)
#         x = layers.BatchNormalization(
#             axis=channel_axis,
#             epsilon=1e-3,
#             momentum=0.999,
#             name=prefix + 'expand/BatchNorm')(
#             x)
#         x = activation(x)

#     if stride == 2:
#         x = layers.ZeroPadding2D(
#             padding=imagenet_utils.correct_pad(x, kernel_size),
#             name=prefix + 'depthwise/pad')(
#             x)
#     x = layers.DepthwiseConv2D(
#         kernel_size,
#         strides=stride,
#         padding='same' if stride == 1 else 'valid',
#         use_bias=False,
#         name=prefix + 'depthwise')(
#         x)
#     x = layers.BatchNormalization(
#         axis=channel_axis,
#         epsilon=1e-3,
#         momentum=0.999,
#         name=prefix + 'depthwise/BatchNorm')(
#         x)
#     x = activation(x)

#     if se_ratio:
#         x = _se_block(x, _depth(infilters * expansion), se_ratio, prefix)

#     x = layers.Conv2D(
#         filters,
#         kernel_size=1,
#         padding='same',
#         use_bias=False,
#         name=prefix + 'project')(
#         x)
#     x = layers.BatchNormalization(
#         axis=channel_axis,
#         epsilon=1e-3,
#         momentum=0.999,
#         name=prefix + 'project/BatchNorm')(
#         x)

#     if stride == 1 and infilters == filters:
#         x = layers.Add(name=prefix + 'Add')([shortcut, x])
#     return x




