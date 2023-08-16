import paddle

# Define the linear layer.
weight_attr = paddle.ParamAttr(
    name="weight",
    initializer=paddle.nn.initializer.Constant(value=0.5))
bias_attr = paddle.ParamAttr(
    name="bias",
    initializer=paddle.nn.initializer.Constant(value=1.0))
linear = paddle.nn.Linear(2, 4, weight_attr=weight_attr, bias_attr=bias_attr)
# linear.weight: [[0.5 0.5 0.5 0.5]
#                 [0.5 0.5 0.5 0.5]]
# linear.bias: [1. 1. 1. 1.]

x = paddle.randn((3, 2), dtype="float32")
# x: [[-0.32342386 -1.200079  ]
#     [ 0.7979031  -0.90978354]
#     [ 0.40597573  1.8095392 ]]
y = linear(x)
# y: [[0.23824859 0.23824859 0.23824859 0.23824859]
#     [0.9440598  0.9440598  0.9440598  0.9440598 ]
#     [2.1077576  2.1077576  2.1077576  2.1077576 ]]
