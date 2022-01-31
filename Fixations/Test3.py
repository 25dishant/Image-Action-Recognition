# import mxnet as mx
import pdb
import torch
import torchvision

# # pdb.set_trace()
# layer = mx.gluon.rnn.LSTM(1024, 16)
# layer.initialize()
# input = mx.nd.random.uniform(shape=(16, 1024))
# input = input.reshape(16,1,1024)
# # by default zeros are used as begin state
# # output = layer(input)
# # manually specify begin state.
# h0 = mx.nd.random.uniform(shape=(16, 1, 1024))
# c0 = mx.nd.random.uniform(shape=(16, 1, 1024))
# output, hn = layer(input, [h0, c0])
# output = output.reshape(16,1024)
# print(output)

# input = mx.nd.random.uniform(shape=(1, 1024))
# print(input)
# input = input.reshape(16,1,1024)
# print(input)

# pdb.set_trace()
# rnn = torch.nn.LSTM(10, 20, 2)
# input = torch.randn(5, 3, 10)
# h0 = torch.randn(2, 3, 20)
# c0 = torch.randn(2, 3, 20)
# output, (hn, cn) = rnn(input, (h0, c0))


inp = torch.rand([1,1024,24,32])
gt_box = [[  1,  55, 305, 374],
        [ 35,  88, 321, 374]]

gt_box = torch.tensor(gt_box)
gt_box2 = torch.tensor(gt_box)

batch_num = torch.arange(gt_box.shape[0])
print(batch_num)
print(gt_box)
box = torch.Tensor([[0, 0, 0, 20, 20], [0, 0, 10, 30, 30]])
box = torch.column_stack((batch_num,gt_box))
print(box)
print(box.size())
roisn = torchvision.ops.roi_pool(inp,box,(7,7),spatial_scale=0.1)

print(roisn.shape)

X = torch.arange(16.).reshape(1, 1, 4, 4)




# pool = torchvision.ops.roi_pool(X, box, output_size=(2, 2), spatial_scale=0.1)

# print(pool)