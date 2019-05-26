
import torch
import torchvision

class TestFun(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input, v_th=[]):
        return g.op("TestFun", input, v_th_f=v_th)

    @staticmethod
    def forward(ctx, input, v_th=[]):
        return input

class LIFCell(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input, input_weights, recurrent_weights, v_th=[], v_leak=[]):
        return g.op("LIFCell", input, input_weights, recurrent_weights, v_th_f=v_th, v_leak_f=v_leak)

    @staticmethod
    def forward(ctx, input, input_weights, recurrent_weights, v_th=[], v_leak=[]):
        return input


class LIFCellModule(torch.nn.Module):
    def __init__(self):
        super(LIFCellModule, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=1, padding=0, bias=False)
        self.input_weights = torch.nn.Parameter(torch.randn(10,10))
        self.recurrent_weights = torch.nn.Parameter(torch.randn(10,10))

    def forward(self, a):
        return LIFCell.apply(self.conv(a), self.input_weights, self.recurrent_weights)

model = LIFCellModule()
model.train(True)

input_shape = (3, 100, 100)
a = torch.randn(1, *input_shape)
output_names = [ "c" ]

torch.onnx.export(model, a, "torch.onnx", input_names=["input_spikes", "input_weights", "recurrent_weights"], verbose=True)