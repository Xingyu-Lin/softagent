
import torch
import torch.nn as nn

from rlpyt.models.mlp import MlpModel
from rlpyt.models.utils import conv2d_output_shape


class Conv2dModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            channels,
            kernel_sizes,
            strides,
            paddings=None,
            nonlinearity=torch.nn.ReLU,  # Module, not Functional.
            use_maxpool=False,  # if True: convs use stride 1, maxpool downsample.
            use_layernorm=True,
            ):
        super().__init__()
        h, w, in_channels = image_shape
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = (in_channels,) + tuple(channels[:-1])
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            maxp_strides = strides
            strides = ones
        else:
            maxp_strides = ones
        conv_layers = [torch.nn.Conv2d(in_channels=ic, out_channels=oc,
            kernel_size=k, stride=s, padding=p) for (ic, oc, k, s, p) in
            zip(in_channels, channels, kernel_sizes, strides, paddings)]
        sequence = list()
        for conv_layer, maxp_stride, c, s in zip(conv_layers, maxp_strides, channels, strides):
            h, w = h // s, w // s
            sequence.append(conv_layer)
            if use_layernorm:
                sequence.append(nn.LayerNorm((c, h, w)))
            sequence.append(nonlinearity())
            if maxp_stride > 1:
                sequence.append(torch.nn.MaxPool2d(maxp_stride))  # No padding.
        self.conv = torch.nn.Sequential(*sequence)

    def forward(self, input):
        return self.conv(input)

    def conv_out_size(self, h, w, c=None):
        for child in self.conv.children():
            try:
                h, w = conv2d_output_shape(h, w, child.kernel_size,
                    child.stride, child.padding)
            except AttributeError:
                pass  # Not a conv or maxpool layer.
            try:
                c = child.out_channels
            except AttributeError:
                pass  # Not a conv layer.
        return h * w * c


class Conv2dHeadModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            channels,
            kernel_sizes,
            strides,
            hidden_sizes,
            output_size=None,  # if None: nonlinearity applied to output.
            paddings=None,
            nonlinearity=torch.nn.ReLU,
            use_maxpool=False,
            extra_input_size=0,
            ):
        super().__init__()
        self._extra_input_size = extra_input_size
        h, w, c = image_shape
        self.conv = Conv2dModel(
            image_shape=image_shape,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            nonlinearity=nonlinearity,
            use_maxpool=use_maxpool,
        )
        conv_out_size = self.conv.conv_out_size(h, w)
        if hidden_sizes or output_size:
            self.head = MlpModel(conv_out_size + extra_input_size, hidden_sizes,
                output_size=output_size, nonlinearity=nonlinearity)
            if output_size is not None:
                self._output_size = output_size
            else:
                self._output_size = (hidden_sizes if
                    isinstance(hidden_sizes, int) else hidden_sizes[-1])
        else:
            self.head = lambda x: x
            self._output_size = conv_out_size

    def forward_embedding(self, input):
        return self.conv(input).view(input.shape[0], -1)

    def forward_output(self, input, extra_input=None):
        if self._extra_input_size > 0:
            assert extra_input.shape[1] == self._extra_input_size, (extra_input.shape, self._extra_input_size, input.shape)
            extra_input = extra_input.view(extra_input.shape[0], -1)
            mlp_input = torch.cat((input, extra_input), dim=-1)
        else:
            mlp_input = input
        return self.head(mlp_input)

    def forward(self, input, extra_input=None):
        embedding = self.forward_embedding(input)
        return self.forward_output(embedding, extra_input=extra_input)

    @property
    def output_size(self):
        return self._output_size
