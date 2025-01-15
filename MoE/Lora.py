import torch.nn as nn
from collections import OrderedDict
import torch.nn as nn
import torch
import math
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """
    Base lora class
    """

    def __init__(
            self,
            r,
            lora_alpha,
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        # Mark the weight as unmerged
        self.merged = False

    def reset_parameters(self):
        raise NotImplementedError

    def train(self, mode: bool = True):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError


class LoRALinear(LoRALayer):
    def __init__(self, r, lora_alpha, linear_layer):
        """
        LoRA class for nn.Linear class
        :param r: low rank dimension
        :param lora_alpha: scaling factor
        :param linear_layer: target nn.Linear layer for applying Lora
        """
        super().__init__(r, lora_alpha)
        self.linear = linear_layer

        in_features = self.linear.in_features
        out_features = self.linear.out_features

        # Lora configuration
        self.lora_A = nn.Parameter(self.linear.weight.new_zeros((r, in_features)))
        self.lora_B = nn.Parameter(self.linear.weight.new_zeros((out_features, r)))

        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def get_task_weights(self):
        return (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling

    def train(self, mode: bool = True):
        self.linear.train(mode)

    def eval(self):
        self.linear.eval()

    def forward(self, x):
        result = F.linear(
            input=x,
            weight=(self.lora_B @ self.lora_A) * self.scaling, bias=self.linear.bias
        )
        return result


class LoraConv2d(LoRALayer):
    def __init__(self, r, lora_alpha, conv_layer):
        super().__init__(r, lora_alpha)

        self.conv = conv_layer

        in_channels = self.conv.in_channels
        out_channels = self.conv.out_channels
        kernel_size = self.conv.kernel_size[0]

        self.lora_A = nn.Parameter(
            self.conv.weight.new_zeros((kernel_size * r, in_channels * kernel_size))
        )
        self.lora_B = nn.Parameter(
            self.conv.weight.new_zeros((out_channels * kernel_size, kernel_size * r))
        )

        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        self.conv.train(mode)

    def eval(self):
        self.conv.eval()

    def get_task_weights(self):
        return (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling

    def forward(self, x):
        result = F.conv2d(
            x,
            (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
            self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups
        )
        return result


class MLoraConv2d(LoRALayer):
    def __init__(self, r, lora_alpha, conv_layer, num_E=3):
        """
        LoRA class for nn.Conv2d class
        """
        super().__init__(r, lora_alpha)
        self.conv = conv_layer

        self.num_E = num_E

        self.experts = []
        self.gates = []

        for _ in range(self.num_E):
            self.experts.append(LoraConv2d(r, lora_alpha, conv_layer))

            self.gates.append(nn.Sequential(
                nn.Conv2d(self.conv.out_channels, self.conv.out_channels, kernel_size=1, stride=1, padding=0,
                          bias=False),
                nn.BatchNorm2d(self.conv.out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.conv.out_channels, self.conv.out_channels, kernel_size=1, stride=1, padding=0,
                          bias=False),
                nn.Dropout2d(0.1),
                nn.BatchNorm2d(self.conv.out_channels),
                nn.Sigmoid()
            ))
            # self.merges.append(nn.Parameter(
            # self.conv.weight.new_ones(self.conv.weight.shape)
            # ))
        self.experts = nn.ModuleList(self.experts)
        self.gates = nn.ModuleList(self.gates)
        # self.merges = nn.ParameterList(self.merges)

    def get_query(self, x):
        self.query = x

    def self_gate(self, x):
        return x * torch.sigmoid(x)

    def get_task_weights(self):
        return [self.experts[i].get_task_weights() for i in range(self.num_E)]

    def train(self, mode: bool = True):
        self.conv.train(mode)

    def eval(self):
        self.conv.eval()

    def forward(self, x):

        '''results = []
        for i in range(self.num_E):
            results.append(self.experts[i](x).unsqueeze(-2))
        results = torch.cat(results, dim=-2)
        self.query = self.query.unsqueeze(-1)
        att = torch.softmax(torch.matmul(results, self.query), dim=-1).permute(0, 1, 2, 4, 3)
        results = torch.matmul(att, results)
        return results.squeeze()
        '''

        results = self.conv(x)
        results += torch.mul(2 * self.gates[0](self.query), self.experts[0](x))
        # results = self.experts[0](x)
        for i in range(1, self.num_E):
            results += torch.mul(2 * self.gates[i](self.query), self.experts[i](x))
            # results += self.experts[i](x)

        return results


class MLoraLinear(LoRALayer):
    def __init__(self, r, lora_alpha, linear_layer, num_E=3):
        """
        LoRA class for nn.Conv2d class
        """
        super().__init__(r, lora_alpha)
        self.linear = linear_layer

        self.num_E = num_E

        self.experts = []
        self.gates = []

        for _ in range(self.num_E):
            self.experts.append(LoRALinear(r, lora_alpha, linear_layer))

            self.gates.append(nn.Sequential(
                #nn.Linear(768, self.linear.out_features),
                #nn.LayerNorm(self.linear.out_features),
                #nn.ReLU(inplace=True),
                nn.Linear(768, self.linear.out_features),
                nn.LayerNorm(self.linear.out_features),
                nn.Dropout(0.1),
                nn.Sigmoid()
            ))
            # self.merges.append(nn.Parameter(
            # self.conv.weight.new_ones(self.conv.weight.shape)
            # ))
        self.experts = nn.ModuleList(self.experts)
        self.gates = nn.ModuleList(self.gates)
        # self.merges = nn.ParameterList(self.merges)

    def get_query(self, x):
        self.query = x

    def self_gate(self, x):
        return x * torch.sigmoid(x)

    def get_task_weights(self):
        return [self.experts[i].get_task_weights() for i in range(self.num_E)]

    def train(self, mode: bool = True):
        self.linear.train(mode)

    def eval(self):
        self.linear.eval()

    def forward(self, x):
        results = self.linear(x)
        results += torch.mul(2 * self.gates[0](self.query), self.experts[0](x))
        for i in range(1, self.num_E):
            results += torch.mul(2 * self.gates[i](self.query), self.experts[i](x))
        return results
