import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_size, hidden_layers[0]), nn.ReLU()]
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            layers.append(nn.ReLU())

        # 定义最后一个隐藏层到输出层
        layers.append(nn.Linear(hidden_layers[-1], output_size))

        # 使用 nn.Sequential 组合层
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
