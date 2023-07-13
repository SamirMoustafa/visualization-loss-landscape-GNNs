import torch
from torch.nn import Linear, Module, ReLU, Sequential
from torch.nn.functional import dropout, log_softmax
from torch_geometric.nn import GINConv

from a2q.MessagePassingV1 import GINConvMultiQuantV1
from a2q.modules import LinearQuantizationV1


class GIN(Module):
    def __init__(self, dataset, num_layers, hidden_units, bit, is_q=False, drop_out=0):
        super(GIN, self).__init__()
        gin_layer = GINConvMultiQuantV1
        self.bit = bit
        self.drop_out = drop_out
        num_nodes = dataset.data.num_nodes
        para_list = [
            [
                {"alpha_init": 0.01, "gama_init": 0.01, "alpha_std": 0.1, "gama_std": 0.1},
                {"alpha_init": 0.01, "gama_init": 0.01, "alpha_std": 0.1, "gama_std": 0.1},
                {"gama_init": 0.01, "gama_std": 0.1},
            ]
        ]
        if is_q:
            self.conv1 = gin_layer(
                Sequential(
                    LinearQuantizationV1(
                        dataset.num_node_features,
                        hidden_units,
                        num_nodes,
                        bit,
                        all_positive=True,
                        para_dict=para_list[0][0],
                        quant_fea=True,
                    ),
                    ReLU(),
                ),
                train_eps=True,
                in_features=num_nodes,
                bit=bit,
                para_dict=para_list[0][0],
                quant_fea=False,
            )
        else:
            self.conv1 = GINConv(
                Sequential(
                    Linear(dataset.num_node_features, hidden_units),
                    ReLU(),
                ),
                train_eps=True,
            )
        self.convs = torch.nn.ModuleList()
        if is_q:
            for i in range(num_layers - 1):
                self.convs.append(
                    gin_layer(
                        Sequential(
                            LinearQuantizationV1(
                                hidden_units,
                                dataset.num_classes,
                                num_nodes,
                                bit,
                                para_dict=para_list[0][0],
                                all_positive=True,
                                quant_fea=True,
                            ),
                            ReLU(),
                        ),
                        train_eps=True,
                        in_features=num_nodes,
                        bit=bit,
                        para_dict=para_list[0][0],
                        quant_fea=True,
                        out_features=hidden_units,
                    )
                )
        else:
            for i in range(num_layers - 1):
                self.convs.append(
                    GINConv(
                        Sequential(
                            Linear(hidden_units, hidden_units),
                            ReLU(),
                        ),
                        train_eps=True,
                    )
                )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = dropout(x, p=self.drop_out, training=self.training)
        for conv in self.convs:
            x = conv(x, edge_index)
        return log_softmax(x, dim=1)
