from copy import deepcopy as dc
import dgl
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
import new_knn_graph


@dataclass
class LightGraph:

    in_cuda: bool                   # if cuda graph
    n: int                          # number of nodes
    batch_size: int                 # batch graph size
    ndata: dict                     # node attributes
    edata: dict                     # edge attributes
    tsp_value: torch.tensor        # current K-cut value

    def number_of_nodes(self):
        return self.n

    def number_of_edges(self):
        return self.n * (self.n - 1)


def to_cuda(G_, copy=True):
    if copy:
        G = dc(G_)
    else:
        G = G_
    for node_attr in G.ndata.keys():
        if node_attr in ('x', 'adj', 'label'):
            G.ndata[node_attr] = G.ndata[node_attr].cuda()
    return G


class GraphGenerator:

    def __init__(self, n, ajr):
        self.n = n
        self.ajr = ajr
        self.nonzero_idx = [i for i in range(n ** 2) if i % (n + 1) != 0]
        self.adj_mask = torch.tensor(range(0, n ** 2, n)).unsqueeze(1).expand(n, ajr + 1)

    def generate_graph(self, x=None, batch_size=1, seed=None, cuda_flag=True):

        n = self.n
        ajr = self.ajr

        # init batch graphs
        bg = LightGraph(cuda_flag, self.n, batch_size, {}, {}, torch.zeros(batch_size))

        # assign 2-d coordinates 'x'
        if x is None:
            if seed is not None:
                np.random.seed(seed)
                bg.ndata['x'] = torch.tensor(np.random.rand(batch_size * n, 2)).float()
            else:
                bg.ndata['x'] = torch.rand((batch_size * n, 2))

        else:
            bg.ndata['x'] = x.cpu()

        # label
        label = torch.tensor(range(n)).unsqueeze(1).repeat(batch_size, 1).view(-1)

        batch_mask = torch.tensor(range(0, n * batch_size, n)).unsqueeze(1).expand(batch_size, n).flatten()
        if seed is not None:
            perm_idx = torch.cat([torch.tensor(np.random.permutation(n)) for _ in range(batch_size)]) + batch_mask
        else:
            perm_idx = torch.cat([torch.randperm(n) for _ in range(batch_size)]) + batch_mask
        bg.ndata['label'] = label[perm_idx].view(batch_size, n)

        # d/adj
        _, neighbor_idx, square_dist_matrix = new_knn_graph.knn_graph(bg.ndata['x'].view(batch_size, n, -1), ajr + 1, extend_info = True)
        square_dist_matrix = F.relu(square_dist_matrix, inplace=True)  # numerical error could result in NaN in sqrt. value
        bg.ndata['adj'] = torch.sqrt(square_dist_matrix).view(bg.n * bg.batch_size, -1)

        bg.edata['d'] = bg.ndata['adj'].view(batch_size, -1, 1)[:, self.nonzero_idx, :].view(-1, 1)

        # e_type
        path_matrix = pathMatrix(bg.ndata['label'], in_cuda=False)
        path_matrix = path_matrix.view(batch_size, -1)[:, self.nonzero_idx].view(-1, 1)

        neighbor_idx -= torch.tensor(range(0, batch_size * n, n)).view(batch_size, 1, 1).repeat(1, n, ajr + 1) \
                        - torch.tensor(range(0, batch_size * n * n, n * n)).view(batch_size, 1, 1).repeat(1, n, ajr + 1)
        adjacent_matrix = torch.zeros((batch_size * n * n, 1))
        adjacent_matrix[neighbor_idx + self.adj_mask.repeat(batch_size, 1, 1)] = 1
        adjacent_matrix = adjacent_matrix.view(batch_size, n * n, 1)[:, self.nonzero_idx, :].view(-1, 1)
        bg.edata['e_type'] = torch.cat([adjacent_matrix, path_matrix], dim=1)

        if cuda_flag:
            to_cuda(bg, copy=False)
        # kcut value
        bg.tsp_value = calc_S(bg)
        return bg


def calc_S(states):
    return (states.edata['e_type'][:, 1] * states.edata['d'][:, 0]).view(states.batch_size, -1).sum(dim=1) / 2


def reset_label(state, label):
    if isinstance(label, torch.Tensor):
        state.ndata['label'] = label
    else:
        state.ndata['label'] = torch.tensor(label)

    state.ndata['label'] = state.ndata['label'].unsqueeze(0)
    path_matrix = pathMatrix(state.ndata['label'], in_cuda=False)
    path_matrix = path_matrix.view(1, -1)[:, [i for i in range(state.n ** 2) if i % (state.n + 1) != 0]].view(-1, 1)
    state.edata['e_type'][:, 1:2] = path_matrix
    if state.in_cuda:
        to_cuda(state, copy=False)
    state.tsp_value = calc_S(state)



def make_batch(graphs):

    bg = LightGraph(graphs[0].in_cuda
                    , graphs[0].number_of_nodes()
                    , len(graphs) * graphs[0].batch_size
                    , {}, {}, torch.zeros(len(graphs)))

    for node_attr in graphs[0].ndata.keys():
        bg.ndata[node_attr] = torch.cat([g.ndata[node_attr] for g in graphs])
    for edge_attr in graphs[0].edata.keys():
        bg.edata[edge_attr] = torch.cat([g.edata[edge_attr] for g in graphs])
    bg.tsp_value = torch.cat([g.tsp_value for g in graphs])

    return bg


def un_batch(graphs, copy=True):

    n = graphs.number_of_nodes()
    e = graphs.number_of_edges()
    batch_size = graphs.batch_size

    ndata = {}.fromkeys(graphs.ndata.keys())
    edata = {}.fromkeys(graphs.edata.keys())
    if copy:
        tsp_value = graphs.tsp_value.clone()
    else:
        tsp_value = graphs.tsp_value

    for node_attr in graphs.ndata.keys():
        if node_attr == 'label':
            nn = 1
        else:
            nn = n
        if copy:
            ndata[node_attr] = [graphs.ndata[node_attr][i*nn:(i+1)*nn, :].clone() for i in range(batch_size)]
        else:
            ndata[node_attr] = [graphs.ndata[node_attr][i * nn:(i + 1) * nn, :] for i in range(batch_size)]
    for edge_attr in graphs.edata.keys():
        if copy:
            edata[edge_attr] = [graphs.edata[edge_attr][i*e:(i+1)*e, :].clone() for i in range(batch_size)]
        else:
            edata[edge_attr] = [graphs.edata[edge_attr][i * e:(i + 1) * e, :] for i in range(batch_size)]
    graph_list = [LightGraph(graphs.in_cuda
                             , n
                             , 1
                             , dict([(n_attr, ndata[n_attr][i]) for n_attr in graphs.ndata.keys()])
                             , dict([(e_attr, edata[e_attr][i]) for e_attr in graphs.edata.keys()])
                             , tsp_value[i:i+1]) for i in range(batch_size)]

    return graph_list


def perm_weight(graphs, eps=0.1):
    graphs.edata['d'] *= F.relu(torch.ones(graphs.edata['d'].shape).cuda() + eps * torch.randn(graphs.edata['d'].shape).cuda())


def pathMatrix(permutation_labels, diag=0, in_cuda=True):
    b = permutation_labels.shape[0]
    n = permutation_labels.shape[1]
    label_roll1 = torch.cat([permutation_labels[:, -1].unsqueeze(1), permutation_labels[:, 0:-1]], dim=1)  # (b, n)
    label_roll2 = torch.cat([permutation_labels[:, 1:], permutation_labels[:, 0].unsqueeze(1)], dim=1)  # (b, n)
    if diag:
        path_matrix = torch.eye(n).repeat(b, 1)
    else:
        path_matrix = torch.zeros(b * n, n)
    if in_cuda:
        path_matrix = path_matrix.cuda()
        label_roll1 += torch.tensor(range(0, n * b, n)).unsqueeze(1).cuda()
        label_roll2 += torch.tensor(range(0, n * b, n)).unsqueeze(1).cuda()
    else:
        label_roll1 += torch.tensor(range(0, n * b, n)).unsqueeze(1)
        label_roll2 += torch.tensor(range(0, n * b, n)).unsqueeze(1)
    path_matrix[label_roll1.flatten(), permutation_labels.flatten()] = 1
    path_matrix[label_roll2.flatten(), permutation_labels.flatten()] = 1

    return path_matrix
