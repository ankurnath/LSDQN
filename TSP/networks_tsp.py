import torch.nn as nn
from TSP.envs_tsp import *


class GCN(nn.Module):
    def __init__(self, n, hidden_dim, activation=F.relu):
        super(GCN, self).__init__()
        self.n = n
        self.l1 = nn.Linear(2, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(1, hidden_dim)
        self.activ = activation

    def forward(self, graphs, feature, use_label=True, use_edge=False):

        b = graphs.batch_size
        n = graphs.n
        adjM = graphs.ndata['adj']
        n1_h = torch.bmm(adjM.view(b, n, n), feature.view(b, n, -1)).view(b * n, -1)  # (bn, h)
        x = graphs.ndata['x']

        if use_edge:
            n2_h = self.activ(self.l4(adjM.view(b, n, n, 1))).sum(dim=-2).view(b * n, -1) / n  # (bn, h)
            h = self.activ(self.l1(x) + self.l2(n1_h) + self.l3(n2_h), inplace=True)
        else:
            h = self.activ(self.l1(x) + self.l2(n1_h), inplace=True)

        return h


def bPtAP(P, A, b, n):
    return torch.bmm(torch.bmm(P.transpose(1, 2), A.view(b, n, n)), P)


def batch_trace(A, b, k, cuda_flag):
    if cuda_flag:
        return (A * torch.eye(k).repeat(b, 1, 1).cuda()).sum(dim=2).sum(dim=1) / 2
    else:
        return (A * torch.eye(k).repeat(b, 1, 1)).sum(dim=2).sum(dim=1) / 2


class DQNet(nn.Module):
    def __init__(self, n, hidden_dim):
        super(DQNet, self).__init__()
        self.n = n
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([GCN(n, hidden_dim)])
        # baseline
        self.t5 = nn.Linear(self.hidden_dim + 8, 1)
        self.t5_ = nn.Linear(self.hidden_dim + 1, 1)
        self.t6 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.t7_0 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.t7_1 = nn.Linear(self.hidden_dim, self.hidden_dim)

        # RNN encoder
        self.rnn = nn.RNN(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=2, batch_first=True)

        self.L12_h = nn.Linear(8, self.hidden_dim)
        self.L12_12 = nn.Linear(8, 8)
        self.L12_1 = nn.Linear(8, 1)

        self.L1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.L2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.L3 = nn.Linear(2, 1)
        self.L4 = nn.Linear(self.hidden_dim, 2)
        self.L5 = nn.Linear(2, 1)

    def forward_prop(self, graphs, actions=None, gnn_step=3, top_ratio=0.1):

        n = graphs.n
        b = graphs.batch_size
        bn = b * n
        num_action = actions.shape[0] // b
        rangeb = torch.tensor(range(b)).unsqueeze(1).repeat(1, num_action).flatten()
        if graphs.in_cuda:
            rangeb = rangeb.cuda()
        actions01 = torch.cat([graphs.ndata['label'][rangeb, actions[:, 0]].unsqueeze(1)
                                , graphs.ndata['label'][rangeb, actions[:, 1]].unsqueeze(1)], dim=1)


        A = graphs.ndata['adj']

        h = torch.zeros((bn, self.hidden_dim))

        if graphs.in_cuda:
            h = h.cuda(device=A.device)

        for _ in range(gnn_step):
            h = self.layers[0].forward(graphs, h)  # (bn, h)


        rnn_output = self.rnn(h.view(b, n, -1).cuda())
        state_embedding = rnn_output[1][-1, :, :]

        # Action proposal network: 2-layer MLP
        proto_a = F.relu(self.L2(F.relu(self.L1(state_embedding.detach())))).unsqueeze(1)  # (b, 1, h)

        h_a = self.L4(h.view(b, n, -1) * proto_a).view(b*n, 2)  # (b, n, h) -> (b, n, 2) -> (bn, 2)
        h_a_0 = h_a[actions01[:, 0], :]
        h_a_1 = h_a[actions01[:, 1], :]
        prop_a_score = self.L5(F.relu((h_a_0 + h_a_1 + self.L3(h_a_0 * h_a_1)))).view(b, -1).softmax(dim=1)  # (b, num_action)

        topk_action_num = int(top_ratio * num_action)

        prop_action_indices = torch.multinomial(prop_a_score, topk_action_num).view(-1)  # (b, topk) -> (b * topk)

        topk_mask = torch.tensor(range(0, num_action*b,num_action)).unsqueeze(1).repeat(1,topk_action_num).view(-1)
        if graphs.in_cuda:
            topk_mask = topk_mask.cuda()
        prop_action_indices_ = prop_action_indices + topk_mask
        prop_actions = actions[prop_action_indices_, :]

        return prop_a_score.view(b * num_action, 1)[prop_action_indices_, :].view(b, -1), prop_actions


    def forward(self, graphs, actions=None, action_type='swap', gnn_step=3, leak=True):

        n = graphs.n
        b = graphs.batch_size
        bn = b * n
        num_action = actions.shape[0] // b
        rangeb = torch.tensor(range(b)).unsqueeze(1).repeat(1, num_action).flatten()
        if graphs.in_cuda:
            rangeb = rangeb.cuda()
        actions01 = torch.cat([graphs.ndata['label'][rangeb, actions[:, 0]].unsqueeze(1)
                                , graphs.ndata['label'][rangeb, actions[:, 1]].unsqueeze(1)], dim=1)

        A = graphs.ndata['adj']
        h = torch.zeros((bn, self.hidden_dim))

        if graphs.in_cuda:
            h = h.cuda(device=A.device)

        for _ in range(gnn_step):
            h = self.layers[0].forward(graphs, h)  # (bn, h)

        action_mask = torch.tensor(range(0, bn, n))\
            .unsqueeze(1).expand(b, 2)\
            .repeat(1, num_action)\
            .view(num_action * b, -1)

        if graphs.in_cuda:
            action_mask = action_mask.cuda()

        left_shift = torch.cat([graphs.ndata['label'][:, -1].unsqueeze(1), graphs.ndata['label'][:, 0:-1]], dim=1)  # (b, n)
        right_shift = torch.cat([graphs.ndata['label'][:, 1:], graphs.ndata['label'][:, 0].unsqueeze(1)], dim=1)  # (b, n)

        # left neighbor index
        li = left_shift[rangeb, actions[:, 0]].unsqueeze(1)  # (b * num_action, 1)
        # right neighbor index
        ri = right_shift[rangeb, actions[:, 1]].unsqueeze(1)  # (b * num_action, 1)
        pad4 = torch.cat([actions01, li, ri], dim=1) + action_mask.repeat(1, 2)  # (b * num_action, 4)

        X = graphs.ndata['x']  # (bn, 2)
        Y = X[pad4]  # (b * num_action, 4, 2)

        R = (Y[:, 0] - Y[:, 2]).norm(dim=1) + (Y[:, 1] - Y[:, 3]).norm(dim=1) \
            - (Y[:, 1] - Y[:, 2]).norm(dim=1) - (Y[:, 0] - Y[:, 3]).norm(dim=1)

        rnn_output = self.rnn(h.view(b, n, -1).cuda())
        graph_embedding = rnn_output[1][-1, :, :]

        if leak:
            Q_sa = self.t5_(torch.cat(
                [F.relu(
                    (
                        self.t6(graph_embedding).view(b, 1, -1) + self.L12_h(Y.view(b, num_action, 8))
                     ).view(b * num_action, -1), inplace=True), R.unsqueeze(1)], dim=1)
                ).squeeze()
        else:
            Q_sa = self.t5(F.relu(
                    (
                            torch.cat([self.t6(graph_embedding).repeat(num_action, 1), self.L12_12(Y.view(b*num_action, 8))], dim=1)
                    ).view(b * num_action, -1), inplace=True)
            ).squeeze()

        return 0, R, h, Q_sa
