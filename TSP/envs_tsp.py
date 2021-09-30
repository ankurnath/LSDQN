from TSP.graph_handler_tsp import *
from TSP.graph_handler_tsp import pathMatrix


def peek_greedy_reward(states, actions=None, action_type='swap'):
    """
    :param states: LightGraph
    :param actions:
    :param action_type:
    :return:
    """
    b = states.batch_size
    n = states.n
    bn = b * n

    if actions is None:
        actions = get_legal_actions(states=states, action_type=action_type)
    num_action = actions.shape[0] // b

    rangeb = torch.tensor(range(b)).unsqueeze(1).repeat(1, num_action).flatten()
    if states.in_cuda:
        rangeb = rangeb.cuda()
    actions01 = torch.cat([states.ndata['label'][rangeb, actions[:, 0]].unsqueeze(1)
                            , states.ndata['label'][rangeb, actions[:, 1]].unsqueeze(1)], dim=1)

    action_mask = torch.tensor(range(0, bn, n))\
        .unsqueeze(1).expand(b, 2)\
        .repeat(1, num_action)\
        .view(num_action * b, -1)

    if states.in_cuda:
        action_mask = action_mask.cuda()

    left_shift = torch.cat([states.ndata['label'][:, -1].unsqueeze(1), states.ndata['label'][:, 0:-1]], dim=1)  # (b, n)
    right_shift = torch.cat([states.ndata['label'][:, 1:], states.ndata['label'][:, 0].unsqueeze(1)], dim=1)  # (b, n)

    # left neighbor index
    li = left_shift[rangeb, actions[:, 0]].unsqueeze(1)  # (b * num_action, 1)
    # right neighbor index
    ri = right_shift[rangeb, actions[:, 1]].unsqueeze(1)  # (b * num_action, 1)
    pad4 = torch.cat([actions01, li, ri], dim=1) + action_mask.repeat(1, 2)  # (b * num_action, 4)

    # print('pad4', pad4)
    X = states.ndata['x']  # (bn, 2)
    Y = X[pad4]  # (b * num_action, 4, 2)

    R = (Y[:, 0] - Y[:, 2]).norm(dim=1) + (Y[:, 1] - Y[:, 3]).norm(dim=1) \
        - (Y[:, 1] - Y[:, 2]).norm(dim=1) - (Y[:, 0] - Y[:, 3]).norm(dim=1)

    return R


def get_legal_actions(states, action_dropout=1.0, action_type=None, pause_action=False, return_num_action=False):
    """
    :param states: LightGraph
    :param action_type:
    :param action_dropout:
    :param pause_action:
    :param return_num_action: if only returns the number of actions
    :return:
    """
    n = states.n
    a = torch.ones(n, n)
    b = torch.tril(a)
    b[0, n - 1] = 1
    legal_actions = (1 - b).nonzero()
    # only allows adjacent swap
    # legal_actions = torch.cat([torch.cat([torch.tensor(range(0, n - 1)).unsqueeze(1), torch.tensor(range(1, n)).unsqueeze(1)], dim=1), torch.tensor([[n - 1, 0]])], dim=0)
    if states.in_cuda:
        legal_actions = legal_actions.cuda()
    num_actions = legal_actions.shape[0]
    legal_actions = legal_actions.repeat(states.batch_size, 1)

    if return_num_action:
        if pause_action:
            return int(num_actions * action_dropout) + 1
        else:
            return int(num_actions * action_dropout)

    if action_dropout < 1.0:
        num_actions = legal_actions.shape[0] // states.batch_size
        maintain_actions = int(num_actions * action_dropout)
        maintain = [np.random.choice(range(_ * num_actions, (_ + 1) * num_actions), maintain_actions, replace=False) for _ in range(states.batch_size)]
        legal_actions = legal_actions[torch.tensor(maintain).flatten(), :]
    if pause_action:
        legal_actions = legal_actions.reshape(states.batch_size, -1, 2)
        legal_actions = torch.cat([legal_actions, (legal_actions[:, 0] * 0).unsqueeze(1)], dim=1).view(-1, 2)

    return legal_actions

def flip_label(a, i, range):
    a[i, range] = a[i, range].flip(0)

def step_batch(states, action, action_type=None):
    """
    :param states: LightGraph
    :param action: torch.tensor((batch_size, 2))
    :return:
    """
    assert states.batch_size == action.shape[0]

    batch_size = states.batch_size
    n = states.n

    ii, jj = action[:, 0], action[:, 1]

    old_S = states.tsp_value

    # flip label in 2-opt style
    [flip_label(states.ndata['label'], row, range(ii[row], jj[row]+1)) for row in range(batch_size)]
    # rewire edges
    nonzero_idx = [i for i in range(n ** 2) if i % (n + 1) != 0]
    path_matrix = pathMatrix(permutation_labels=states.ndata['label'], in_cuda=states.in_cuda)
    states.edata['e_type'][:, 1:2] = path_matrix.view(batch_size, -1)[:, nonzero_idx].view(-1, 1)
    # compute new S
    new_S = calc_S(states)
    states.tsp_value = new_S

    rewards = old_S - new_S

    return states, rewards
