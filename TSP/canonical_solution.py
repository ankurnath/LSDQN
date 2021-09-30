import re
import itertools
from tqdm import tqdm
from TSP.envs_tsp import *
import os

def read_TSPLIB_data(ajr, cuda_flag, scale=False):
    problem_list = ['eil51'
        , 'berlin52'
        , 'st70'
        , 'eil76'
        , 'pr76'
        , 'rat99'
        , 'kroA100'
        , 'kroB100'
        , 'kroC100'
        , 'kroD100'
        , 'kroE100'
        , 'rd100'
        , 'eil101'
        , 'lin105'
        , 'pr107'
        , 'pr124'
        , 'bier127'
        , 'ch130'
        , 'pr136'
        , 'pr144'
        , 'ch150'
        , 'kroA150'
        , 'kroB150'
        , 'pr152'
        , 'u159'
        , 'rat195'
        , 'd198'
        , 'kroA200'
        , 'kroB200'
        , 'ts225'
        , 'tsp225'
        , 'pr226'
        # , 'gil262'
        # , 'pr264'
        # , 'a280'
        # , 'pr299'
        # , 'lin318'
        # , 'linhp318'
        ]

    opt_value = [426, 7542, 675, 538, 108159, 1211, 21282, 22141, 20749, 21294, 22068, 7910, 629, 14379, 44303, 59030,
                 118282, 6110, 96772, 58537, 6528, 26524, 26130, 73682, 42080, 2323, 15780, 29368, 29437, 126643, 3916,
                 80369, 2378, 49135, 2579, 48191, 42029, 41345]

    size_list = [int(x) for x in itertools.chain(*[re.findall(r"\d+\.?\d*", name) for name in problem_list])]

    test_g = []
    scales = []
    print('read tsplib data...')
    for problem_num in tqdm(range(len(problem_list))):
        name = problem_list[problem_num]
        path_x = os.path.abspath('.') + '/TSP/ALL_tsp/' + name + '.tsp'
        with open(path_x, 'r') as f:
            x = torch.tensor([(float(x.split()[1]), float(x.split()[2])) for x in
                              [l.strip('\n') for l in f.readlines() if
                               len(l.strip('\n| ')) > 0 and l.strip('\n| ')[0] in ['1', '2', '3', '4', '5', '6', '7', '8',
                                                                                   '9']]])
            # print(path_x, x)
        G = GraphGenerator(n=x.shape[0], ajr=ajr)

        if scale:
            scales.append(x.max().numpy())
            test_g.append(G.generate_graph(x=x / scales[-1], cuda_flag=cuda_flag))
        else:
            test_g.append(G.generate_graph(x=x, cuda_flag=cuda_flag))

    return test_g, problem_list, opt_value, size_list, scales


def greedy_solver(graph, step=10):
    Actions = []
    Rewards = []
    graph = dc(graph)
    for j in range(step):
        actions = get_legal_actions(states=graph)
        r = peek_greedy_reward(states=graph, actions=actions)
        Rewards.append(r.max().item())
        chosen_action = actions[r.argmax()].unsqueeze(0)
        Actions.append(chosen_action)
        _, rr = step_batch(states=graph, action=chosen_action)

    return graph, Actions, Rewards

def farthest(X, D, start=0):
    D = (D+0.5).int()
    n = X.shape[0]
    tour = [start]
    tour.append(D[start, :].argmax().item())
    for i in range(n - 2):
        farthest_insertion_i = D[tour, :].min(dim=0).values.argmax().item()
        tour.append(tour[0])
        insert_change = [D[tour[j], farthest_insertion_i] + D[farthest_insertion_i, tour[j+1]] - D[tour[j], tour[j+1]] for j in range(len(tour)-1)]
        tour.pop()
        insert_position = np.argmin(insert_change)
        tour.insert(insert_position + 1, farthest_insertion_i)

    S = D[tour[0], tour[n-1]]
    for i in range(n - 1):
        S += D[tour[i], tour[i+1]]
    return tour, S


if __name__ == '__main__':

    test_g_origin, problem_list, opt_value, size_list, scales = read_TSPLIB_data(ajr=1, cuda_flag=True, scale=False)
    test_g, problem_list, opt_value, size_list, scales = read_TSPLIB_data(ajr=1, cuda_flag=True, scale=True)

    output_value_2opt = []
    output_value_farthest = []
    for problem_num in range(len(problem_list)):

        X = test_g[problem_num].ndata['x']
        D = test_g_origin[problem_num].ndata['adj']
        G = GraphGenerator(n=X.shape[0], ajr=1)


        trial = 1
        opt2_result = []
        far_result = []
        for i in tqdm(range(trial)):
            g = G.generate_graph(x=X, cuda_flag=True)
            g1, a, r = greedy_solver(graph=g, step=size_list[problem_num]*2)
            opt2_result.append(g1.tsp_value)
            t, S = farthest(X, D, start=0)
            far_result.append(S)
        print(problem_list[problem_num])
        print('2-opt:', min(opt2_result).numpy()[0] * scales[problem_num].item())
        print('farthest:', min(far_result).cpu().numpy().item())
        output_value_2opt.append(min(opt2_result).numpy()[0] * scales[problem_num].item())
        output_value_farthest.append(min(far_result).cpu().numpy().item())

    r_2opt = 0
    r_farthest = 0
    for i in range(len(output_value_2opt)):
        r_2opt += output_value_2opt[i] / opt_value[i]
        r_farthest += output_value_farthest[i] / opt_value[i]
        print('2-opt:', r_2opt, 'farthest:', r_farthest)
    print('avg approx ratio for 2-opt:', r_2opt / len(output_value_2opt))
    print('avg approx ratio for farthest:', r_farthest / len(output_value_farthest))
