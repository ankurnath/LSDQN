from TSP.ddqn_tsp import DQN
from TSP.envs_tsp import *
import argparse
import numpy as np
import time
import os
import pickle
from tqdm import tqdm
import json
from TSP.validation_tsp import test_summary
from TSP.canonical_solution import read_TSPLIB_data

cuda_flag = True

parser = argparse.ArgumentParser(description="GNN with RL")
parser.add_argument('--save_folder', default='/test')
parser.add_argument('--target_mode', default=False)
parser.add_argument('--n', default=50, help="size of TSP")
parser.add_argument('--ajr', default=49, help="")
parser.add_argument('--h', default=32, help="hidden dimension")
parser.add_argument('--rollout_step', default=1)
parser.add_argument('--q_step', default=2)
parser.add_argument('--batch_size', default=500, help='')
parser.add_argument('--n_episode', default=1, help='')
parser.add_argument('--episode_len', default=100, help='')
parser.add_argument('--grad_accum', default=1, help='')
parser.add_argument('--action_type', default='swap', help="")
parser.add_argument('--gnn_step', default=3, help='')
parser.add_argument('--gpu', default='0', help="")
parser.add_argument('--resume', default=False)
parser.add_argument('--problem_mode', default='complete', help="")
parser.add_argument('--readout', default='mlp', help="")
parser.add_argument('--edge_info', default='adj_dist')
parser.add_argument('--clip_target', default=0)
parser.add_argument('--explore_method', default='epsilon_greedy')
parser.add_argument('--priority_sampling', default=0)
parser.add_argument('--gamma', type=float, default=0.9, help="")
parser.add_argument('--eps0', type=float, default=0.5, help="")
parser.add_argument('--eps', type=float, default=0.1, help="")
parser.add_argument('--explore_end_at', type=float, default=0.1, help="")
parser.add_argument('--anneal_frac', type=float, default=0.9, help="")
parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
parser.add_argument('--action_dropout', type=float, default=1.0)
parser.add_argument('--n_epoch', default=50000)
parser.add_argument('--save_ckpt_step', default=50000)
parser.add_argument('--target_update_step', default=5)
parser.add_argument('--replay_buffer_size', default=5000, help="")
parser.add_argument('--test_batch_size', default=1, help='')
parser.add_argument('--validation_step', default=1000, help='')
parser.add_argument('--sample_batch_episode', type=int, default=0, help='')
parser.add_argument('--ddqn', default=False)

args = vars(parser.parse_args())
gpu = args['gpu']
os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
save_folder = args['save_folder']
resume = args['resume']
target_mode = args['target_mode']
problem_mode = args['problem_mode']
readout = args['readout']
action_type = args['action_type']
n = int(args['n'])
ajr = int(args['ajr'])
h = int(args['h'])
edge_info = args['edge_info']
clip_target = bool(int(args['clip_target']))
explore_method = args['explore_method']
priority_sampling = bool(int(args['priority_sampling']))
gamma = float(args['gamma'])
lr = args['lr']  # learning rate
action_dropout = args['action_dropout']  # learning rate
replay_buffer_size = int(args['replay_buffer_size'])
target_update_step = int(args['target_update_step'])
batch_size = int(args['batch_size'])
grad_accum = int(args['grad_accum'])
sample_batch_episode = bool(args['sample_batch_episode'])
n_episode = int(args['n_episode'])
test_episode = int(args['test_batch_size'])
validation_step = int(args['validation_step'])
episode_len = int(args['episode_len'])
gnn_step = int(args['gnn_step'])
rollout_step = int(args['rollout_step'])
q_step = int(args['q_step'])
n_epoch = int(args['n_epoch'])
explore_end_at = float(args['explore_end_at'])
anneal_frac = float(args['anneal_frac'])
eps = list(np.linspace(float(args['eps0']), float(args['eps']), int(n_epoch * explore_end_at)))
eps.extend(list(np.linspace(float(args['eps']), 0.0, int(n_epoch * anneal_frac))))
eps.extend([0] * int(n_epoch))
save_ckpt_step = int(args['save_ckpt_step'])
ddqn = bool(args['ddqn'])

path = save_folder + '/'
if not os.path.exists(path):
    os.makedirs(path)

G = GraphGenerator(n=n, ajr=ajr)
Gt = GraphGenerator(n=n, ajr=ajr)

# model to be trained
alg = DQN(graph_generator=G, hidden_dim=h, action_type=action_type
          , gamma=gamma, eps=.1, lr=lr, action_dropout=action_dropout
          , sample_batch_episode=sample_batch_episode
          , replay_buffer_max_size=replay_buffer_size
          , epi_len=episode_len
          , new_epi_batch_size=n_episode
          , cuda_flag=cuda_flag
          , explore_method=explore_method
          , priority_sampling=priority_sampling)

model_version = 0
with open(path + 'dqn_0', 'wb') as model_file:
    pickle.dump(alg, model_file)

with open(path + 'params', 'w') as params_file:
    params_file.write(time.strftime("%Y--%m--%d %H:%M:%S", time.localtime()))
    params_file.write('\n------------------------------------\n')
    params_file.write(json.dumps(args))
    params_file.write('\n------------------------------------\n')

bg_test = G.generate_graph(batch_size=test_episode, cuda_flag=cuda_flag)

test_g, problem_list, opt_value, size_list, scales = read_TSPLIB_data(ajr=ajr, cuda_flag=cuda_flag, scale=True)


def run_dqn(alg):
    r = np.ones((100, len(test_g))) * 100
    t = 0

    for n_iter in tqdm(range(n_epoch)):

        alg.eps = eps[n_iter]

        # TODO memory usage :: episode_len * num_episodes * hidden_dim
        log = alg.train_dqn(target_bg=None
                            , epoch=n_iter
                            , batch_size=batch_size
                            , num_episodes=n_episode
                            , episode_len=episode_len
                            , gnn_step=gnn_step
                            , q_step=q_step
                            , grad_accum=grad_accum
                            , rollout_step=rollout_step
                            , ddqn=ddqn)
        if n_iter % target_update_step == target_update_step - 1:
            alg.update_target_net()
        print('Epoch: {}. R: {}. Q error: {}. H: {}'
              .format(n_iter
                      , np.round(log.get_current('tot_return'), 2)
                      , np.round(log.get_current('Q_error'), 3)
                      , np.round(log.get_current('entropy'), 3)))

        if n_iter % save_ckpt_step == save_ckpt_step - 1:
            with open(path + 'dqn_' + str(model_version + n_iter + 1), 'wb') as model_file:
                pickle.dump(alg.model, model_file)
            t += 1

        if validation_step and n_iter % validation_step == 0:
            test_round = n_iter // 1000
            output_value = []

            for i in range(len(test_g)):
                Gt.__init__(n=test_g[i].n, ajr=ajr)
                test = test_summary(alg=alg, graph_generator=Gt, action_type=action_type, q_net=readout,
                                    forbid_revisit=0)
                test.run_test(problem=test_g[i], trial_num=1, batch_size=test_episode, gnn_step=gnn_step,
                              episode_len=size_list[i] * 2, explore_prob=0.0, Temperature=1e-8, cuda_flag=cuda_flag)
                epi_r0 = test.show_result()
                output_value.append(epi_r0 * scales[i])
                r[test_round, i] = output_value[i] / opt_value[i]
                epi_r0 = r.min(axis=0).mean()

if __name__ == '__main__':
    run_dqn(alg)
