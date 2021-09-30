from ddqn import DQN
from envs import *
import argparse
import numpy as np
import time
import os
import pickle
from tqdm import tqdm
import json
from validation import test_summary
from read_data import read_sensor_data, gen_graphs, write_result, save_fig, open_data, save_fig_with_result, ri_cal_room_acc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import scipy.io as scio

from GeneticAlgorithm.colocation import run

cuda_flag = True

parser = argparse.ArgumentParser(description="GNN with RL")
parser.add_argument('--save_folder', default='/test')
parser.add_argument('--train_distr', default='cluster', help="")
parser.add_argument('--test_distr', default='cluster', help="")
parser.add_argument('--target_mode', default=False)
parser.add_argument('--k', default=10, help="size of K-cut")
parser.add_argument('--m', default='4', help="cluster size")
parser.add_argument('--ajr', default=39, help="")
parser.add_argument('--h', default=64, help="hidden dimension")
parser.add_argument('--rollout_step', default=1)
parser.add_argument('--q_step', default=2)
parser.add_argument('--batch_size', default=1000, help='')
parser.add_argument('--n_episode', default=10, help='')
parser.add_argument('--episode_len', default=100, help='')
parser.add_argument('--grad_accum', default=1, help='')
parser.add_argument('--action_type', default='swap', help="")
parser.add_argument('--gnn_step', default=3, help='')
parser.add_argument('--test_batch_size', default=10, help='')
parser.add_argument('--validation_step', default=1000, help='')
parser.add_argument('--gpu', default='2', help="")
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
parser.add_argument('--n_epoch', default=20001)
parser.add_argument('--save_ckpt_step', default=20001)
parser.add_argument('--target_update_step', default=5)
parser.add_argument('--replay_buffer_size', default=5000, help="")
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
k = int(args['k'])
m = [int(i) for i in args['m'].split(',')]
if len(m) == 1:
    m = m[0]
    N = k * m
else:
    N = sum(m)
if k == 3 and m == 4:
    run_validation_33 = True
else:
    run_validation_33 = False
ajr = int(args['ajr'])
train_graph_style = args['train_distr']
test_graph_style = args['test_distr']
h = int(args['h'])
edge_info = args['edge_info']
clip_target = bool(int(args['clip_target']))
explore_method = args['explore_method']
priority_sampling = bool(int(args['priority_sampling']))
gamma = float(args['gamma'])
lr = args['lr']    # learning rate
action_dropout = args['action_dropout']    # learning rate
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
eps.extend([0]*int(n_epoch))
save_ckpt_step = int(args['save_ckpt_step'])
ddqn = bool(args['ddqn'])

path = save_folder + '/'
if not os.path.exists(path):
    os.makedirs(path)

G = GraphGenerator(k=k, m=m, ajr=ajr, style=train_graph_style)

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

synthetic_test_g = G.generate_graph(batch_size=test_episode, style=test_graph_style, cuda_flag=cuda_flag)
x = synthetic_test_g.ndata['x'].tolist()
pca = PCA(n_components=2)
x = pca.fit_transform(x)

train_x, test_x = read_sensor_data(problem_list=['0', '1', '2', '3', '4'])
# print(len(test_x))
real_test_g = gen_graphs(x=test_x, k=k, m=m, ajr=ajr, graph_generator=G, batch_size=1, cuda_flag=cuda_flag)

# pca = PCA(n_components=2)
# X = pca.fit_transform(synthetic_test_g.ndata['x'])
# X_embedded = MDS().fit_transform(X)
# X_embedded = TSNE(n_components=2).fit_transform(X)
# save_fig(synthetic_test_g.ndata['x'])
# save_fig(X)

# size, X = open_data("./result/RelationalInferenceOutput/corr_2.mat")
# save_fig(X)

def run_dqn(alg):
    t = 0
    accuracy_file = './result/accuracy/SyntheticTrainingSyntheticTesting/accuracy_test_5.txt'
    real_data = False

    with open(accuracy_file, 'w+') as the_file:
        the_file.write("Learning Rate: " + str(lr) + "\n")

        the_file.write("Action Dropout: " + str(action_dropout) + "\n")

        the_file.write("Replay Buffer Size: " + str(replay_buffer_size) + "\n")

        the_file.write("Number of Episodes: " + str(n_episode) + "\n")
    

    for n_iter in tqdm(range(n_epoch)):
        alg.eps = eps[n_iter]
        
        log = alg.train_dqn(epoch=n_iter
                        , batch_size=batch_size
                        , num_episodes=n_episode
                        , episode_len=episode_len
                        , gnn_step=gnn_step
                        , q_step=q_step
                        , grad_accum=grad_accum
                        , rollout_step=rollout_step
                        , ddqn=ddqn)

        #
        # if n_iter%5 == 0:
        #     train_g = gen_graphs(x=train_x, k=k, m=m, ajr=ajr, batch_size=n_episode, cuda_flag=cuda_flag, train=True, scale=False)
        # log = alg.train_dqn(target_bg=train_g[n_iter%5]
        #                 , epoch=n_iter
        #                 , batch_size=batch_size
        #                 , num_episodes=n_episode
        #                 , episode_len=episode_len
        #                 , gnn_step=gnn_step
        #                 , q_step=q_step
        #                 , grad_accum=grad_accum
        #                 , rollout_step=rollout_step
        #                 , ddqn=ddqn)
        # 

        if n_iter % target_update_step == target_update_step - 1:
            alg.update_target_net()
        print('Epoch: {}. R: {}. Q error: {}. H: {}'
            .format(n_iter
            , np.round(log.get_current('tot_return'), 2)
            , np.round(log.get_current('Q_error'), 3)
            , np.round(log.get_current('entropy'), 3)
            ))
        
        if n_iter % save_ckpt_step == save_ckpt_step - 1:
            with open(path + 'dqn_'+str(model_version + n_iter + 1), 'wb') as model_file:
                pickle.dump(alg.model, model_file)
            t += 1

        if validation_step and n_iter % validation_step == 0:
            # test = test_summary(alg=alg, graph_generator=G, action_type=action_type, q_net=readout, forbid_revisit=0)
            # test.run_test(problem=real_test_g[0], trial_num=1, batch_size=test_episode, action_dropout=action_dropout, gnn_step=gnn_step,
            #               episode_len=episode_len, explore_prob=0.0, Temperature=1e-8, cuda_flag=cuda_flag)
            # test.show_result()

            # avr_accuracy, avr_initial_value, avr_S = write_result(accuracy_file=accuracy_file, batch_size=test_episode, k=k, m=m, test_summary=test)
        
            # with open(accuracy_file, 'a') as the_file:
            #     the_file.write('Epoch: {}. Accuracy: {} %. Q error: {}. Initial Value: {}. Final Value: {}.'
            # .format(n_iter
            # , np.round(avr_accuracy, 2)
            # , np.round(log.get_current('Q_error'), 3)
            # , np.round(avr_initial_value, 2)
            # , np.round(avr_S, 2)
            # ) + "\n")
        
            test = test_summary(alg=alg, graph_generator=G, action_type=action_type, q_net=readout, forbid_revisit=0)
            test.run_test(problem=synthetic_test_g, trial_num=1, batch_size=test_episode, action_dropout=action_dropout, gnn_step=gnn_step,
                          episode_len=episode_len, explore_prob=0.0, Temperature=1e-8, cuda_flag=cuda_flag)
            test.show_result()
            # print(test.bg)

            # save_fig_with_result(k, m, test_episode, x, test, n_iter)

            avr_accuracy, avr_initial_value = write_result(batch_size=test_episode, k=k, m=m, test=test)
        
            with open(accuracy_file, 'a') as the_file:
                the_file.write('Epoch: {}. Accuracy: {} %. Q error: {}. Initial Value: {}. Final Value: {}.'
            .format(n_iter
            , np.round(avr_accuracy, 2)
            , np.round(log.get_current('Q_error'), 3)
            , np.round(avr_initial_value, 2)
            , np.round(test.final_value, 2)
            ) + "\n")

            # test_corr = np.corrcoef(np.array(synthetic_test_g.ndata['x']))
            # scio.savemat('./result/RelationalInferenceOutput/corr_' + str(n_iter) + '.mat', {'corr':test_corr})
            # best_solution, acc, ground_truth_fitness, best_fitness = run.ga(path_m = './result/RelationalInferenceOutput/corr_' + str(n_iter) + '.mat', path_c = '../RI-Coequipment/colocation/10_rooms.json')
            # recall, room_wise_acc = ri_cal_room_acc(best_solution)
            # print("recall = %f, room_wise_acc = %f:\n" %(recall, room_wise_acc))

            # print("Ground Truth Fitness %f Best Fitness: %f \n" % (ground_truth_fitness, best_fitness))
            # print("Edge-wise accuracy: %f \n" % (acc))

if __name__ == '__main__':

    run_dqn(alg)
