import time
from networks import *
from log_utils import logger
from dataclasses import dataclass
import itertools


class EpisodeHistory:
    def __init__(self, g, max_episode_len, action_type='swap'):
        self.action_type = action_type
        self.init_state = dc(g)

        self.n = g.number_of_nodes()
        self.max_episode_len = max_episode_len
        self.episode_len = 0
        self.action_seq = []
        self.action_indices = []
        self.reward_seq = []
        self.q_pred = []
        self.action_candidates = []
        self.enc_state_seq = []
        self.sub_reward_seq = []
        if self.action_type == 'swap':
            self.label_perm = torch.tensor(range(self.n)).unsqueeze(0)
        if self.action_type == 'flip':
            self.label_perm = self.init_state.ndata['label'].nonzero()[:, 1].unsqueeze(0)
        self.best_gain_sofar = 0
        self.current_gain = 0
        self.loop_start_position = 0

    def perm_label(self, label, action):
        label = dc(label)
        if self.action_type == 'swap':
            tmp = dc(label[action[0]])
            label[action[0]] = label[action[1]]
            label[action[1]] = tmp
        if self.action_type == 'flip':
            label[action[0]] = action[1]
        return label.unsqueeze(0)

    def write(self, action, action_idx, reward, q_val=None, actions=None, state_enc=None, sub_reward=None, loop_start_position=None):

        new_label = self.perm_label(self.label_perm[-1, :], action)

        self.action_seq.append(action)

        self.action_indices.append(action_idx)

        self.reward_seq.append(reward)

        self.q_pred.append(q_val)

        self.action_candidates.append(actions)

        self.sub_reward_seq.append(sub_reward)

        self.loop_start_position = loop_start_position

        self.label_perm = torch.cat([self.label_perm, new_label], dim=0)

        self.episode_len += 1

    def wrap(self):
        self.reward_seq = torch.tensor(self.reward_seq)
        self.empl_reward_seq = torch.tensor(self.empl_reward_seq)
        self.label_perm = self.label_perm.long()


@dataclass
class sars:
    s0: LightGraph
    a: tuple
    r: float
    s1: LightGraph
    rollout_r: torch.tensor
    rollout_a: torch.tensor


class DQN:
    def __init__(self, graph_generator
                 , hidden_dim=32
                 , action_type='swap'
                 , gamma=1.0, eps=0.1, lr=1e-4, action_dropout=1.0
                 , sample_batch_episode=False
                 , replay_buffer_max_size=5000
                 , epi_len=50, new_epi_batch_size=10
                 , cuda_flag=True
                 , explore_method='epsilon_greedy'
                 , priority_sampling='False'):

        self.cuda_flag = cuda_flag
        self.graph_generator = graph_generator
        self.gen_training_sample_first = False
        if self.gen_training_sample_first:
            self.training_instances = un_batch(self.graph_generator.generate_graph(batch_size=100, cuda_flag=self.cuda_flag), copy=False)
        self.action_type = action_type
        self.k = graph_generator.k
        self.ajr = graph_generator.ajr
        self.hidden_dim = hidden_dim  # hidden dimension for node representation
        self.n = graph_generator.n
        self.eps = eps  # constant for exploration in dqn
        self.explore_method = explore_method
        if cuda_flag:
            self.model = DQNet(k=self.k, n=self.n, hidden_dim=self.hidden_dim).cuda()
        else:
            self.model = DQNet(k=self.k, n=self.n, hidden_dim=self.hidden_dim)
        # self.model = torch.nn.DataParallel(self.model)
        self.model_target = dc(self.model)
        self.gamma = gamma  # reward decay const
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.optimizer2 = torch.optim.Adam([p[1] for p in filter(lambda p: p[0] in ['L1.weight', 'L2.weight', 'L1.bias', 'L2.bias'], self.model.named_parameters())], lr=lr)
        self.sample_batch_episode = sample_batch_episode
        self.experience_replay_buffer = []
        self.replay_buffer_max_size = replay_buffer_max_size
        self.buf_epi_len = epi_len  # 50
        self.new_epi_batch_size = new_epi_batch_size  # 10
        self.cascade_replay_buffer = [[] for _ in range(self.buf_epi_len)]
        self.cascade_replay_buffer_weight = torch.zeros((self.buf_epi_len, self.new_epi_batch_size))
        self.stage_max_sizes = [self.replay_buffer_max_size // self.buf_epi_len] * self.buf_epi_len  # [100, 100, ..., 100]
        # self.stage_max_sizes = list(range(100,100+4*50, 4))
        self.buffer_actual_size = sum(self.stage_max_sizes)
        self.priority_sampling = priority_sampling
        self.cascade_buffer_kcut_value = torch.zeros((self.buf_epi_len, self.new_epi_batch_size))
        self.action_dropout = action_dropout
        self.log = logger()
        self.Q_err = 0  # Q error
        self.log.add_log('tot_return')
        self.log.add_log('Q_error')
        self.log.add_log('Reconstruction_error')
        self.log.add_log('Act_Prop_error')
        self.log.add_log('entropy')
        self.log.add_log('R_signal_posi_len')
        self.log.add_log('R_signal_nega_len')
        self.log.add_log('R_signal_posi_mean')
        self.log.add_log('R_signal_nega_mean')
        self.log.add_log('R_signal_nonzero')
        self.log.add_log('S_new_training_sample')

    def _updata_lr(self, step, max_lr, min_lr, decay_step):
        for g in self.optimizer.param_groups:
            g['lr'] = max(max_lr / ((max_lr / min_lr) ** (step / decay_step) ), min_lr)

    def run_batch_episode(self, target_bg=None, action_type='swap', gnn_step=3, episode_len=50, batch_size=10, rollout_step=1):

        sum_r = 0

        if target_bg is None:
            if self.gen_training_sample_first:
                bg = make_batch(np.random.choice(self.training_instances, batch_size, replace=False))
            else:
                bg = self.graph_generator.generate_graph(batch_size=batch_size, cuda_flag=self.cuda_flag)

            self.log.add_item('S_new_training_sample', torch.mean(bg.kcut_value).item())
        else:
            assert target_bg.in_cuda == self.cuda_flag
            bg = dc(target_bg)
            perm_weight(bg)

        num_actions = get_legal_actions(states=bg, action_type=action_type, action_dropout=self.action_dropout, return_num_action=True)

        action_mask = torch.tensor(range(0, num_actions * batch_size, num_actions))
        if self.cuda_flag:
            action_mask = action_mask.cuda()

        explore_dice = (torch.rand(episode_len, batch_size) < self.eps)
        explore_replace_mask = explore_dice.nonzero()
        explore_step_offset = torch.cat([torch.zeros([1], dtype=torch.long), torch.cumsum(explore_dice.sum(dim=1), dim=0)], dim=0)
        explore_replace_actions = torch.randint(high=num_actions, size=(explore_replace_mask.shape[0], ))
        if self.cuda_flag:
            explore_replace_actions = explore_replace_actions.cuda()

        t = 0
        while t < episode_len:

            batch_legal_actions = get_legal_actions(states=bg, action_type=action_type, action_dropout=self.action_dropout)

            # epsilon greedy strategy
            _, _, _, Q_sa = self.model(bg, batch_legal_actions, action_type=action_type, gnn_step=gnn_step)

            best_actions = Q_sa.view(-1, num_actions).argmax(dim=1)
            explore_episode_indices = explore_replace_mask[explore_step_offset[t]: explore_step_offset[t + 1]][:, 1]
            explore_actions = explore_replace_actions[explore_step_offset[t]: explore_step_offset[t + 1]]
            best_actions[explore_episode_indices] = explore_actions

            best_actions += action_mask

            actions = batch_legal_actions[best_actions]

            # update bg inplace and calculate batch rewards
            g0 = [g for g in un_batch(bg)]  # current_state
            _, rewards = step_batch(states=bg, action=actions, action_type=action_type)
            g1 = [g for g in un_batch(bg)]  # after_state

            _rollout_rerward = torch.zeros((rollout_step))
            _rollout_action = torch.zeros((rollout_step*2)).int()
            if self.cuda_flag:
                _rollout_rerward = _rollout_rerward.cuda()
                _rollout_action = _rollout_action.cuda()
            if self.sample_batch_episode:
                self.experience_replay_buffer.extend([sars(g0[i], actions[i], rewards[i], g1[i], _rollout_rerward, _rollout_action) for i in range(batch_size)])
            else:  # using cascade buffer

                self.cascade_replay_buffer[t].extend([sars(g0[i], actions[i], rewards[i], g1[i], _rollout_rerward, _rollout_action) for i in range(batch_size)])

                if self.priority_sampling:
                    # compute prioritized weights
                    batch_legal_actions = get_legal_actions(states=bg, action_type=action_type, action_dropout=self.action_dropout)
                    _, _, _, Q_sa_next = self.model(bg, batch_legal_actions, action_type=action_type, gnn_step=gnn_step)
                    delta = Q_sa[best_actions] - (rewards + self.gamma * Q_sa_next.view(-1, num_actions).max(dim=1).values)
                    # delta = (Q_sa[best_actions] - (rewards + self.gamma * Q_sa_next.view(-1, num_actions).max(dim=1).values)) / (torch.clamp(torch.abs(Q_sa[best_actions]),0.1))
                    self.cascade_replay_buffer_weight[t, :batch_size] = torch.abs(delta.detach())
            R = [reward.item() for reward in rewards]
            sum_r += sum(R)

            t += 1

        self.log.add_item('tot_return', sum_r)

        return R

    def soft_target(self, Q_sa, batch_size, Temperature=0.1):
        mean_Q_sa = torch.mean(Q_sa.view(batch_size, -1), dim=1)
        return torch.log(torch.mean(torch.exp((Q_sa.view(batch_size, -1) - mean_Q_sa.unsqueeze(1)).clamp(-8, 8) / Temperature),
                             dim=1)) * Temperature + mean_Q_sa

    def sample_actions_from_q(self, prop_a_score, Q_sa, batch_size, Temperature=1.0, eps=None, top_k=1):
        num_actions = Q_sa.shape[0] // batch_size
        if self.explore_method == 'epsilon_greedy':
            # len = batch_size * topk  (g0_top1, g1_top1, ..., g0_top2, ...)
            best_actions = Q_sa.view(batch_size, num_actions).topk(k=top_k, dim=1).indices.t().flatten()

            # update prop_net
            b = best_actions.shape[0]
            L = -torch.log(prop_a_score[range(b), best_actions]+1e-8) + 1.0 * (torch.log(prop_a_score+1e-8) * prop_a_score).sum(dim=1)
            L_sum = L.sum()
            L_sum.backward(retain_graph=True)

            self.optimizer2.step()
            self.optimizer2.zero_grad()
            self.log.add_item('Act_Prop_error', L_sum.item())

        if self.explore_method == 'softmax' or self.explore_method == 'soft_dqn':

            best_actions = torch.multinomial(F.softmax(Q_sa.view(-1, num_actions) / Temperature), 1).view(-1)

        if eps is None:
            eps = self.eps
        explore_replace_mask = (torch.rand(batch_size * top_k) < eps).nonzero()
        explore_actions = torch.randint(high=num_actions, size=(explore_replace_mask.shape[0], ))
        if self.cuda_flag:
            explore_actions = explore_actions.cuda()
        best_actions[explore_replace_mask[:, 0]] = explore_actions
        # add action batch offset
        if self.cuda_flag:
            best_actions += torch.tensor(range(0, num_actions * batch_size, num_actions)).repeat(top_k).cuda()
        else:
            best_actions += torch.tensor(range(0, num_actions * batch_size, num_actions)).repeat(top_k)
        return best_actions

    def rollout(self, bg, rollout_step, top_num=5):

        # batch_size = self.new_epi_batch_size * self.buf_epi_len * top_num
        batch_size = bg.batch_size
        rollout_rewards = torch.zeros((batch_size, rollout_step))
        rollout_actions = torch.zeros((batch_size, rollout_step * 2)).int()
        if self.cuda_flag:
            rollout_rewards = rollout_rewards.cuda()
            rollout_actions = rollout_actions.cuda()

        for step in range(rollout_step):
            batch_legal_actions = get_legal_actions(states=bg, action_type=self.action_type, action_dropout=self.action_dropout)

            prop_a_score, prop_actions = self.model.forward_prop(bg, batch_legal_actions, action_type=self.action_type)

            _, _, _, Q_sa = self.model(bg, prop_actions, action_type=self.action_type)

            chosen_actions = self.sample_actions_from_q(prop_a_score, Q_sa, batch_size, eps=0.0, top_k=1)
            _actions = prop_actions[chosen_actions]

            # update bg inplace and calculate batch rewards
            _, _rewards = step_batch(states=bg, action_type=self.action_type, action=_actions)

            rollout_rewards[:, step] = _rewards
            rollout_actions[:, 2 * step:2 * step + 2] = _actions
            # print('step', step, rewards)
        return rollout_rewards, rollout_actions

    def run_cascade_episode(self, target_bg=None, action_type='swap', gnn_step=3, rollout_step=0, verbose=False, epoch=None):

        sum_r = 0

        T0 = time.time()

        # generate new start states
        if target_bg is None:
            if self.gen_training_sample_first:
                new_graphs = make_batch(np.random.choice(self.training_instances, self.new_epi_batch_size, replace=False))
            else:
                new_graphs = self.graph_generator.generate_graph(batch_size=self.new_epi_batch_size, cuda_flag=self.cuda_flag)

            if epoch is not None:
                # shift the starting point of an episode
                self.rollout(bg=new_graphs, rollout_step=epoch // 2000)

            new_graphs = un_batch(new_graphs, copy=False)

            self.log.add_item('S_new_training_sample', torch.mean(torch.cat([new_graphs[i].kcut_value for i in range(self.new_epi_batch_size)])).item())

        else:
            assert target_bg.in_cuda == self.cuda_flag
            new_graphs = dc(target_bg)
            perm_weight(new_graphs)

        if verbose:
            T1 = time.time(); print('t1', T1 - T0)

        # extend previous states(no memory copy here)
        new_graphs.extend(list(itertools.chain(*[[tpl.s1 for tpl in self.cascade_replay_buffer[i][-self.new_epi_batch_size:]] for i in range(self.buf_epi_len-1)])))
        if verbose:
            T2 = time.time(); print('t2', T2 - T1)

        # make batch and copy new states
        bg = make_batch(new_graphs)
        if verbose:
            T3 = time.time(); print('t3', T3 - T2)

        batch_size = self.new_epi_batch_size * self.buf_epi_len

        if verbose:
            T4 = time.time(); print('t4', T4 - T3)

        batch_legal_actions = get_legal_actions(states=bg, action_type=self.action_type, action_dropout=self.action_dropout)
        if verbose:
            T5 = time.time(); print('t5', T5 - T4)
        # epsilon greedy strategy
        # TODO: multi-gpu parallelization

        prop_a_score, prop_actions = self.model.forward_prop(bg, batch_legal_actions, action_type=self.action_type)
        _, _, _, Q_sa = self.model(bg, prop_actions, action_type=self.action_type)

        if verbose:
            T6 = time.time(); print('t6', T6 - T5)

        if not rollout_step:
            # TODO: can alter explore strength according to kcut_valueS
            chosen_actions = self.sample_actions_from_q(prop_a_score, Q_sa, batch_size, Temperature=self.eps)
            actions = prop_actions[chosen_actions]
            rollout_rewards = torch.zeros(batch_size, 1)
            rollout_actions = torch.zeros(batch_size, 2).int()
        else:
            top_num = 1  # rollout for how many top actions
            rollout_bg = make_batch([bg] * top_num)

            # chosen_actions - len = batch_size * topk
            chosen_actions = self.sample_actions_from_q(prop_a_score, Q_sa, batch_size, Temperature=self.eps, top_k=top_num)

            topk_actions = prop_actions[chosen_actions]

            bg1, rewards1 = step_batch(states=rollout_bg, action_type=action_type, action=topk_actions)

            rollout_rewards, rollout_actions = self.rollout(bg=bg1, rollout_step=rollout_step, top_num=top_num)

            # select actions based on rollout rewards
            # rollout_selected_actions = torch.cat([rewards1.view(-1, 1), rollout_rewards], dim=1)\
            rollout_selected_actions = torch.cat([rollout_rewards], dim=1)\
                .cumsum(dim=1).max(dim=1)\
                .values.view(top_num, -1)\
                .argmax(dim=0) * batch_size
            if self.cuda_flag:
                rollout_selected_actions += torch.tensor(range(batch_size)).cuda()
            else:
                rollout_selected_actions += torch.tensor(range(batch_size))

            # update bg inplace and calculate batch rewards
            actions = topk_actions[rollout_selected_actions, :]
            # rewards = rewards1[rollout_selected_actions]
            rollout_rewards = rollout_rewards[rollout_selected_actions, :]
            rollout_actions = rollout_actions[rollout_selected_actions, :]

        # update bg inplace and calculate batch rewards
        _, rewards = step_batch(states=bg, action_type=action_type, action=actions)

        g0 = new_graphs  # current_state
        g1 = un_batch(bg, copy=False)  # after_state

        [self.cascade_replay_buffer[t].extend(
            [sars(g0[j+t*self.new_epi_batch_size]
            , actions[j+t*self.new_epi_batch_size]
            , rewards[j+t*self.new_epi_batch_size] #+ (4/5)**t
            , g1[j+t*self.new_epi_batch_size]
            , rollout_rewards[j+t*self.new_epi_batch_size, :]
            , rollout_actions[j+t*self.new_epi_batch_size, :])
            for j in range(self.new_epi_batch_size)])
         for t in range(self.buf_epi_len)]

        if self.priority_sampling:
            # compute prioritized weights
            batch_legal_actions = get_legal_actions(states=bg, action_type=action_type, action_dropout=self.action_dropout)
            _, _, _, Q_sa_next = self.model(bg, batch_legal_actions, action_type=action_type, gnn_step=gnn_step)

            delta = Q_sa[chosen_actions] - (rewards + self.gamma * Q_sa_next.view(-1, num_actions).max(dim=1).values)
            # delta = (Q_sa[chosen_actions] - (rewards + self.gamma * Q_sa_next.view(-1, num_actions).max(dim=1).values)) / torch.clamp(torch.abs(Q_sa[chosen_actions]),0.1)
            self.cascade_replay_buffer_weight = torch.cat([self.cascade_replay_buffer_weight, torch.abs(delta.detach().cpu().view(self.buf_epi_len, self.new_epi_batch_size))], dim=1).detach()
            # print(self.cascade_replay_buffer_weight)

        R = [reward.item() for reward in rewards]
        sum_r += sum(R)

        self.log.add_item('tot_return', sum_r)


        return R

    def sample_from_buffer(self, batch_size, q_step, gnn_step):

        batch_size = min(batch_size, len(self.experience_replay_buffer))

        sample_buffer = np.random.choice(self.experience_replay_buffer, batch_size, replace=False)
        # make batches
        batch_begin_state = make_batch([tpl.s0 for tpl in sample_buffer])
        batch_end_state = make_batch([tpl.s1 for tpl in sample_buffer])
        R = [tpl.r.unsqueeze(0) for tpl in sample_buffer]
        batch_begin_action = torch.cat([tpl.a.unsqueeze(0) for tpl in sample_buffer], axis=0)
        batch_end_action = get_legal_actions(state=batch_end_state, action_type=self.action_type, action_dropout=self.action_dropout)
        action_num = batch_end_action.shape[0] // batch_begin_action.shape[0]

        # only compute limited number for Q_s1a
        # TODO: multi-gpu parallelization
        _, _, _, Q_s1a_ = self.model(batch_begin_state, batch_begin_action, action_type=self.action_type, gnn_step=gnn_step)
        _, _, _, Q_s2a = self.model_target(batch_end_state, batch_end_action, action_type=self.action_type, gnn_step=gnn_step)


        q = self.gamma ** q_step * Q_s2a.view(-1, action_num).max(dim=1).values - Q_s1a_
        Q = q.unsqueeze(0)

        return torch.cat(R), Q

    def sample_from_cascade_buffer(self, batch_size, q_step, gnn_step, rollout_step=0):

        batch_size = min(batch_size, len(self.cascade_replay_buffer[0]) * self.buf_epi_len)

        batch_sizes = [
            min(batch_size * self.stage_max_sizes[i] // self.buffer_actual_size, len(self.cascade_replay_buffer[0]))
            for i in range(self.buf_epi_len)]
        sample_buffer = list(itertools.chain(*[np.random.choice(a=self.cascade_replay_buffer[i]
                                                            , size=batch_sizes[i]
                                                            , replace=False
                                                            ) for i in range(self.buf_epi_len)]))

        # make batches
        batch_begin_state = make_batch([tpl.s0 for tpl in sample_buffer])
        batch_end_state = make_batch([tpl.s1 for tpl in sample_buffer])
        R = torch.cat([tpl.r.unsqueeze(0) for tpl in sample_buffer])
        if self.cuda_flag:
            R = R.cuda()

        if rollout_step:
            rollout_R = torch.cat([R.unsqueeze(1), torch.cat([tpl.rollout_r.unsqueeze(0) for tpl in sample_buffer])], dim=1)
            R = rollout_R[:, :q_step].sum(dim=1)
            rollout_A = torch.cat([tpl.rollout_a.unsqueeze(0) for tpl in sample_buffer])
            step_batch(states=batch_end_state, action=rollout_A[:, 0:2 * (q_step - 1)], action_type=self.action_type)

        batch_begin_action = torch.cat([tpl.a.unsqueeze(0) for tpl in sample_buffer], axis=0)
        _, reconstruct_S, _, Q_s1a_ = self.model(batch_begin_state, batch_begin_action, action_type=self.action_type, gnn_step=gnn_step)

        #  foward the end state for (q_step - 1) steps

        batch_end_action = get_legal_actions(states=batch_end_state, action_type=self.action_type, action_dropout=self.action_dropout)

        #  action proposal network
        prop_a_score, prop_actions = self.model_target.forward_prop(batch_end_state, batch_end_action, action_type=self.action_type)

        _, _, _, Q_s2a = self.model_target(batch_end_state, prop_actions, action_type=self.action_type, gnn_step=gnn_step)

        chosen_actions = self.sample_actions_from_q(prop_a_score, Q_s2a, batch_size, Temperature=self.eps)

        q = self.gamma ** q_step * Q_s2a[chosen_actions].detach() - Q_s1a_

        Q = q.unsqueeze(0)

        return R, Q, reconstruct_S

    def back_loss(self, R, Q, err_S, update_model=True):

        R = R.cuda(device=Q.device)
        L_dqn = torch.pow(R + Q, 2).sum()
        L = L_dqn
        L.backward(retain_graph=False)

        self.Q_err += L_dqn.item()

        if update_model:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.log.add_item('Q_error', self.Q_err)
            self.log.add_item('Reconstruction_error', 0)
            self.Q_err = 0
            self.log.add_item('entropy', 0)

    def train_dqn(self, target_bg=None, epoch=0, batch_size=16, num_episodes=10, episode_len=50, gnn_step=10, q_step=1, grad_accum=1, rollout_step=0, ddqn=False):
        """
        :param batch_size:
        :param num_episodes:
        :param episode_len: #steps in each episode
        :param gnn_step: #iters when running gnn
        :param q_step: reward delay step
        :param ddqn: train in ddqn mode
        :return:
        """
        if self.sample_batch_episode:
            T3 = time.time()
            self.run_batch_episode(action_type=self.action_type, gnn_step=gnn_step, episode_len=episode_len,
                                   batch_size=num_episodes)
            T4 = time.time()

            # trim experience replay buffer
            self.trim_replay_buffer(epoch)

            R, Q = self.sample_from_buffer(batch_size=batch_size, q_step=q_step, gnn_step=gnn_step)

            T6 = time.time()
        else:
            T3 = time.time()
            if epoch == 0:
                # buf_epi_len calls of model(new_epi_batch_size) * action_num
                self.run_batch_episode(target_bg=target_bg, action_type=self.action_type, gnn_step=gnn_step, episode_len=self.buf_epi_len,
                                   batch_size=self.new_epi_batch_size, rollout_step=rollout_step)
            else:
                # 1 call of model(buf_epi_len * new_epi_batch_size) * action_num
                self.run_cascade_episode(target_bg=target_bg, action_type=self.action_type, gnn_step=gnn_step, rollout_step=rollout_step, epoch=None)
            T4 = time.time()
            # trim experience replay buffer
            self.trim_replay_buffer(epoch)

            # 1 call of model(batch_size) * action_num + 1 call of model(batch_size)
            R, Q, err_S = self.sample_from_cascade_buffer(batch_size=batch_size, q_step=q_step, rollout_step=rollout_step, gnn_step=gnn_step)

            T6 = time.time()

            for _ in range(grad_accum - 1):
                self.back_loss(R, Q, err_S, update_model=False)
                del R, Q
                torch.cuda.empty_cache()
                R, Q, err_S = self.sample_from_cascade_buffer(batch_size=batch_size, q_step=q_step, rollout_step=rollout_step,
                                                       gnn_step=gnn_step)
            self.back_loss(R, Q, err_S, update_model=True)
            del R, Q, err_S
            torch.cuda.empty_cache()
            T7 = time.time()


            self._updata_lr(step=epoch, max_lr=2e-3, min_lr=1e-3, decay_step=10000)

            print('Rollout time:', T4-T3)
            print('Sample and forward time', T6-T4)
            print('Backloss time', T7-T6)

        return self.log

    def trim_replay_buffer(self, epoch):
        if len(self.experience_replay_buffer) > self.replay_buffer_max_size:
            self.experience_replay_buffer = self.experience_replay_buffer[-self.replay_buffer_max_size:]

        if epoch * self.buf_epi_len * self.new_epi_batch_size > self.replay_buffer_max_size:
            for i in range(self.buf_epi_len):
                self.cascade_replay_buffer[i] = self.cascade_replay_buffer[i][-self.stage_max_sizes[i]:]

    def update_target_net(self):
        self.model_target.load_state_dict(self.model.state_dict())
