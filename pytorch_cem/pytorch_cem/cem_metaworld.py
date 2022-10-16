import torch
import os
import time
import logging
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import imageio

logger = logging.getLogger(__name__)


def pytorch_cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w / w_sum)[:, None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()


class CEM():
    """
    Cross Entropy Method control
    This implementation batch samples the trajectories and so scales well with the number of samples K.
    """

    def __init__(self, dynamics, running_cost, nx, nu, num_samples=100, num_iterations=3, num_elite=10, horizon=15,
                 device="cpu",
                 terminal_state_cost=None,
                 u_min=None,
                 u_max=None,
                 choose_best=False,
                 init_cov_diag=1):
        """

        :param dynamics: function(state, action) -> next_state (K x nx) taking in batch state (K x nx) and action (K x nu)
        :param running_cost: function(state, action) -> cost (K x 1) taking in batch state and action (same as dynamics)
        :param nx: state dimension
        :param num_samples: K, number of trajectories to sample
        :param horizon: T, length of each trajectory
        :param device: pytorch device
        :param terminal_state_cost: function(state) -> cost (K x 1) taking in batch state
        """
        self.d = device
        self.dtype = torch.double  # TODO determine dtype
        self.K = num_samples  # N_SAMPLES
        self.T = horizon  # TIMESTEPS
        self.M = num_iterations
        self.num_elite = num_elite
        self.choose_best = choose_best

        # dimensions of state and control
        self.nx = nx
        self.nu = nu

        self.mean = None
        self.cov = None

        self.F = dynamics
        self.running_cost = running_cost
        self.terminal_state_cost = terminal_state_cost
        self.init_cov_diag = init_cov_diag
        self.u_min = u_min
        self.u_max = u_max
        # make sure if any of them is specified, both are specified
        if self.u_max is not None and self.u_min is None:
            self.u_min = -self.u_max
        if self.u_min is not None and self.u_max is None:
            self.u_max = -self.u_min
        if self.u_min is not None:
            self.u_min = self.u_min.to(device=self.d)
            self.u_max = self.u_max.to(device=self.d)
        self.action_distribution = None

        # regularize covariance
        self.cov_reg = torch.eye(self.T * self.nu, device=self.d, dtype=self.dtype) * init_cov_diag * 1e-5

    def reset(self):
        """
        Clear controller state after finishing a trial
        """
        # action distribution, initialized as N(0,I)
        # we do Hp x 1 instead of H x p because covariance will be Hp x Hp matrix instead of some higher dim tensor
        self.mean = torch.zeros(self.T * self.nu, device=self.d, dtype=self.dtype)
        self.cov = torch.eye(self.T * self.nu, device=self.d, dtype=self.dtype) * self.init_cov_diag

    def _bound_samples(self, samples):
        if self.u_max is not None:
            for t in range(self.T):
                u = samples[:, self._slice_control(t)]
                cu = torch.max(torch.min(u, self.u_max), self.u_min)
                samples[:, self._slice_control(t)] = cu
        return samples

    def _slice_control(self, t):
        return slice(t * self.nu, (t + 1) * self.nu)

    def _evaluate_trajectories(self, samples, init_state, env):
        cost_total = torch.zeros(self.K, device=self.d, dtype=self.dtype)
        # state = init_state.view(1, -1).repeat(self.K, 1)
        # for t in range(self.T):
        #     u = samples[:, self._slice_control(t)]
        #     state = self.F(state, u)
        #     cost_total += self.running_cost(state, u).to(self.dtype)
        saved_env_state = env.get_env_state()
        saved_path_length = env.cur_path_length
        for k in range(self.K):
            for t in range(saved_path_length, self.T):
                u = samples[k, self._slice_control(t)]
                obs, _, _, _ = env.step(u.cpu().numpy().squeeze())
                curr_state = torch.Tensor(tabletop_obs(env._get_low_dim_info())).to(self.d).unsqueeze(0)
                cost_total += self.running_cost(curr_state, u).to(self.dtype).to(self.d)

            if self.terminal_state_cost:
                cost_total += self.terminal_state_cost(curr_state, _).to(self.dtype).to(self.d)
            
            env.set_env_state(saved_env_state)
            env.cur_path_length = saved_path_length

        return cost_total

    def _sample_top_trajectories(self, state, num_elite, env):
        # sample K action trajectories
        # in case it's singular
        self.action_distribution = MultivariateNormal(self.mean, covariance_matrix=self.cov)
        samples = self.action_distribution.sample((self.K,))
        # bound to control maximums
        samples = self._bound_samples(samples)

        cost_total = self._evaluate_trajectories(samples, state, env)
        # select top k based on score
        top_costs, topk = torch.topk(cost_total, num_elite, largest=False, sorted=False)
        top_samples = samples[topk]
        return top_samples, cost_total.mean()

    def command(self, state, env, choose_best=False):
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        state = state.to(dtype=self.dtype, device=self.d)

        self.reset()

        average_sampled_cost = torch.zeros(self.M)
        for m in range(self.M):
            top_samples, average_cost = self._sample_top_trajectories(state, self.num_elite, env)
            # fit the gaussian to those samples
            self.mean = torch.mean(top_samples, dim=0)
            self.cov = pytorch_cov(top_samples, rowvar=False)
            if torch.matrix_rank(self.cov) < self.cov.shape[0]:
                self.cov += self.cov_reg
            average_sampled_cost[m] = average_cost

        if choose_best and self.choose_best:
            top_sample = self._sample_top_trajectories(state, 1)
        else:
            top_sample = self.action_distribution.sample((1,))

        # only apply the first action from this trajectory
        u = top_sample[0, self._slice_control(0)]

        return u, average_sampled_cost.mean()

def tabletop_obs(info):
    hand = np.array([info['hand_x'], info['hand_y'], info['hand_z']])
    mug = np.array([info['mug_x'], info['mug_y'], info['mug_z']])
    mug_quat = np.array([info['mug_quat_x'], info['mug_quat_y'], info['mug_quat_z'], info['mug_quat_w']])
    init_low_dim = np.concatenate([hand, mug, mug_quat, [info['drawer']], [info['coffee_machine']], [info['faucet']]])
    return init_low_dim

def evaluate_iteration(low_dim_state, very_start, task_num):
    right_to_left = low_dim_state[3:4] - very_start[3:4]
    left_to_right = -low_dim_state[3:4] + very_start[3:4]
    forward = low_dim_state[4:5] - very_start[4:5]
    drawer_move = np.abs(low_dim_state[10] - very_start[10])
    drawer_open = np.abs(low_dim_state[10])
    move_faucet = low_dim_state[12]

    if task_num == 5: # close drawer
        criteria = drawer_open < 0.01 and np.abs(right_to_left) < 0.01
    elif task_num == 94: # move cup right to left
        criteria = left_to_right < -0.05 # and drawer_move < 0.03 and move_faucet < 0.01
    elif task_num == 45: # Push cup forward
        criteria = abs(right_to_left) < 0.04 and forward > 0.1
    else:
        print("No criteria set")
        assert(False)

    try:
        return criteria[0]
    except:
        return criteria

def run_cem_metaworld(cem, env, retrain_dynamics, task_num, terminal_cost, logdir, retrain_after_iter=50, iter=1000, use_gt=False, render=True, choose_best=False):
    dataset = torch.zeros((retrain_after_iter, cem.nx + cem.nu), dtype=cem.dtype, device=cem.d)
    
    # for visualizing CEM iterations
    logdir_iteration = os.path.join(logdir, 'iteration')
    if not os.path.isdir(logdir_iteration):
        os.makedirs(logdir_iteration)

    iteration_reward = 0

    total_successes = 0
    total_iterations = 0
    _, start_info = env.reset_model()
    very_start = tabletop_obs(start_info)
    states = []
    actions = []
    
    rolling_success = deque()
    succ_rate_history = []
    dvd_reward_history = []
    engineered_reward_history = []
    average_samples_reward_history = []

    all_obs = []
    for i in range(iter):
        if use_gt:
            low_dim_info = env._get_low_dim_info()
            state = tabletop_obs(low_dim_info)
        else:
            state = env.get_obs().flatten()

        command_start = time.perf_counter()
        action, average_sampled_cost = cem.command(state, env, choose_best=choose_best)
        elapsed = time.perf_counter() - command_start
        s, _, done, low_dim_info = env.step(action.cpu().numpy())
        states.append(s)
        actions.append(action.cpu().numpy())

        all_obs.append((s * 255).astype(np.uint8))
        average_samples_reward_history.append(-average_sampled_cost)

        r = tabletop_obs(low_dim_info)[3] - very_start[3]
        iteration_reward += r
        
        logger.debug(f"{i}: state reward: {r:.6f} action taken: {action} time taken: {elapsed:.5f}s")
        if render:
            env.render()

        plt.figure()
        plt.plot([i for i in range(len(average_samples_reward_history))], average_samples_reward_history)
        plt.xlabel('Iteration')
        plt.ylabel('Mean Trajectory Reward')
        plt.title(f'Mean Reward of Sampled Trajectories')
        plt.savefig(os.path.join(logdir, 'mean_reward_sampled_traj_iteration.png'))
        plt.close()
        
        if done:
            low_dim_info = env._get_low_dim_info()
            low_dim_state = tabletop_obs(low_dim_info)
            succ = evaluate_iteration(low_dim_state, very_start, task_num)

            states_reshape = torch.from_numpy(np.stack(states))
            states_reshape = torch.reshape(states_reshape, (states_reshape.shape[0], -1)).unsqueeze(0).unsqueeze(0)
            dvd_reward = -terminal_cost(states_reshape, actions).item()
            engineered_reward = low_dim_state[3] # task 94 specific

            result = 'SUCCESS' if succ else 'FAILURE'
            total_successes += succ
            total_iterations += 1
            rolling_success.append(succ)
            print(f'----------Iteration done: {result} | dvd_reward: {dvd_reward} | engineered_reward: {engineered_reward}----------')
            print(f'----------Currently at {total_successes} / {total_iterations}----------')
            
            if len(rolling_success) > 10:
                rolling_success.popleft()
            
            succ_rate = sum(rolling_success) / len(rolling_success)

            dvd_reward_history.append(dvd_reward)
            engineered_reward_history.append(engineered_reward)
            succ_rate_history.append(succ_rate)

            print('----------REPLOTTING----------')
            plt.figure()
            plt.plot([i for i in range(len(dvd_reward_history))], dvd_reward_history)
            plt.ylim([0, 1])
            plt.xlabel('Iteration')
            plt.ylabel('DVD Reward')
            plt.title('DVD Reward')
            plt.savefig(os.path.join(logdir, 'dvd_rewards_iteration.png'))
            plt.close()

            plt.figure()
            plt.plot([i for i in range(len(engineered_reward_history))], engineered_reward_history)
            plt.xlabel('Iteration')
            plt.ylabel('Engineered Reward')
            plt.title('Engineered Reward')
            plt.savefig(os.path.join(logdir, 'engineered_reward_iteration.png'))
            plt.close()
            
            plt.figure()
            plt.plot([i for i in range(len(succ_rate_history))], succ_rate_history)
            plt.xlabel('Iteration')
            plt.ylabel('Rolling Success Rate')
            plt.title('Rolling Success Rate')
            plt.savefig(os.path.join(logdir, 'rolling_success_rate_iteration.png'))
            plt.close()

            # store video of path
            imageio.mimsave(os.path.join(logdir_iteration, f'iteration{total_iterations}.gif'), all_obs, fps=20)
            
            _, start_info = env.reset_model()
            very_start = tabletop_obs(start_info)
            states = []
            actions = []
            iteration_reward = 0
            all_obs = []

        di = i % retrain_after_iter
        if di == 0 and i > 0:
            retrain_dynamics(dataset)
            # don't have to clear dataset since it'll be overridden, but useful for debugging
            dataset.zero_()
        dataset[di, :cem.nx] = torch.tensor(state, dtype=cem.dtype)
        dataset[di, cem.nx:] = action
    return total_successes, total_iterations, dataset
