import os
import torch
import time
import logging
from torch.distributions.multivariate_normal import MultivariateNormal
import functools
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').disabled = True


def _ensure_non_zero(cost, beta, factor):
    return torch.exp(-factor * (cost - beta))


def is_tensor_like(x):
    return torch.is_tensor(x) or type(x) is np.ndarray


# from arm_pytorch_utilities, standalone since that package is not on pypi yet
def handle_batch_input(func):
    """For func that expect 2D input, handle input that have more than 2 dimensions by flattening them temporarily"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # assume inputs that are tensor-like have compatible shapes and is represented by the first argument
        batch_dims = []
        for arg in args:
            if is_tensor_like(arg) and len(arg.shape) > 2:
                batch_dims = arg.shape[:-1]  # last dimension is type dependent; all previous ones are batches
                break
        # no batches; just return normally
        if not batch_dims:
            return func(*args, **kwargs)

        # reduce all batch dimensions down to the first one
        args = [v.view(-1, v.shape[-1]) if (is_tensor_like(v) and len(v.shape) > 2) else v for v in args]
        ret = func(*args, **kwargs)
        # restore original batch dimensions; keep variable dimension (nx)
        if type(ret) is tuple:
            ret = [v if (not is_tensor_like(v) or len(v.shape) == 0) else (
                v.view(*batch_dims, v.shape[-1]) if len(v.shape) == 2 else v.view(*batch_dims)) for v in ret]
        else:
            if is_tensor_like(ret):
                if len(ret.shape) == 2:
                    ret = ret.view(*batch_dims, ret.shape[-1])
                else:
                    ret = ret.view(*batch_dims)
        return ret

    return wrapper


class MPPI():
    """
    Model Predictive Path Integral control
    This implementation batch samples the trajectories and so scales well with the number of samples K.

    Implemented according to algorithm 2 in Williams et al., 2017
    'Information Theoretic MPC for Model-Based Reinforcement Learning',
    based off of https://github.com/ferreirafabio/mppi_pendulum
    """

    def __init__(self, dynamics, running_cost, nx, noise_sigma, num_samples=100, horizon=15, device="cpu",
                 terminal_state_cost=None,
                 lambda_=1.,
                 noise_mu=None,
                 u_min=None,
                 u_max=None,
                 u_init=None,
                 U_init=None,
                 u_scale=1,
                 u_per_command=1,
                 step_dependent_dynamics=False,
                 rollout_samples=1,
                 rollout_var_cost=0,
                 rollout_var_discount=0.95,
                 sample_null_action=False,
                 noise_abs_cost=False,
                 online_sampling=True):
        """
        :param dynamics: function(state, action) -> next_state (K x nx) taking in batch state (K x nx) and action (K x nu)
        :param running_cost: function(state, action) -> cost (K) taking in batch state and action (same as dynamics)
        :param nx: state dimension
        :param noise_sigma: (nu x nu) control noise covariance (assume v_t ~ N(u_t, noise_sigma))
        :param num_samples: K, number of trajectories to sample
        :param horizon: T, length of each trajectory
        :param device: pytorch device
        :param terminal_state_cost: function(state) -> cost (K x 1) taking in batch state
        :param lambda_: temperature, positive scalar where larger values will allow more exploration
        :param noise_mu: (nu) control noise mean (used to bias control samples); defaults to zero mean
        :param u_min: (nu) minimum values for each dimension of control to pass into dynamics
        :param u_max: (nu) maximum values for each dimension of control to pass into dynamics
        :param u_init: (nu) what to initialize new end of trajectory control to be; defeaults to zero
        :param U_init: (T x nu) initial control sequence; defaults to noise
        :param step_dependent_dynamics: whether the passed in dynamics needs horizon step passed in (as 3rd arg)
        :param rollout_samples: M, number of state trajectories to rollout for each control trajectory
            (should be 1 for deterministic dynamics and more for models that output a distribution)
        :param rollout_var_cost: Cost attached to the variance of costs across trajectory rollouts
        :param rollout_var_discount: Discount of variance cost over control horizon
        :param sample_null_action: Whether to explicitly sample a null action (bad for starting in a local minima)
        :param noise_abs_cost: Whether to use the absolute value of the action noise to avoid bias when all states have the same cost
        """
        self.d = device
        self.dtype = noise_sigma.dtype
        self.K = num_samples  # N_SAMPLES
        self.T = horizon  # TIMESTEPS

        # dimensions of state and control
        self.nx = nx
        self.nu = 1 if len(noise_sigma.shape) == 0 else noise_sigma.shape[0]
        self.lambda_ = lambda_

        if noise_mu is None:
            noise_mu = torch.zeros(self.nu, dtype=self.dtype)

        if u_init is None:
            u_init = torch.zeros_like(noise_mu)

        # handle 1D edge case
        if self.nu == 1:
            noise_mu = noise_mu.view(-1)
            noise_sigma = noise_sigma.view(-1, 1)

        # bounds
        self.u_min = u_min
        self.u_max = u_max
        self.u_scale = u_scale
        self.u_per_command = u_per_command
        # make sure if any of them is specified, both are specified
        if self.u_max is not None and self.u_min is None:
            if not torch.is_tensor(self.u_max):
                self.u_max = torch.tensor(self.u_max)
            self.u_min = -self.u_max
        if self.u_min is not None and self.u_max is None:
            if not torch.is_tensor(self.u_min):
                self.u_min = torch.tensor(self.u_min)
            self.u_max = -self.u_min
        if self.u_min is not None:
            self.u_min = self.u_min.to(device=self.d)
            self.u_max = self.u_max.to(device=self.d)

        self.noise_mu = noise_mu.to(self.d)
        self.noise_sigma = noise_sigma.to(self.d)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)
        # T x nu control sequence
        self.U = U_init
        self.u_init = u_init.to(self.d)

        if self.U is None:
            self.U = self.noise_dist.sample((self.T,))

        self.step_dependency = step_dependent_dynamics
        self.F = dynamics
        self.online_sampling = online_sampling
        self.running_cost = running_cost
        self.terminal_state_cost = terminal_state_cost
        self.sample_null_action = sample_null_action
        self.noise_abs_cost = noise_abs_cost
        self.state = None

        # handling dynamics models that output a distribution (take multiple trajectory samples)
        self.M = rollout_samples
        self.rollout_var_cost = rollout_var_cost
        self.rollout_var_discount = rollout_var_discount

        # sampled results from last command
        self.cost_total = None
        self.cost_total_non_zero = None
        self.omega = None
        self.states = None
        self.actions = None

    @handle_batch_input
    def _dynamics(self, state, u, t):
        return self.F(state, u, t) if self.step_dependency else self.F(state, u)

    @handle_batch_input
    def _running_cost(self, state, u):
        return self.running_cost(state, u)

    def command(self, state, env):
        """
        :param state: (nx) or (K x nx) current state, or samples of states (for propagating a distribution of states)
        :returns action: (nu) best action
        """
        # shift command 1 time step
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = self.u_init

        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state.to(dtype=self.dtype, device=self.d)

        cost_total = self._compute_total_cost_batch(env)

        beta = torch.min(cost_total)
        self.cost_total_non_zero = _ensure_non_zero(cost_total, beta, 1 / self.lambda_)

        eta = torch.sum(self.cost_total_non_zero)
        self.omega = (1. / eta) * self.cost_total_non_zero
        for t in range(self.T):
            self.U[t] += torch.sum(self.omega.view(-1, 1) * self.noise[:, t], dim=0)
        action = self.U[:self.u_per_command]
        # reduce dimensionality if we only need the first command
        if self.u_per_command == 1:
            action = action[0]

        return action

    def reset(self):
        """
        Clear controller state after finishing a trial
        """
        self.U = self.noise_dist.sample((self.T,))

    def _compute_rollout_costs(self, perturbed_actions, env):
        K, T, nu = perturbed_actions.shape
        assert nu == self.nu

        cost_total = torch.zeros(K, device=self.d, dtype=self.dtype)
        cost_samples = cost_total.repeat(self.M, 1)
        cost_var = torch.zeros_like(cost_total)

        # allow propagation of a sample of states (ex. to carry a distribution), or to start with a single state
        if self.state.shape == (K, self.nx):
            state = self.state
        else:
            state = self.state.view(1, -1).repeat(K, 1)

        # rollout action trajectory M times to estimate expected cost
        state = state.repeat(self.M, 1, 1)

        states = torch.zeros(K, T, self.nx)
        actions = torch.zeros(K, T, nu)

        # to use for online sampling
        saved_env_state = env.get_env_state()
        saved_path_length = env.cur_path_length
        for k in range(K):
            curr_state = state[:, k, :]
            for t in range(T):
                u = self.u_scale * perturbed_actions[k, t].repeat(self.M, 1, 1)
                obs, _, _, _ = env.step(u.cpu().numpy().squeeze())
                # curr_state = self._dynamics(curr_state, u, t)
                curr_state = torch.Tensor(tabletop_obs(env._get_low_dim_info())).to(self.d).unsqueeze(0)
                c = self._running_cost(curr_state, u).to(self.dtype)
                cost_samples += c
                if self.M > 1:
                    cost_var += c.var(dim=0) * (self.rollout_var_discount ** t)

                # Save total states/actions
                states[k, t] = curr_state
                actions[k, t] = u

            # reset to saved_env_state
            env.set_env_state(saved_env_state)
            env.cur_path_length = saved_path_length

        # action perturbation cost
        if self.terminal_state_cost:
            c = self.terminal_state_cost(states, actions)
            cost_samples += c.to(self.dtype).to(cost_samples.device)
        cost_total += cost_samples.mean(dim=0)
        cost_total += cost_var * self.rollout_var_cost
        return cost_total, states, actions

    def _compute_total_cost_batch(self, env):
        # parallelize sampling across trajectories
        # resample noise each time we take an action
        self.noise = self.noise_dist.sample((self.K, self.T))
        # broadcast own control to noise over samples; now it's K x T x nu
        self.perturbed_action = self.U + self.noise
        if self.sample_null_action:
            self.perturbed_action[self.K - 1] = 0
        # naively bound control
        self.perturbed_action = self._bound_action(self.perturbed_action)
        # bounded noise after bounding (some got cut off, so we don't penalize that in action cost)
        self.noise = self.perturbed_action - self.U
        if self.noise_abs_cost:
            action_cost = self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv
            # NOTE: The original paper does self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv, but this biases
            # the actions with low noise if all states have the same cost. With abs(noise) we prefer actions close to the
            # nomial trajectory.
        else:
            action_cost = self.lambda_ * self.noise @ self.noise_sigma_inv # Like original paper

        self.cost_total, self.states, self.actions = self._compute_rollout_costs(self.perturbed_action, env)
        self.actions /= self.u_scale

        # action perturbation cost
        perturbation_cost = torch.sum(self.U * action_cost, dim=(1, 2))
        self.cost_total += perturbation_cost
        return self.cost_total

    def _bound_action(self, action):
        if self.u_max is not None:
            for t in range(self.T):
                u = action[:, self._slice_control(t)]
                cu = torch.max(torch.min(u, self.u_max), self.u_min)
                action[:, self._slice_control(t)] = cu
        return action

    def _slice_control(self, t):
        return slice(t * self.nu, (t + 1) * self.nu)

    def get_rollouts(self, state, num_rollouts=1):
        """
            :param state: either (nx) vector or (num_rollouts x nx) for sampled initial states
            :param num_rollouts: Number of rollouts with same action sequence - for generating samples with stochastic
                                 dynamics
            :returns states: num_rollouts x T x nx vector of trajectories

        """
        state = state.view(-1, self.nx)
        if state.size(0) == 1:
            state = state.repeat(num_rollouts, 1)

        T = self.U.shape[0]
        states = torch.zeros((num_rollouts, T + 1, self.nx), dtype=self.U.dtype, device=self.U.device)
        states[:, 0] = state
        for t in range(T):
            states[:, t + 1] = self._dynamics(states[:, t].view(num_rollouts, -1),
                                              self.u_scale * self.U[t].view(num_rollouts, -1), t)
        return states[:, 1:]

def tabletop_obs(info):
    hand = np.array([info['hand_x'], info['hand_y'], info['hand_z']])
    mug = np.array([info['mug_x'], info['mug_y'], info['mug_z']])
    mug_quat = np.array([info['mug_quat_x'], info['mug_quat_y'], info['mug_quat_z'], info['mug_quat_w']])
    init_low_dim = np.concatenate([hand, mug, mug_quat, [info['drawer']], [info['coffee_machine']], [info['faucet']]])
    return init_low_dim

def evaluate_episode(low_dim_state, very_start, task_num):
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

def run_mppi_metaworld(mppi, env, retrain_dynamics, task_num, terminal_cost, logdir, retrain_after_iter=50, iter=1000, use_gt=False, render=True):
    dataset = torch.zeros((retrain_after_iter, mppi.nx + mppi.nu), dtype=mppi.U.dtype, device=mppi.d)
    total_reward = 0

    total_successes = 0
    total_episodes = 0
    _, start_info = env.reset_model()
    very_start = tabletop_obs(start_info)
    states = []
    actions = []
    rolling_success = deque()
    dvd_reward_history = []
    engineered_reward_history = []
    succ_rate_history = []
    for i in range(iter):
        if use_gt:
            low_dim_info = env._get_low_dim_info()
            state = tabletop_obs(low_dim_info)
        else:
            state = env.get_obs().flatten()

        command_start = time.perf_counter()
        action = mppi.command(state, env)
        # import ipdb; ipdb.set_trace()
        elapsed = time.perf_counter() - command_start

        s, r, done, _ = env.step(action.cpu().numpy())
        states.append(s)
        actions.append(action.cpu().numpy())
        total_reward += r
        logger.debug(f"action taken: {action} time taken: {elapsed:.5f}s")
        if render:
            env.render()

        if done:
            low_dim_info = env._get_low_dim_info()
            low_dim_state = tabletop_obs(low_dim_info)
            succ = evaluate_episode(low_dim_state, very_start, task_num)

            states_reshape = torch.from_numpy(np.stack(states))
            states_reshape = torch.reshape(states_reshape, (states_reshape.shape[0], -1)).unsqueeze(0).unsqueeze(0)
            dvd_reward = -terminal_cost(states_reshape, actions).item()
            engineered_reward = low_dim_state[3] # task 94 specific

            result = 'SUCCESS' if succ else 'FAILURE'
            total_successes += succ
            total_episodes += 1
            rolling_success.append(succ)
            print(f'----------Episode done: {result} | dvd_reward: {dvd_reward} | engineered_reward: {engineered_reward}----------')
            print(f'----------Currently at {total_successes} / {total_episodes}----------')
            
            if len(rolling_success) > 10:
                rolling_success.popleft()
            
            succ_rate = sum(rolling_success) / len(rolling_success)

            dvd_reward_history.append(dvd_reward)
            engineered_reward_history.append(engineered_reward)
            succ_rate_history.append(succ_rate)

            if total_episodes % 2 == 0:
                print('----------REPLOTTING----------')
                plt.figure()
                plt.plot([i for i in range(len(dvd_reward_history))], dvd_reward_history)
                plt.ylim([0, 1])
                plt.xlabel('Episode')
                plt.ylabel('DVD Reward')
                plt.savefig(os.path.join(logdir, 'dvd_rewards_episode.png'))
                plt.close()

                plt.figure()
                plt.plot([i for i in range(len(engineered_reward_history))], engineered_reward_history)
                plt.xlabel('Episode')
                plt.ylabel('Engineered Reward')
                plt.savefig(os.path.join(logdir, 'engineered_reward_episode.png'))
                plt.close()
                
                plt.figure()
                plt.plot([i for i in range(len(succ_rate_history))], succ_rate_history)
                plt.xlabel('Episode')
                plt.ylabel('Rolling Success Rate')
                plt.savefig(os.path.join(logdir, 'rolling_success_rate_episode.png'))
                plt.close()
            
            _, start_info = env.reset_model()
            very_start = tabletop_obs(start_info)
            states = []
            actions = []

        di = i % retrain_after_iter
        if di == 0 and i > 0:
            retrain_dynamics(dataset)
            # don't have to clear dataset since it'll be overridden, but useful for debugging
            dataset.zero_()
        dataset[di, :mppi.nx] = torch.tensor(state, dtype=mppi.U.dtype)
        dataset[di, mppi.nx:] = action

    return total_reward, total_successes, total_episodes, dataset