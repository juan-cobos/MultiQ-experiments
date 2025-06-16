import numpy as np
import matplotlib.pyplot as plt
from enum import IntEnum
from scipy.stats import sem

np.random.seed(45)

ALPHA_RANGE = np.arange(1, 101, 10) / 100  # 0.01 - 1.1
BETA_RANGE = np.arange(1, 101, 10) / 10  # 0.1 - 10.1
STEPS = 5000
LAM = 5
TW = 4

class StateSpace(IntEnum):
    FOOD = 0
    LEVER1 = 1
    LEVER2 = 2

class ActionSpace(IntEnum):
    STAY = 0
    LEFT = 1
    RIGHT = 2

class Agent:
    def __init__(self, alpha=0.1, beta=5, gamma=0.95):
        self.reset_params()

        # Hyper params
        self.alpha = alpha # alpha := learning rate
        self.beta = beta
        self.gamma = gamma

    def reset_params(self):
        self.curr_state = None
        self.last_action = None
        self.Q = None  # State-Action Value
        self.V = None  # Expected Reward
        self.num_rewards = 0
        self.dwell_time_elapsed = 0
        self.dwell_time_target = np.random.poisson(LAM)
        self.trajectory = []  # curr_state, action

    def select_action(self):
        def softmax(x):
            shift_x = self.beta * (x - np.max(x))  # max correction
            exps = np.exp(shift_x)
            return exps / np.sum(exps)
        probs = softmax(self.Q[self.curr_state])
        selected_action = np.random.choice(ActionSpace, p=probs)
        self.trajectory.append((self.curr_state, selected_action))
        return selected_action

    def get_side_preference(self):
        state_hist = np.array(self.trajectory)[:, 0] # shape [T, 2] - [T, (state, action)]
        side_preference = (state_hist == StateSpace.LEVER1).sum() - (state_hist == StateSpace.LEVER2).sum()
        side_preference /= (state_hist == StateSpace.LEVER1).sum() + (state_hist == StateSpace.LEVER2).sum()
        return np.abs(side_preference)

class JointSMDP:
    """ Joint Semi-Markov Decision Process """
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.reset_params()

    def reset_params(self):
        self.agents = [Agent() for _ in range(self.num_agents)]
        self.joint_state = np.zeros((self.num_agents, len(StateSpace)))
        self.rew_mask = np.zeros((self.num_agents, len(StateSpace)))
        self.joint_rewards = 0
        self.rewarded_side = 0

        self.time_window = TW
        self.delta_time_window = 0
        self.init_side = None
        self.open_window = False
        self.same_state_mask = np.zeros(self.num_agents)
        # Agents init
        for i, agent in enumerate(self.agents):
            agent.reset_params()
            agent.curr_state = np.random.choice(StateSpace)
            agent.Q = np.zeros((len(StateSpace), len(ActionSpace)))
            self.joint_state[i, agent.curr_state] = i + 1

    def step(self):
        rewards = np.zeros(self.num_agents)
        done = False
        observation = np.zeros((self.num_agents, len(StateSpace)))
        all_next_states = []
        
        for i, agent in enumerate(self.agents):
            if agent.dwell_time_elapsed < agent.dwell_time_target: # Agent still in dwell time
                agent.dwell_time_elapsed += 1
                next_state = agent.curr_state
                all_next_states.append(next_state)
                observation[i, next_state] = i + 1
                agent.trajectory.append((agent.curr_state, agent.last_action))
                continue

            action = agent.select_action()
            agent.last_action = action
            if agent.curr_state + action < len(StateSpace):
                next_state =  agent.curr_state + action
            else:
                next_state = agent.curr_state + action - len(StateSpace)

            all_next_states.append(next_state) # Gather all next states for reward rule
            observation[i, next_state] = i + 1 # Fill joint_state with agent id in next_state

            # Open window in init state side
            if (next_state == StateSpace.LEVER1 or next_state == StateSpace.LEVER2) and not self.open_window:
                self.open_window = True
                self.init_side = next_state
                self.same_state_mask[i] = 1

            reward = self.rew_mask[i, next_state]
            agent.num_rewards += reward
            rewards[i] = reward
            self.rew_mask[i, next_state] = 0 # Retrieved reward

            # TODO: model used on dopaminergic activity
            #rpe = reward - p * dwell_time + agent.V[next_state] - agent.V[agent.curr_state]
            #rpe = ...
            #agent.V[agent.curr_state] += agent.alpha * rpe

            # Q-learning Update
            max_Q_next = np.max(agent.Q[next_state])
            # TODO: discuss if gamma should be raised to the power of time_elapsed
            scaled_target = agent.alpha * (reward + agent.gamma ** agent.dwell_time_elapsed * max_Q_next)
            agent.Q[agent.curr_state,action] = (1 - agent.alpha) * agent.Q[agent.curr_state,action] + scaled_target

            # Reset dwell times
            agent.dwell_time_elapsed = 0
            agent.dwell_time_target = np.random.poisson(LAM)

            # Move to next_state
            agent.curr_state = next_state

        # If time window is open
        if self.open_window:
            for j, s_next in enumerate(all_next_states):
                if s_next == self.init_side: # mask agent next states same as init_side
                    self.same_state_mask[j] = 1

            if np.all(self.same_state_mask == 1):
                self.rew_mask[:, StateSpace.FOOD] += 1 # Place rewards for all

            self.delta_time_window += 1

            # Reset time_window params
            if self.delta_time_window >= self.time_window:
                self.open_window = False
                self.delta_time_window = 0
                self.same_state_mask[:] = 0

        self.joint_state = observation
        return observation, rewards, done

if __name__ == "__main__":
    np.random.seed(34)
    steps = 1000
    num_episodes = 30
    num_agents = 2
    env = JointSMDP(num_agents=num_agents)
    reward_rates = np.zeros((num_agents, num_episodes, steps))
    side_preferences = np.zeros((num_agents, num_episodes, steps))

    for episode in range(num_episodes):
        env.reset_params()
        for t in range(steps):
            obs, rewards, done = env.step()
            for idx in range(len(env.agents)):
                reward_rates[idx, episode, t] = env.agents[idx].num_rewards / (t + 1)
                side_preferences[idx, episode, t] = env.agents[idx].get_side_preference()

    # Average over episodes
    reward_rate_mean = np.mean(reward_rates, axis=1)
    reward_rate_sem = sem(reward_rates, axis=1)
    side_preference_mean = np.mean(side_preferences, axis=1)
    side_preference_sem = sem(side_preferences, axis=1)

    # Plot single agent data (both agents are similar in joint task)
    x = np.arange(steps)
    plt.plot(reward_rate_mean[0, :], color="blue", label=f"Reward rate")
    plt.fill_between(
        x,
        reward_rate_mean[0, :] - reward_rate_sem[0, :],
        reward_rate_mean[0, :] + reward_rate_sem[0, :],
        alpha=0.5, color="blue"
    )

    plt.plot(side_preference_mean[0, :], color="orange", label=f"Side preference")
    plt.fill_between(
        x,
        side_preference_mean[0, :] - side_preference_sem[0, :],
        side_preference_mean[0, :] + side_preference_sem[0, :],
        alpha=0.5, color="orange"
    )

    plt.xlabel("Steps", fontsize=14)
    plt.ylabel("Percentage (%)", fontsize=14)
    plt.legend()
    plt.show()
