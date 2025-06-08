import numpy as np
import matplotlib.pyplot as plt
from enum import IntEnum
from scipy.stats import sem

ALPHA_RANGE = np.arange(1, 101, 10) / 100  # 0.01 - 1.1
BETA_RANGE = np.arange(1, 101, 10) / 10  # 0.1 - 10.1
STEPS = 5000

class StateSpace(IntEnum):
    FOOD = 0
    LEVER1 = 1
    LEVER2 = 2

class ActionSpace(IntEnum):
    STAY = 0
    LEFT = 1
    RIGHT = 2

class Agent:
    def __init__(self, alpha=0.1, beta=3, gamma=0.99):
        self.reset_params()

        # Hyper params
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def reset_params(self):
        self.curr_state = None
        self.Q = None
        self.num_rewards = 0
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

class JointMDP:
    """ Joint Markov Decision Process """
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.reset_params()

    def reset_params(self):
        self.agents = [Agent() for _ in range(self.num_agents)]
        self.joint_state = np.zeros((self.num_agents, len(StateSpace)))
        self.rew_mask = np.zeros((self.num_agents, len(StateSpace)))
        self.joint_rewards = 0
        self.rewarded_side = 0

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
            action = agent.select_action()

            if agent.curr_state + action < len(StateSpace):
                next_state =  agent.curr_state + action
            else:
                next_state = agent.curr_state + action - len(StateSpace)

            all_next_states.append(next_state) # Gather all next states for reward rule
            observation[i, next_state] = i + 1 # Fill joint_state with agent id in next_state

            reward = self.rew_mask[i, next_state]
            agent.num_rewards += reward
            rewards[i] = reward
            self.rew_mask[i, next_state] = 0 # Retrieved reward

            # Q-learning Update (we could also try with SARSA instead)
            max_Q_next = np.max(agent.Q[next_state])
            scaled_target = agent.alpha * (reward + agent.gamma * max_Q_next)
            agent.Q[agent.curr_state, action] = (1 - agent.alpha) * agent.Q[agent.curr_state, action] + scaled_target

            # Move to next_state
            agent.curr_state = next_state

        # Place a reward for all agents at FOOD state if there was a joint action (state)
        all_next_states = np.array(all_next_states) # Required for using np.all equivalence
        if np.all(all_next_states == StateSpace.LEVER1) or np.all(all_next_states == StateSpace.LEVER2):
            self.rew_mask[:, StateSpace.FOOD] += 1
            self.joint_rewards += 1
            self.rewarded_side += 1 if np.all(all_next_states == StateSpace.LEVER1) else -1

        self.joint_state = observation
        return observation, rewards, done

if __name__ == "__main__":
    np.random.seed(34)
    steps = 5000
    num_agents = 2
    env = JointMDP(num_agents=num_agents)
    num_episodes = 30
    side_preferences = np.zeros((num_agents, num_episodes, steps))
    reward_rates = np.zeros((num_agents, num_episodes, steps))
    for episode in range(num_episodes):
        env.reset_params()
        for t in range(steps):
            obs, rewards, done = env.step()
            for i, agent in enumerate(env.agents):
                #print(f"Agent {i+1} obtained {agent.num_rewards}")
                #print(f"Agent {i+1} Q values {agent.Q}\n")
                side_pref = agent.get_side_preference()
                side_preferences[i, episode, t] = side_pref
                reward_rates[i, episode, t] = agent.num_rewards / (t+1)
                #print(side_pref)

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
        side_preference_mean[0, :] - side_preference_sem[0,:],
        side_preference_mean[0, :] + side_preference_sem[0,:],
        alpha=0.5, color="orange"
    )

    plt.xlabel("Steps", fontsize=14)
    plt.ylabel("Percentage (%)", fontsize=14)
    plt.legend()
    plt.show()