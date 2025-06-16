import numpy as np
import matplotlib.pyplot as plt
from enum import IntEnum
from scipy.stats import sem


class StateSpace(IntEnum):
    FOOD = 0
    LEVER1 = 1
    LEVER2 = 2

class ActionSpace(IntEnum):
    STAY = 0
    LEFT = 1
    RIGHT = 2

class Agent:
    def __init__(self, alpha, beta, w_self, gamma=1):
        self.reset_params()

        # Hyper params
        self.alpha = alpha
        self.beta = beta
        self.w_self = w_self
        self.gamma = gamma
        self.w_other = 1 - w_self

    def reset_params(self):
        self.curr_state = None
        self.action = None
        self.Q_self = None
        self.Q_other = None
        self.Q_joint = None
        self.num_rewards = 0
        self.trajectory = []  # curr_state, action

    def select_action(self):
        def softmax(x):
            shift_x = self.beta * (x - np.max(x))  # max correction
            exps = np.exp(shift_x)
            return exps / np.sum(exps)
        probs = softmax(self.Q_joint[self.curr_state])
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
    def __init__(self, num_agents, alpha, beta, w_self):
        self.num_agents = num_agents
        self.reset_params(alpha, beta, w_self)

    def reset_params(self, alpha, beta, w_self):
        self.agents = [Agent(alpha, beta, w_self) for _ in range(self.num_agents)]
        self.joint_state = np.zeros((self.num_agents, len(StateSpace)))
        self.rew_mask = np.zeros((self.num_agents, len(StateSpace)))
        self.nb_joint_rewards = 0
        self.joint_rew_side = 0

        # Agents init
        for i, agent in enumerate(self.agents):
            agent.reset_params()
            agent.curr_state = np.random.choice(StateSpace)
            agent.Q_self = np.zeros((len(StateSpace), len(ActionSpace)))
            agent.Q_other = np.zeros((len(StateSpace), len(ActionSpace)))
            agent.Q_joint = np.zeros((len(StateSpace), len(ActionSpace)))
            self.joint_state[i, agent.curr_state] = i + 1

    def step(self):
        joint_rewards = np.zeros(self.num_agents)
        done = False
        observation = np.zeros((self.num_agents, len(StateSpace)))

        all_next_states = []
        # Update self state
        for i, agent in enumerate(self.agents):
            agent.action = agent.select_action()

            if agent.curr_state + agent.action < len(StateSpace):
                next_state =  agent.curr_state + agent.action
            else:
                next_state = agent.curr_state + agent.action - len(StateSpace)

            all_next_states.append(next_state) # Gather all next states for reward rule
            observation[i, next_state] = i + 1 # Fill joint_state with agent id in next_state

            reward = self.rew_mask[i, next_state]
            agent.num_rewards += reward
            joint_rewards[i] = reward
            self.rew_mask[i, next_state] = 0 # Retrieved reward

            # Q-learning Update
            max_Q_next = np.max(agent.Q_self[next_state])
            rpe = reward + agent.gamma * max_Q_next - agent.Q_self[agent.curr_state, agent.action]
            agent.Q_self[agent.curr_state, agent.action] += agent.alpha * rpe

        # Update Other Q-values and Q_joint
        for i, focal_agent in enumerate(self.agents):
            for j, other_agent in enumerate(self.agents):
                if i == j: # Not comparing self_values
                    continue
                # This is redundant only when self params are the same as other params. You can imagine self.alpha is different from other.alpa
                # If focal_agent.alpha_other == focal_agent.alpha_self, then:
                #focal_agent.Q_other += other_agent.Q_self / (self.num_agents - 1) # Expectation over all agents

                # Q-learning other
                other_max_Q_next = np.max(focal_agent.Q_other[all_next_states[j]])
                other_target_value =  joint_rewards[j] + focal_agent.gamma * other_max_Q_next
                other_rpe = other_target_value - focal_agent.Q_other[other_agent.curr_state, other_agent.action]
                focal_agent.Q_other[other_agent.curr_state, other_agent.action] += focal_agent.alpha * other_rpe

            focal_agent.Q_joint = focal_agent.w_self * focal_agent.Q_self + focal_agent.w_other * focal_agent.Q_other
            # Move to next_state
            focal_agent.curr_state = all_next_states[i]

        # Place a reward for all agents at FOOD state if there was a joint action (state)
        all_next_states = np.array(all_next_states) # Required for using np.all equivalence
        if np.all(all_next_states == StateSpace.LEVER1) or np.all(all_next_states == StateSpace.LEVER2):
            self.rew_mask[:, StateSpace.FOOD] += 1
            self.nb_joint_rewards += 1
            self.joint_rew_side += 1 if np.all(all_next_states == StateSpace.LEVER1) else -1

        self.joint_state = observation
        return observation, joint_rewards, done


if __name__ == "__main__":
    np.random.seed(34)
    steps = 2500
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
                side_preferences[i, episode, t] = agent.get_side_preference()
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