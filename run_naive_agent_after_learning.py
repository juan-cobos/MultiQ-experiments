from joint_MDP import *

def plot(x, y):
    fig, ax = plt.subplots(1, 2, figsize=(24, 12))
    for i in range(len(env.agents)):
        ax[0].plot(x[i], label=f"Agent{i + 1}")
        ax[1].plot(y[i], label=f"Agent{i + 1}")
    ax[0].legend()
    ax[1].legend()
    plt.show()

if __name__ == "__main__":

    np.random.seed(34)
    steps = 5000
    num_agents = 2
    env = JointMDP(num_agents=num_agents)

    # First task
    reward_rates = np.zeros((num_agents, steps))
    side_preferences = np.zeros((num_agents, steps))
    for t in range(steps):
        obs, rewards, done = env.step()
        for idx in range(len(env.agents)):
            reward_rates[idx, t] = env.agents[idx].num_rewards / (t+1)
            side_preferences[idx, t] = env.agents[idx].get_side_preference()

    print("Q-values after learning joint task\n")
    for agent in env.agents:
        print(agent.Q, "\n")

    plot(reward_rates, side_preferences)

    # Keep Q-values from one agent and reset env (similar to reintroduce a naive agent)
    Q_buffer = env.agents[0].Q.copy()
    env.reset_params()
    env.agents[0].Q = Q_buffer.copy()

    print("Q-values after reintroducing new agent\n")
    for agent in env.agents:
        print(agent.Q, "\n")

    # Second task
    reward_rates = np.zeros((num_agents, steps))
    side_preferences = np.zeros((num_agents, steps))
    for t in range(steps):
        obs, rewards, done = env.step()
        for idx in range(len(env.agents)):
            reward_rates[idx, t] = env.agents[idx].num_rewards / (t+1)
            side_preferences[idx, t] = env.agents[idx].get_side_preference()

    print("Q-values after relearning\n")
    for agent in env.agents:
        print(agent.Q, "\n")

    plot(reward_rates, side_preferences)
