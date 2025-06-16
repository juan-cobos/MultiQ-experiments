import numpy as np
import matplotlib.pyplot as plt
from enum import IntEnum

ALPHA_RANGE = np.arange(1, 101, 10) / 100  # 0.01 - 1.1
BETA_RANGE = np.arange(1, 101, 10) / 10  # 0.1 - 10.1
STEPS = 5000
RUNS = 30

class StateSpace(IntEnum):
    FOOD = 0
    LEVER1 = 1
    LEVER2 = 2

class ActionSpace(IntEnum):
    STAY = 0
    LEFT = 1
    RIGHT = 2

def select_action(x, beta):
    def softmax(x):
        shift_x = beta * (x - np.max(x))  # max correction
        exps = np.exp(shift_x)
        return exps / np.sum(exps)
    probs =  softmax(x)
    selected_action = np.random.choice(ActionSpace, p=probs)
    return selected_action

def get_side_preference(state_hist):
    side_preference = (state_hist == StateSpace.LEVER1).sum() - (state_hist == StateSpace.LEVER2).sum()
    side_preference /= (state_hist == StateSpace.LEVER1).sum() + (state_hist == StateSpace.LEVER2).sum()
    return side_preference

def run_simulation(steps, alpha, beta, gamma):
    # Model init
    Q1 = np.zeros((len(StateSpace), len(ActionSpace)))
    s1 = np.random.choice(StateSpace)

    Q2 = np.zeros((len(StateSpace), len(ActionSpace)))
    s2 = np.random.choice(StateSpace)
    
    r1_mask = np.zeros(len(StateSpace))
    r2_mask = np.zeros(len(StateSpace))

    joint_side_preference = 0
    joint_rewards = 0

    placed_rewards = np.zeros(steps)
    s1_hist = np.zeros(steps)
    s2_hist = np.zeros(steps)
    total_rew_hist1 = np.zeros(steps)
    total_rew_hist2 = np.zeros(steps)
    side_pref_hist1 = np.zeros(steps)
    side_pref_hist2 = np.zeros(steps)
    total_rewards1 = 0
    total_rewards2 = 0
    for t in range(steps):
        a1 = select_action(Q1[s1], beta=beta)
        a2 = select_action(Q2[s2], beta=beta)
        s1_hist[t] = s1
        s2_hist[t] = s2
        side_pref_hist1[t] = get_side_preference(s1_hist)
        side_pref_hist2[t] = get_side_preference(s2_hist)

        s1_prime = s1 + a1 if s1 + a1 < len(StateSpace) else s1 + a1 - len(StateSpace)
        s2_prime = s2 + a2 if s2 + a2 < len(StateSpace) else s2 + a2 - len(StateSpace)
        
        # Obtain rewards
        r1 = r1_mask[s1_prime]
        r2 = r2_mask[s2_prime]

        total_rewards1 += r1
        total_rewards2 += r2
        total_rew_hist1[t] += r1
        total_rew_hist2[t] += r2
        r1_mask[s1_prime] = 0 # Reward retrieval
        r2_mask[s2_prime] = 0 # Reward retrieval

        # Place rewards
        all_states = np.array([s1_prime, s2_prime])
        if np.all(all_states == StateSpace.LEVER1) or np.all(all_states == StateSpace.LEVER2):
            placed_rewards[t] += 1
            joint_rewards += 1
            joint_side_preference += 1 if np.all(all_states == StateSpace.LEVER1) else -1
            r1_mask[StateSpace.FOOD] += 1
            r2_mask[StateSpace.FOOD] += 1

        max_Q1_next = np.max(Q1[s1_prime])
        max_Q2_next = np.max(Q2[s2_prime])

        # Q-learning Update
        Q1[s1, a1] += alpha * (r1 + gamma * max_Q1_next - Q1[s1, a1])
        Q2[s2, a2] += alpha * (r2 + gamma * max_Q2_next - Q2[s2, a2])

        # Update current state (s_t)
        s1 = s1_prime
        s2 = s2_prime

    #print("Q_1(s, a)\n", Q1)
    #print("Q_2(s, a)\n", Q2)

    perf1 = np.cumsum(total_rew_hist1)/np.arange(steps)
    perf2 = np.cumsum(total_rew_hist2)/np.arange(steps)
    performance = (perf1, perf2)
    joint_performance = placed_rewards.sum() / steps # Either cumsum / arange(steps) or sum/steps
    print("Joint performance", joint_performance)
    joint_side_preference /= joint_rewards
    print("Joint side preference", joint_side_preference)
    #side_preference = (side_pref1, side_pref2)
    side_preference = side_pref_hist1, side_pref_hist2
    return joint_performance, np.abs(joint_side_preference)

def create_hmaps(gamma):
    performance_hmap = np.zeros((len(ALPHA_RANGE), len(BETA_RANGE)))
    side_pref_hmap = np.zeros((len(ALPHA_RANGE), len(BETA_RANGE)))
    for _ in range(RUNS):
        for i, alpha in enumerate(ALPHA_RANGE):
            for j, beta in enumerate(BETA_RANGE):
                perf, side_pref = run_simulation(STEPS, alpha, beta, gamma)
                performance_hmap[i, j] += perf
                side_pref_hmap[i, j] += side_pref
    return performance_hmap/RUNS, side_pref_hmap/RUNS

if __name__ == "__main__":

    np.random.seed(34)
    gamma_range = [0.9, 0.95, 0.99]
    fig, ax = plt.subplots(len(gamma_range), 2)
    for i, gamma in enumerate(gamma_range):
        perf_hmap, side_hmap = create_hmaps(gamma)

        # Get max performance position
        max_perf_pos = np.unravel_index(np.argmax(perf_hmap), perf_hmap.shape)

        # Plot Side Preference heatmap
        im1 = ax[i][0].imshow(perf_hmap, cmap="inferno", vmin=0, vmax=1, interpolation="gaussian")
        ax[i][0].set_yticks(np.arange(len(ALPHA_RANGE)), ALPHA_RANGE)
        ax[i][0].set_xticks(np.arange(len(BETA_RANGE)), BETA_RANGE)
        ax[i][0].set_xlabel("Beta ($\\beta$)", fontsize=14)
        ax[i][0].set_ylabel("Alpha ($\\alpha$)", fontsize=14)
        ax[i][0].text(-1, 0.5, f"Gamma ($\\gamma$) = {gamma}", transform=ax[i][0].transAxes,
                      va='center', ha='center', fontsize=12)

        # Plot Performance heatmap
        im2 = ax[i][1].imshow(side_hmap, cmap="inferno", vmin=0, vmax=1, interpolation="gaussian")
        ax[i][1].scatter(max_perf_pos[1], max_perf_pos[0], color="lime", marker="*") # Shifted X and Y to match imshow representation
        ax[i][1].set_yticks(np.arange(len(ALPHA_RANGE)), ALPHA_RANGE)
        ax[i][1].set_xticks(np.arange(len(BETA_RANGE)), BETA_RANGE)
        ax[i][1].set_xlabel("Beta ($\\beta$)", fontsize=14)
        ax[i][1].set_ylabel("Alpha ($\\alpha$)", fontsize=14)

    ax[0][0].set_title("Joint Performance", fontsize=18)
    ax[0][1].set_title("Joint Side Preference", fontsize=18)

    print("Max perf", max_perf_pos)
    print("Pef hmap", perf_hmap)
    print("Side hmap", side_hmap)
    cbar = fig.colorbar(im2, ax=ax, orientation='vertical')
    cbar.set_label("Intensity", fontsize=18)
    plt.show()

# 30 dyads at least