from scipy.ndimage import rotate

from joint_Q import *

ALPHA_RANGE = np.arange(1, 101, 10) / 100  # 0.01 - 1.1
BETA_RANGE = np.arange(1, 101, 10) / 10  # 0.1 - 10.1
W_SELF_RANGE = [1.0, 0.75, 0.5, 0.25, 0.0]
NUM_AGENTS = 2
EPISODES = 30
STEPS = 100

def run_simulation(alpha, beta, w_self):
    env = JointMDP(num_agents=NUM_AGENTS, alpha=alpha, beta=beta, w_self=w_self)
    for t in range(STEPS):
        _, _, _ = env.step()
    reward_rate = env.nb_joint_rewards / STEPS
    side_preference = env.joint_rew_side / STEPS # [-1, 1]
    return reward_rate, np.abs(side_preference)

def create_hmaps(w_self):
    rew_rate_hmap = np.zeros((len(ALPHA_RANGE), len(BETA_RANGE)))
    side_pref_hmap = np.zeros((len(ALPHA_RANGE), len(BETA_RANGE)))
    for _ in range(EPISODES):
        for i, alpha in enumerate(ALPHA_RANGE):
            for j, beta in enumerate(BETA_RANGE):
                rew_rate, side_pref = run_simulation(alpha, beta, w_self)
                rew_rate_hmap[i, j] += rew_rate
                side_pref_hmap[i, j] += side_pref
    return rew_rate_hmap/EPISODES, side_pref_hmap/EPISODES

if __name__ == "__main__":

    np.random.seed(36)
    fig, ax = plt.subplots(len(W_SELF_RANGE), 2)
    for i, weight_self in enumerate(W_SELF_RANGE):
        rew_hmap, side_hmap = create_hmaps(weight_self)

        # Get max performance position
        max_perf_pos = np.unravel_index(np.argmax(rew_hmap), rew_hmap.shape)

        # Plot Reward Rate heatmap
        # TODO: modify vmin and vmax to get sharper view on the differences
        im1 = ax[i][0].imshow(rew_hmap, cmap="inferno", vmin=0, vmax=1, interpolation="gaussian")
        ax[i][0].set_yticks(np.arange(len(ALPHA_RANGE)), ALPHA_RANGE)
        ax[i][0].set_xticks(np.arange(len(BETA_RANGE)), BETA_RANGE, rotation=45)
        ax[i][0].set_xlabel("Beta ($\\beta$)", fontsize=14)
        ax[i][0].set_ylabel("Alpha ($\\alpha$)", fontsize=14)
        ax[i][0].text(-1, 0.5, f"W_self = {weight_self}", transform=ax[i][0].transAxes,
                      va='center', ha='center', fontsize=12)

        # Plot Side Preference heatmap
        im2 = ax[i][1].imshow(side_hmap, cmap="inferno", vmin=0, vmax=1, interpolation="gaussian")
        ax[i][1].scatter(max_perf_pos[1], max_perf_pos[0], color="lime",
                         marker="*")  # Shifted X and Y to match imshow representation
        ax[i][1].set_yticks(np.arange(len(ALPHA_RANGE)), ALPHA_RANGE)
        ax[i][1].set_xticks(np.arange(len(BETA_RANGE)), BETA_RANGE, rotation=45)
        ax[i][1].set_xlabel("Beta ($\\beta$)", fontsize=14)
        ax[i][1].set_ylabel("Alpha ($\\alpha$)", fontsize=14)

    ax[0][0].set_title("Joint Reward Rate", fontsize=18)
    ax[0][1].set_title("Joint Side Preference", fontsize=18)

    print("Max rew rate", max_perf_pos)
    print("Rew Rate hmap", rew_hmap)
    print("Side Pref hmap", side_hmap)
    cbar = fig.colorbar(im2, ax=ax, orientation='vertical')
    cbar.set_label("Intensity", fontsize=18)
    plt.show()
