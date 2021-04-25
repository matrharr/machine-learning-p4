import matplotlib.pyplot as plt

def plot_discount(disc, res, title, name):
    plt.plot(disc, [r.iter for r in res], '-b', marker='o')
    plt.title(title)
    plt.xlabel('Discount Rate')
    plt.ylabel('Iterations')
    plt.savefig(f'figures/{name}.png')
    plt.clf()

def plot_ep(ep, res, title, name):
    plt.plot(ep, [r.iter for r in res], '-b', marker='o')
    plt.title(title)
    plt.xlabel('Epsilon')
    plt.ylabel('Iterations')
    plt.savefig(f'figures/{name}.png')
    plt.clf()

def plot_rewards(x_data, res, title, name, x_axis):
    rewards = []
    for r in res:
        indvdl_reward = 0
        for d in r.run_stats:
            indvdl_reward += d['Reward']

        rewards.append(indvdl_reward)

    plt.plot(x_data, rewards, '-b', marker='o')
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel('Reward')
    plt.savefig(f'figures/{name}.png')
    plt.clf()