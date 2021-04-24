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
