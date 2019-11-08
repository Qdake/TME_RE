from deap import tools
import matplotlib.pyplot as plt
import numpy as np
def plot(x,y,x_,y_,legend):
    fig, ax = plt.subplots()
    ax.plot(x,y)
    ax.legend([legend])
    ax.set_xlabel(x_)
    ax.set_ylabel(y_)
    plt.show()