import matplotlib.pyplot as plt

def plot_chart(x, y, title="Chart", x_label="x", y_label="y", color="skyblue"):
    """
    Creates and displays a line chart using matplotlib.
    
    Parameters:
    -----------
    x : array-like
        The x-axis data points to be plotted
    y : array-like
        The y-axis data points to be plotted
    Title : str, optional
    """

    plt.plot(x,y, color=color)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()