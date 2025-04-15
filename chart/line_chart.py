import matplotlib.pyplot as plt
import numpy as np

def LineChart(y, Title="Line chart_data", X_label='x', Y_label='y', color='skyblue'):
    """
    Creates and displays a line chart using matplotlib.
    
    Parameters:
    -----------
    y : array-like
        The y-axis data points to be plotted
    Title : str, optional
        The title of the chart (default: "Line chart_data")
    X_label : str, optional
        The label for the x-axis (default: 'x')
    Y_label : str, optional
        The label for the y-axis (default: 'y')
    color : str, optional
        The color of the line (default: 'skyblue')
        
    Returns:
    --------
    None
        Displays the line chart using matplotlib's show() function
    """
    plt.plot(x, y, color=color)
    plt.title(Title)
    plt.xlabel(X_label)
    plt.ylabel(Y_label)
    plt.show()
