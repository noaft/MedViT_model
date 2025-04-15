import matplotlib.pyplot as plt

def BarChart(x, y, Title="Bar chart_data", X_label='x', Y_label='y', color='skyblue'):
    """
    Draw a bar chart using the given x and y data.

    Parameters:
    - x (list): Categories or labels on the x-axis
    - y (list): Corresponding values for each category
    - Title (str): Title of the bar chart
    - X_label (str): Label for the x-axis
    - Y_label (str): Label for the y-axis
    - color (str): Color of the bars (default: 'skyblue')
    """
    # Create the bar chart
    plt.bar(x, y, color=color)

    # Set the chart title and axis labels
    plt.title(Title)
    plt.xlabel(X_label)
    plt.ylabel(Y_label)

    # Display the chart
    plt.show()
