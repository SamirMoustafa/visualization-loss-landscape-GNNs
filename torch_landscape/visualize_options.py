class VisualizationOptions:
    """
    A class for all options which can be set for the visualization of a loss surface and trajectories.
    """

    def __init__(
        self,
        min_x_value: int = -1,
        max_x_value: int = 1,
        min_y_value: int = -1,
        max_y_value: int = 1,
        num_points: int = 20,
        use_log_z: bool = False,
        show_title: bool = True,
        show_axes_labels: bool = True
    ):
        """
        Initializes the visualization options.
        :param min_x_value: The minimum value for the x coordinate (basis vector b1).
        :param max_x_value: The maximum value for the x coordinate (basis vector b1).
        :param min_y_value: The minimum value for the y coordinate (basis vector b2).
        :param max_y_value: The maximum value for the y coordinate (basis vector b2).
        :param num_points: The count of points for the x and y-axis.
        :param use_log_z: Set to true to plot the logarithm of the z values.
        :param show_title: Set to true to show the title (above the plot), set false to hide the title.
        :parameter show_axes_labels: Set to true to show labels of x/y/z axis.
        """
        self.min_x_value = min_x_value
        self.max_x_value = max_x_value
        self.min_y_value = min_y_value
        self.max_y_value = max_y_value
        self.num_points = num_points
        self.use_log_z = use_log_z
        self.show_title = show_title
        self.show_axis_labels = show_axes_labels
