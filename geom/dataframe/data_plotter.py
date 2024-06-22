import matplotlib.pyplot as plt
import logging

from dataframe.data_handler import DataHandler


class DataPlotter:
    def __init__(self, data_handler: DataHandler):
        self.data_handler = data_handler

    def plot_data(
        self,
        plot_type: str,
        x: str,
        y: str,
        level: str,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
    ):
        """Plots the data from the specified DataFrame."""
        if level not in ["training", "epoch", "server", "info"]:
            raise ValueError("Level must be 'training', 'epoch', 'server', or 'info'")

        data = self.data_handler.get_data(level)

        if plot_type == "line":
            self.plot_line(data, x, y, title, xlabel, ylabel)
        elif plot_type == "bar":
            self.plot_bar(data, x, y, title, xlabel, ylabel)
        elif plot_type == "scatter":
            self.plot_scatter(data, x, y, title, xlabel, ylabel)
        else:
            raise ValueError("Plot type must be 'line', 'bar', or 'scatter'")

        logging.info(f"Plotted {plot_type} plot for {level} data: {y} vs {x}")

    def plot_line(self, data, x: str, y: str, title: str, xlabel: str, ylabel: str):
        plt.figure(figsize=(10, 6))
        plt.plot(data[x], data[y])
        plt.title(title if title else f"{y} vs {x}")
        plt.xlabel(xlabel if xlabel else x)
        plt.ylabel(ylabel if ylabel else y)
        plt.grid(True)
        plt.show()

    def plot_bar(self, data, x: str, y: str, title: str, xlabel: str, ylabel: str):
        plt.figure(figsize=(10, 6))
        plt.bar(data[x], data[y])
        plt.title(title if title else f"{y} vs {x}")
        plt.xlabel(xlabel if xlabel else x)
        plt.ylabel(ylabel if ylabel else y)
        plt.grid(True)
        plt.show()

    def plot_scatter(self, data, x: str, y: str, title: str, xlabel: str, ylabel: str):
        plt.figure(figsize=(10, 6))
        plt.scatter(data[x], data[y])
        plt.title(title if title else f"{y} vs {x}")
        plt.xlabel(xlabel if xlabel else x)
        plt.ylabel(ylabel if ylabel else y)
        plt.grid(True)
        plt.show()
