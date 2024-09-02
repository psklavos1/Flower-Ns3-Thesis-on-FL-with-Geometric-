import matplotlib.pyplot as plt
import logging
import numpy as np
import pandas as pd


class DataPlotter:
    def plot_data(
        self,
        plot_type: str,
        x: str,
        y: list,
        data: pd.DataFrame,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        labels: list = None,
        cumulative: bool = False,
        comment: str = "",
    ):
        """Plots the data from the provided DataFrame."""
        if cumulative:
            for col in y:
                data[col] = data[col].cumsum()

        if plot_type == "line":
            self.plot_line(data, x, y, title, xlabel, ylabel, labels, comment)
        elif plot_type == "bar":
            self.plot_bar(data, x, y, title, xlabel, ylabel, labels, comment)
        elif plot_type == "scatter":
            self.plot_scatter(data, x, y, title, xlabel, ylabel, labels, comment)
        elif plot_type == "area":
            self.plot_area(data, x, y, title, xlabel, ylabel, labels, comment)
        elif plot_type == "pie":
            self.plot_pie(data, y[0], title, comment, labels)
        elif plot_type == "hist":
            self.plot_hist(data, x, y, title, xlabel, ylabel, comment)
        else:
            raise ValueError(
                "Plot type must be 'line', 'bar', 'scatter', 'area', 'pie', or 'hist'"
            )

        logging.info(f"Plotted {plot_type} plot for data: {y} vs {x}")

    def add_comment_to_legend(self, ax, comment):
        handles, labels = ax.get_legend_handles_labels()
        handles.append(plt.Line2D([0], [0], color="w", label=comment))
        labels.append(comment)
        ax.legend(handles=handles, labels=labels)

    def plot_line(
        self,
        data: pd.DataFrame,
        x: str,
        y: list,
        title: str,
        xlabel: str,
        ylabel: str,
        labels: list,
        comment: str = "",
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, y_col in enumerate(y):
            ax.plot(data[x], data[y_col], label=labels[i] if labels else y_col)
        ax.set_title(title if title else f"{', '.join(y)} vs {x}")
        ax.set_xlabel(xlabel if xlabel else x)
        ax.set_ylabel(ylabel if ylabel else ", ".join(y))
        ax.legend()
        ax.grid(True)
        self.add_comment_to_legend(ax, comment)
        plt.show()

    def plot_bar(
        self,
        data: pd.DataFrame,
        x: str,
        y: list,
        title: str,
        xlabel: str,
        ylabel: str,
        labels: list,
        comment: str = "",
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.35
        indices = np.arange(len(data[x]))

        for i, y_col in enumerate(y):
            ax.bar(
                indices + bar_width * i,
                data[y_col],
                width=bar_width,
                label=labels[i] if labels else y_col,
            )
        ax.set_title(title if title else f"{', '.join(y)} vs {x}")
        ax.set_xlabel(xlabel if xlabel else x)
        ax.set_ylabel(ylabel if ylabel else ", ".join(y))
        ax.set_xticks(indices + bar_width * (len(y) - 1) / 2)
        ax.set_xticklabels(data[x])
        ax.legend()
        ax.grid(True)
        self.add_comment_to_legend(ax, comment)
        plt.show()

    def plot_stacked_bar(
        self,
        data: pd.DataFrame,
        x: str,
        y: list,
        title: str,
        xlabel: str,
        ylabel: str,
        labels: list,
        comment: str = "",
        width: float = 0.7,  # Adjust the width of the bars
    ):
        """Plots a stacked bar chart."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Calculate the bottom for each segment of the stack
        bottoms = np.zeros(len(data))
        for i, y_col in enumerate(y):
            ax.bar(
                data[x],
                data[y_col],
                width=width,  # Adjust the width of the bars
                bottom=bottoms,
                label=labels[i] if labels else y_col,
            )
            bottoms += data[y_col]

        ax.set_title(title if title else f"{', '.join(y)} vs {x}")
        ax.set_xlabel(xlabel if xlabel else x)
        ax.set_ylabel(ylabel if ylabel else ", ".join(y))
        ax.legend()
        plt.grid(True)
        self.add_comment_to_legend(ax, comment)
        plt.show()

    def plot_scatter(
        self,
        data: pd.DataFrame,
        x: str,
        y: list,
        title: str,
        xlabel: str,
        ylabel: str,
        labels: list,
        comment: str = "",
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, y_col in enumerate(y):
            ax.scatter(data[x], data[y_col], label=labels[i] if labels else y_col)
        ax.set_title(title if title else f"{', '.join(y)} vs {x}")
        ax.set_xlabel(xlabel if xlabel else x)
        ax.set_ylabel(ylabel if ylabel else ", ".join(y))
        ax.legend()
        ax.grid(True)
        self.add_comment_to_legend(ax, comment)
        plt.show()

    def plot_area(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        title: str,
        xlabel: str,
        ylabel: str,
        labels: list = None,
        comment: str = "",
        group_by: str = None,
    ):
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        color_cycle = iter(colors)

        if group_by:
            for label, df in data.groupby(group_by):
                ax.fill_between(
                    df[x],
                    df[y],
                    label=f"{group_by}: {label}",
                    color=next(color_cycle),
                    alpha=0.5,
                )
        else:
            for i, y_col in enumerate(y):
                ax.fill_between(
                    data[x],
                    data[y_col],
                    label=labels[i] if labels else y_col,
                    color=next(color_cycle),
                    alpha=0.5,
                )

        ax.set_title(title if title else f"{', '.join(y)} vs {x}")
        ax.set_xlabel(xlabel if xlabel else x)
        ax.set_ylabel(ylabel if ylabel else ", ".join(y))

        # Add the comment to the legend
        handles, legend_labels = ax.get_legend_handles_labels()
        handles.append(plt.Line2D([0], [0], color="w", label=comment))
        legend_labels.append(comment)

        ax.legend(handles=handles, labels=legend_labels)
        ax.grid(True)
        plt.show()

    def plot_pie(
        self, data: pd.DataFrame, y: str, title: str, comment: str, labels: list = None
    ):
        fig, ax = plt.subplots(figsize=(10, 6))

        labels = data.index if labels is None else labels
        sizes = data[y]

        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.set_title(title if title else y)
        ax.axis("equal")
        self.add_comment_to_legend(ax, comment)
        plt.show()

    def plot_hist(
        self,
        data: pd.DataFrame,
        x: str,
        y: list,
        title: str,
        xlabel: str,
        ylabel: str,
        comment: str = "",
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        for col in y:
            ax.hist(data[x], weights=data[col], alpha=0.5, label=col)
        ax.set_title(title if title else f"Histogram of {', '.join(y)}")
        ax.set_xlabel(xlabel if xlabel else x)
        ax.set_ylabel(ylabel if ylabel else ", ".join(y))
        ax.legend()
        ax.grid(True)
        self.add_comment_to_legend(ax, comment)
        plt.show()

    def plot_combined_data(
        self,
        plot_type,
        x,
        y,
        data: pd.DataFrame,
        group_by,
        title,
        xlabel,
        ylabel,
        comment: str = "",
    ):
        fig, ax = plt.subplots(figsize=(10, 6))

        for label, df in data.groupby(group_by):
            if plot_type == "line":
                ax.plot(df[x], df[y], label=f"{group_by}: {label}")
            elif plot_type == "bar":
                ax.bar(df[x], df[y], label=f"{group_by}: {label}")
            elif plot_type == "scatter":
                ax.scatter(df[x], df[y], label=f"{group_by}: {label}")
            elif plot_type == "area":
                ax.scatter(df[x], df[y], label=f"{group_by}: {label}")
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        self.add_comment_to_legend(ax, comment)
        plt.grid(True)
        plt.show()
