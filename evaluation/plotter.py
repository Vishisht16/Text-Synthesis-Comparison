import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_training_histories(results_dict: dict, metric: str = 'loss'):
    """
    Plots the training and validation history for a given metric for multiple models.

    Args:
        results_dict (dict): The main dictionary containing training results.
                             Expected format: {'model_name': {'history': KerasHistory, ...}}
        metric (str): The metric to plot (e.g., 'loss', 'accuracy', 'perplexity').
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7))

    for model_name, data in results_dict.items():
        history = data['history']
        
        # Plot training metric
        plt.plot(history.history[metric], label=f'{model_name.upper()} Train {metric.capitalize()}')
        
        # Plot validation metric
        val_metric = f'val_{metric}'
        if val_metric in history.history:
            plt.plot(history.history[val_metric], '--', label=f'{model_name.upper()} Val {metric.capitalize()}')

    plt.title(f'Model Training & Validation {metric.capitalize()} Comparison', fontsize=16)
    plt.ylabel(metric.capitalize(), fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend()
    plt.show()


def plot_comparison_chart(results_dict: dict, metric_name: str, title: str):
    """
    Creates a bar chart to compare a single metric (e.g., training time) across models.

    Args:
        results_dict (dict): The main dictionary containing training results.
        metric_name (str): The key for the metric to plot (e.g., 'training_time').
        title (str): The title for the chart.
    """
    model_names = list(results_dict.keys())
    metric_values = [results[metric_name] for results in results_dict.values()]

    plt.figure(figsize=(10, 6))
    splot = sns.barplot(x=model_names, y=metric_values)
    
    # Add values on top of bars
    for p in splot.patches:
        splot.annotate(format(p.get_height(), '.2f'), 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 9), 
                       textcoords = 'offset points')

    plt.title(title, fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel(metric_name.replace('_', ' ').capitalize(), fontsize=12)
    plt.xticks(rotation=0)
    plt.show()