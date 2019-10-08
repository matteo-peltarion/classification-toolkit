"""Utility functions."""

import os
from os.path import join
import shutil
import logging

import pandas as pd

import seaborn as sns

import matplotlib  # noqa
import matplotlib.pyplot as plt
import torch

import sklearn.metrics as metrics

LOG_FORMAT = '%(asctime)-15s %(levelname)-5s %(name)-15s - %(message)s'


def create_loss_plot(exp_dir, epochs, train_losses, test_losses):
    """Plot losses and save.

    Args:
        exp_dir (str): experiment directory.
        epochs (list): list of epochs (x-axis of loss plot).
        train_losses (list): list with train loss during each epoch.
        test_losses (list): list with test loss during each epoch.

    """
    f = plt.figure()  # noqa
    plt.title("Loss plot")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.plot(epochs, train_losses, 'b', marker='o', label='train loss')
    plt.plot(epochs, test_losses, 'r', marker='o', label='test loss')
    plt.legend()
    plt.savefig(join(exp_dir, 'loss.png'))
    plt.close(f)


def setup_logging(log_path=None, log_level='DEBUG',
                  logger=None, fmt=LOG_FORMAT):
    """Prepare logging for the provided logger.

    Args:
        log_path (str, optional): full path to the desired log file.
        debug (bool, optional): log in verbose mode or not.
        logger (logging.Logger, optional): logger to setup logging upon,
            if it's None, root logger will be used.
        fmt (str, optional): format for the logging message.

    """
    logger = logger if logger else logging.getLogger()
    logger.setLevel(log_level)
    logger.handlers = []

    fmt = logging.Formatter(fmt=fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.info('Log file is %s', log_path)


def save_checkpoint(state, target_dir, file_name='checkpoint.pth.tar',
                    backup_as_best=False,):
    """Save checkpoint to disk.

    Args:
        state: object to save.
        target_dir (str): Full path to the directory in which the checkpoint
            will be stored.
        backup_as_best (bool): Should we backup the checkpoint as the best
            version.
        file_name (str): the name of the checkpoint.

    """
    target_model_path = os.path.join(target_dir, file_name)

    os.makedirs(target_dir, exist_ok=True)
    torch.save(state, target_model_path)

    latest_model_path = os.path.join(target_dir, "model_latest.pth.tar")

    # Also copy as latest
    shutil.copyfile(target_model_path, latest_model_path)

    if backup_as_best:
        best_model_path = os.path.join(target_dir, 'model_best.pth.tar')
        shutil.copyfile(target_model_path, best_model_path)


def cm2df(cm, labels):
    """Print on terminal a confusion_matrix using class labels
    Taken from https://stackoverflow.com/questions/50325786/sci-kit-learn-how-to-print-labels-for-confusion-matrix # noqa
    """

    df = pd.DataFrame()
    # rows
    for i, row_label in enumerate(labels):
        rowdata = {}
        # columns
        for j, col_label in enumerate(labels):
            rowdata[col_label] = cm[i, j]

        df = df.append(pd.DataFrame.from_dict(
            {row_label: rowdata}, orient='index'))

    return df[labels]


def produce_per_class_stats(targets, predicted, labels):
    """
    Given labels, produce stats for each class individually

    Parameters
    ----------

    targets : numpy.ndarray
        A monodimensional array containing ground truth labels for samples.

    predicted : numpy.ndarray
        A monodimensional array containing predicted labels for samples.

    targets : list
        A list of strings containing the names of the classes.

    Returns
    -------

    dict
        Keys are class names, values are dictionaries which contain the various
        class stats.
    """

    stats_per_class = dict()

    for i, row_label in enumerate(labels):

        t_i = (targets == i)
        p_i = (predicted == i)

        accuracy_score = metrics.accuracy_score(t_i, p_i)
        precision_score = metrics.precision_score(t_i, p_i)
        recall_score = metrics.recall_score(t_i, p_i)
        f1_score = metrics.f1_score(t_i, p_i)
        roc_auc_score = metrics.roc_auc_score(t_i, p_i)

        class_stats = {
            'accuracy_score': accuracy_score,
            'precision_score': precision_score,
            'recall_score': recall_score,
            'f1_score': f1_score,
            'roc_auc_score': roc_auc_score,
        }

        stats_per_class[row_label] = class_stats

    return stats_per_class


def plot_tf_log(csv_files, title,
                set_names=['Training', 'Validation'],
                x_col='Step', y_col='Value', x_label="Epoch", y_label=None,
                palette='deep'):
    """Produce plot from csv dumped from tensorboard.

    Parameters
    ----------
        csv_files : list
            a list of paths to csv files containing the values that need to
            be plotted.

        title : str
            The title that will appear above the plot.

        set_names : list, optional
            The list of names for the sets, which will also be displayed in
            the plot's legend. Default ["Training", "Validation"].

        x_col : str, optional
            The name of the column containing values for the x axis.
            Default 'Step'.

        y_col : str, optional
            The name of the column containing values for the y axis.
            Default 'Value'.

        x_label : str, optional
            The label of the for the x axis. Default 'Epoch'.

        y_label : str, optional
            The label of the for the y axis. If `None`, defaults to y_col.
            Default `None`.

        palette :  palette name (str), list, or dict, optional
            The color palette used for plotting.
            See https://seaborn.pydata.org/generated/seaborn.lineplot.html for
            more information. Default 'deep'.
    """

    df_list = list()

    # Sanity check
    assert(len(csv_files) == len(set_names))

    for i, csv_f in enumerate(csv_files):

        df = pd.read_csv(csv_f)
        df['Set'] = set_names[i]

        df_list.append(df)

    # Produce a dataframe with data for all lines
    run_df = pd.concat(df_list)

    sns.set_style("whitegrid")

    # Plot the lines
    sns.lineplot(x=x_col, y=y_col, data=run_df, hue='Set',
                 palette=palette).set_title(title)

    if y_label is None:
        y_label = y_col

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.show()
