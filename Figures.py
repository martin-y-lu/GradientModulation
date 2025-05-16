import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def extract_batch_metric(logs, metric_name):
    """
    Extracts a specific batch-level metric from logs['batches'] and returns it as a PyTorch tensor.
    
    Args:
        logs (dict): The training logs from `train_and_evaluate`.
        metric_name (str): One of 'loss', 'acc', 'grad_norm', 'weight_update_norm'.
        
    Returns:
        torch.Tensor: 1D tensor of values across logged batches.
    """
    return torch.tensor([entry[metric_name] for entry in logs['batches']])

def plot_batch_metric(logs, metric_name, title=None, ylabel=None):
    """
    Plots the specified batch-level metric over time.
    
    Args:
        logs (dict): The logs returned from `train_and_evaluate`.
        metric_name (str): Metric key to extract from logs['batches'].
        title (str): Optional plot title.
        ylabel (str): Optional y-axis label.
    """
    values = extract_batch_metric(logs, metric_name)
    plt.figure(figsize=(10, 4))
    plt.plot(values.numpy())
    plt.title(title or f"{metric_name} across batches")
    plt.xlabel("Logged Batch Index")
    plt.ylabel(ylabel or metric_name)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def log_data_with_l(l, log_data, activation = 'gmodrelu'):
    keys = []
    for entry in log_data:
        if entry[0] == activation and entry[1] == l:
            keys.append(entry)
    keys.sort(key = lambda e: e[2])
    logs = [log_data[e] for e in keys]
    logs_flat = []
    keys_flat = []
    for i, logs in enumerate(logs):
        logs_flat += logs
        keys_flat += [keys[i]]*len(logs)
    return logs_flat, np.array([e[2] for e in keys_flat])

def log_data_with_k(k, log_data, activation = 'gmodrelu'):
    keys = []
    for entry in log_data:
        if entry[0] == activation and entry[2] == k:
            keys.append(entry)
    keys.sort(key = lambda e: e[1])
    logs = [log_data[e] for e in keys]
    logs_flat = []
    keys_flat = []
    for i, logs in enumerate(logs):
        logs_flat += logs
        keys_flat += [keys[i]]*len(logs)
    return logs_flat, np.array([e[1] for e in keys_flat])


def log_data_with_activation(act, log_data, key_map = lambda x:x):
    keys = []
    for entry in log_data:
        if entry == act:
            keys.append(entry)
    logs = [log_data[e] for e in keys]
    logs_flat = []
    keys_flat = []
    for i, logs in enumerate(logs):
        logs_flat += logs
        keys_flat += [key_map(keys[i])]*len(logs)
    return logs_flat, keys_flat

def log_data_with_activations(acts, colors, log_data):
    keys = []
    for entry in log_data:
        if entry in acts:
            keys.append(entry)
    logs = [log_data[e] for e in keys]
    logs_flat = []
    keys_flat = []
    for i, logs in enumerate(logs):
        logs_flat += logs
        keys_flat += [colors[acts.index(keys[i])]]*len(logs)
    return logs_flat, keys_flat

def plot_logs_k(logs, keys,prop = "loss",x_axis = "epoch", file_name = None,i_cut = 1000, y_scale = "linear"):
    cmap = mpl.cm.cool
    norm = mpl.colors.PowerNorm(0.11,vmin=keys.min(), vmax=keys.max())
    smap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    
    # gmodrelu_loss = extract_batch_metric(logs_gmodrelu, "loss")
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, log in enumerate(logs):
        if i < i_cut:
            if x_axis == "epoch":
                loss = np.array(log[prop])
            else:
                loss = extract_batch_metric(log, prop)
            # plt.plot(loss, color = mpl.cm.cool(i*0.2))
            ax.plot(loss, label = f"k = {keys[i]}",color = smap.to_rgba(keys[i]))
           
            # plt.plot(loss.numpy(), color = mpl.cm.cool(i*0.2))
        
    plt.title(f"CIFAR 10 Loss across {x_axis} (l = 0.02)")
    plt.xlabel(f"{x_axis}")
    plt.ylabel(prop)
    plt.yscale(y_scale)
    plt.grid(True)
    
    
    cbar = fig.colorbar(smap, ax=ax, orientation='vertical', label='k')
    plt.tight_layout()
    if file_name != None:
        plt.savefig(file_name)

def plot_logs_k_col(logs, colors,prop = "loss",x_axis = "epoch", file_name = None,i_cut = 1000,e_range = (0,1000), y_scale = "log"):
    
    # gmodrelu_loss = extract_batch_metric(logs_gmodrelu, "loss")
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, log in enumerate(logs):
        if i < i_cut:
            if x_axis == "epoch":
                loss = np.array(log[prop])
            else:
                loss = extract_batch_metric(log, prop)
            # plt.plot(loss, color = mpl.cm.cool(i*0.2))
            ax.plot(range(e_range[0],e_range[1]),loss[e_range[0]:e_range[1]],color = colors[i])
           
            # plt.plot(loss.numpy(), color = mpl.cm.cool(i*0.2))
        
    plt.title(f"CIFAR 10 Loss across {x_axis} (l = 0.02)")
    plt.xlabel(f"{x_axis}")
    plt.yscale(y_scale)
    plt.ylabel(prop)
    plt.grid(True)
    
    plt.tight_layout()
    if file_name != None:
        plt.savefig(file_name)

def extract_metric(log,prop,x_axis):
    if x_axis == "epoch":
        loss = np.array(log[prop])
    else:
        loss = extract_batch_metric(log, prop)
    return loss

def plot_logs_ke(logs, keys,prop = "loss",x_axis = "epoch",index = 9, title = "TITLE", x_label = "X_LABEL", x_scale = "log", file_name = None,i_cut = 1000,e_cut = 100000, baselines = [[],[]]):
    cmap = mpl.cm.cool
    norm = mpl.colors.PowerNorm(1,vmin=keys.min(), vmax=keys.max())
    smap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    
    # gmodrelu_loss = extract_batch_metric(logs_gmodrelu, "loss")
    fig, axs = plt.subplots(1,2,figsize=(8, 5),width_ratios = [1,5])
    # fig, axs = plt.subplots(1,2,figsize=(8, 5),width_ratios = [1,5])
    
    y_data = []
    x_data = []
    for i, log in enumerate(logs):
        if i <i_cut:
            loss = extract_metric(log,prop,x_axis)
            lv = loss[index]
            k = keys[i]
            y_data.append(lv)
            x_data.append(k)        
            # plt.plot(loss, color = smap.to_rgba())
           
            # plt.plot(loss.numpy(), color = mpl.cm.cool(i*0.2))
    axs[1].scatter(x_data,y_data)
    # test_vals = np.array([0.02])
    # axs[1].scatter(test_vals,np.zeros_like(test_vals))
    axs[1].set_xscale(x_scale)
    # axs[1].set_yscale('log')
    # axs[0].set_yscale('log')
    axs[1].grid(True)
        
    plt.title(title)
    axs[1].set_xlabel(x_label)
    axs[0].set_ylabel(prop)
    axs[0].grid(True)
    base_points = np.array([ extract_metric(log,prop,x_axis)[index] for log in baselines[1]])
    axs[0].scatter(baselines[0],base_points)
    axs[0].set_xmargin(0.7)
    # axs[1].set_ylim(bottom = 0.1)
    axs[0].set_ylim( axs[1].get_ylim() ) # align axes
    axs[1].set_yticklabels([]) # set ticks to be empty (no ticks, no tick-labels)
    
    plt.tight_layout()
    if file_name != None:
        plt.savefig(file_name)
    plt.show()