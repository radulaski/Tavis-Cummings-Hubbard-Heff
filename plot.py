import numpy as np
import qutip
from typing import List, Tuple, Dict, Union
import copy
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from matplotlib import cm

import single_cavity
import multi_cavity
from util import *
import metrics

#plotting functions:
#  participation
#  node_participation
#  eigenvectors
#  eigenvalues
#  occupancy

def participation(cavity_array, ax=None, normalize = False, **kwargs):
    """
    Args:
        cavity_array: multi_cavity.CavityArray object
        ax: optional axes to plot on
        kwargs for Figure, Figure.subplots, axes.scatter
    Returns:
        scatter plot of participation ratio for eigenvectors 
        of cavity_array on ax if given or creates new Figure and axis
    """
    p = metrics.participation(cavity_array, normalize)
    if ax is not None:
        ax.scatter(range(len(p)), p, **kwargs)
        return ax
    kw = kwargs_sep(Figure, kwargs)
    fig = Figure(**kw)
    kw = kwargs_sep(fig.subplots, kwargs)
    ax = fig.subplots(**kw)
    ax.set_xlabel('Eigenvector')
    ax.set_ylabel('Participation Ratio')
    kw = kwargs_sep(ax.scatter, kwargs)
    ax.scatter(range(len(p)), p, **kwargs)
    return fig


def node_participation(cavity_array, ax=None, normalize=False, **kwargs):
    p = metrics.node_participation(cavity_array, normalize)
    if ax is not None:
        ax.scatter(range(len(p)), p, **kwargs)
        return ax
    kw = kwargs_sep(Figure, kwargs)
    fig = Figure(**kw)
    kw = kwargs_sep(fig.subplots, kwargs)
    ax = fig.subplots(**kw)
    ax.set_ylabel('Node Participation Ratio')
    ax.set_xlabel('Eigenvector')
    kw = kwargs_sep(ax.scatter, kwargs)
    ax.scatter(range(len(p)), p, **kwargs)
    return fig

def polariton_participation(cavity_array, ax=None, normalize=False, **kwargs):
    p = metrics.polariton_participation(cavity_array, normalize)
    if ax is not None:
        ax.scatter(range(len(p)), p, **kwargs)
        return ax
    kw = kwargs_sep(Figure, kwargs)
    fig = Figure(**kw)
    kw = kwargs_sep(fig.subplots, kwargs)
    ax = fig.subplots(**kw)
    ax.set_ylabel('Polariton Participation Ratio')
    ax.set_xlabel('Eigenvector')
    kw = kwargs_sep(ax.scatter, kwargs)
    ax.scatter(range(len(p)), p, **kwargs)
    return fig

def eigenvalues(cavity_array, ax=None, **kwargs):
    eig_vals, eig_vecs = cavity_array.eigenstates()
    if ax is not None:
        kw = {'fmt': '.'}
        kw.update(kwargs)
        ax.errorbar(range(len(eig_vals)), np.real(eig_vals), abs(np.imag(eig_vals)), **kw)
        return ax
    kw = kwargs_sep(Figure, kwargs)
    fig = Figure(**kw)
    kw = kwargs_sep(fig.subplots, kwargs)
    ax = fig.subplots(**kw)
    kw = dict(fmt='.') #default fmt
    kw.update(kwargs_sep(ax.errorbar, kwargs))
    ax.set_ylabel('Energy')
    ax.set_xlabel('Eigenvector')
    ax.errorbar(range(len(eig_vals)), np.real(eig_vals), abs(np.imag(eig_vals)), **kw)
    return fig



def eigenvectors(cavity_array, eigenvectors=None, axes=None, kind='line', prob=True, set_labels=True, **kwargs):
    """
    Args:
        cavity_array: multi_cavity.CavityArray
        eigenvectors: optional list of eigenvectors to be plotted; if None all are plotted
        axes: optional list of axes plot on
        kind: 'line', 'bar' plot type; default is 'line'
        prob: bool, True to plot magnitude**2 of the eigenvector components else plot real and imag components separately
        kwargs: keyword args
    Returns:
        plot of the eigenvectors; creates and returns fig if ax is None
    """
    _, eig_vecs = cavity_array.eigenstates()
    x = range(eig_vecs.shape[0])
    if eigenvectors is not None:
        eig_vecs = eig_vecs[:,eigenvectors]
    
    #if axes provided
    if axes is not None:
        for i, vec in enumerate(eig_vecs.T):
            if kind == 'line':
                if prob:
                    axes.flat[i].plot(x, np.abs(vec)**2, **kwargs)
                else:
                    axes.flat[i].plot(x, np.real(vec), label='real', **kwargs)
                    axes.flat[i].plot(x, np.imag(vec), label='imag', **kwargs)
            elif kind == 'bar':
                if prob:
                    axes.flat[i].bar(x, np.abs(vec)**2, **kwargs)
                else:
                    axes.flat[i].bar(x, np.real(vec), label='real', **kwargs)
                    axes.flat[i].bar(x, np.imag(vec), label='imag', **kwargs)
        return axes
    
    n = eig_vecs.shape[-1]
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    
    #create fig
    kw = dict(figsize=(4*ncols, 4*nrows)) #default figsize
    kw.update(kwargs_sep(Figure, kwargs))
    fig = Figure(**kw)
    
    #create axes
    #default subplots keywords
    kw = dict(nrows=nrows, ncols=ncols, sharex='all', sharey='all', squeeze=False, gridspec_kw={'wspace':0, 'hspace':0})
    kw.update(kwargs_sep(fig.subplots, kwargs))
    axes = fig.subplots(**kw)
    
    #plot
    kw = None
    if  kind == 'line':
        kw = kwargs_sep(axes.flat[0].plot, kwargs)
    elif kind == 'bar':
        kw = kwargs_sep(axes.flat[0].bar, kwargs)

    for i, vec in enumerate(eig_vecs.T):
        if kind == 'line':
            if prob:
                axes.flat[i].plot(x, np.abs(vec)**2, **kw)
            else:
                axes.flat[i].plot(x, np.real(vec), label='real', **kw)
                axes.flat[i].plot(x, np.imag(vec), label='imag', **kw)
        elif kind == 'bar':
            if prob:
                axes.flat[i].bar(x, np.abs(vec)**2, **kw)
            else:
                axes.flat[i].bar(x, np.real(vec), label='real', **kw)
                axes.flat[i].bar(x, np.imag(vec), label='imag', **kw)
        if set_labels:
            labels = cavity_array.states.labels()
            for ax in axes.flat:
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation='vertical')
    if not prob:
        axes.flat[0].legend()
    return fig


#############################################################################################################
#cavity occupancy

def cavity_occupancy(cavity_array, eigenvectors=None, axes=None, kind='bar', **kwargs):
    """
    Args:
        cavity_array: multi_cavity.CavityArray
        eigenvectors: optional list of eigenvectors to be plotted; if None all are plotted
        axes: optional list of axes plot on
        kind: 'line', 'bar' plot type; default is 'line'; bar is stacked
        kwargs: keyword args
    Returns:
        plot of the cavity occupancy for each eigenvector; creates and returns fig if ax is None
    """
    _, eig_vecs = cavity_array.eigenstates()
    x = range(cavity_array.num_cavities)
    photon_expect = metrics.photon_expect(cavity_array)
    excite_expect = metrics.excite_expect(cavity_array)
    if eigenvectors is not None:
        photon_expect = photon_expect[eigenvectors,:]
        excite_expect = excite_expect[eigenvectors,:]
    
    #if axes provided
    if axes is not None:
        for i, (ph_vec, em_vec) in enumerate(zip(photon_expect, excite_expect)):
            if kind == 'line':
                axes.flat[i].plot(x, ph_vec, **kwargs)
                axes.flat[i].plot(x, em_vec, **kwargs)
            elif kind == 'bar':
                axes.flat[i].bar(x, ph_vec, **kwargs)
                axes.flat[i].bar(x, em_vec, bottom=ph_vec, **kwargs)
        return axes
    
    n = photon_expect.shape[0]
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    
    #create fig
    kw = dict(figsize=(4*ncols, 4*nrows)) #default figsize
    kw.update(kwargs_sep(Figure, kwargs))
    fig = Figure(**kw)
    
    #create axes
    #default subplots keywords
    kw = dict(nrows=nrows, ncols=ncols, sharex='all', sharey='all', squeeze=False, gridspec_kw={'wspace':0, 'hspace':0})
    kw.update(kwargs_sep(fig.subplots, kwargs))
    axes = fig.subplots(**kw)
    
    #plot
    kw = None
    if  kind == 'line':
        kw = kwargs_sep(axes.flat[0].plot, kwargs)
    elif kind == 'bar':
        kw = kwargs_sep(axes.flat[0].bar, kwargs)

    for i, (ph_vec, em_vec) in enumerate(zip(photon_expect, excite_expect)):
        if kind == 'line':
            axes.flat[i].plot(x, ph_vec, label='photon', **kw)
            axes.flat[i].plot(x, em_vec, label='emitters', **kw)
        elif kind == 'bar':
            axes.flat[i].bar(x, ph_vec, label='photon', **kw)
            axes.flat[i].bar(x, em_vec, bottom=ph_vec, label='emitters', **kw)
        if cavity_array.num_cavities < 10:
            axes.flat[i].set_xticks(x)
    axes.flat[0].legend()
    return fig


