# Code for the plotting function

import numpy as np

import matplotlib.pyplot as plt

import cmocean

#configure pyplot
SMALL_SIZE = 13
MEDIUM_SIZE = 15
BIGGER_SIZE = 17

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'


#savefig params
plt.rcParams['savefig.dpi'] = 300 #dpi when saving
plt.rcParams['savefig.bbox'] = 'tight' #bbox when saving
plt.rcParams['savefig.format'] = 'png'

def plot(triang, gt, pred, **kw):

    fig, allaxs = plt.subplots(5,3, 
                                   figsize = (12,8),
                                  sharex=True, sharey=True,
                                  layout = 'constrained', 
                                  subplot_kw = dict(aspect='equal'))
    axs1, axs2, axs3 = allaxs.T


    #safety
    gt = {k:v.copy() for k,v in gt.items()}
    pred = {k:v.copy() for k,v in pred.items()}

    err = {k:gt[k] - pred[k] for k in gt.keys()}
    err= {k:np.linalg.norm(v, axis = -1) for k,v in err.items()}

    gt['velocity'] = np.linalg.norm(gt['velocity'], axis = -1)
    pred['velocity'] = np.linalg.norm(pred['velocity'], axis = -1)

    for k in ['pressure', 'tke', 'epsilon', 'pottemp']:
        gt[k] = gt[k].squeeze()
        pred[k] = pred[k].squeeze()


    kws = {
        'velocity': dict(cmap = cmocean.cm.dense, **kw),
        'tke':      dict(cmap = cmocean.cm.speed, **kw),
        'epsilon':  dict(cmap = cmocean.cm.amp, **kw),
        'pressure': dict(cmap = plt.get_cmap('bwr'), **kw),
        'pottemp':   dict(cmap = cmocean.cm.thermal, **kw),
    }

    for k in gt.keys():
        cat = np.concat([gt[k], pred[k]])
        kws[k].update(dict(vmin = np.nanmin(cat), vmax = np.nanmax(cat)))

    #center pressure cmap
    pmin, pmax = kws['pressure']['vmin'], kws['pressure']['vmax']
    pmax = np.max([-pmin, pmax])
    kws['pressure']['vmin'] = -pmax
    kws['pressure']['vmax'] =  pmax

    #plot
    for i,field in enumerate(gt.keys()):

        tr = axs1[i].tripcolor(triang, gt[field], **kws.get(field, {}))
        axs2[i].tripcolor(triang, pred[field], **kws.get(field, {}))
        axs3[i].tripcolor(triang, err[field], **kws.get(field, {}))

        fig.colorbar(tr, ax = [axs1[i], axs2[i], axs3[i]], shrink = 0.7, label = field)

    plt.close(fig)
    
    return fig