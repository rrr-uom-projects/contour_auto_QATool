import matplotlib
matplotlib.rcParams.update({'font.size': 14})
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from os.path import join

source_dir = "/path/to/directory/containing/training/curve/data/"   ## TODO: update path variable here ##

fig, axs = plt.subplots(ncols=5, figsize=(10,4), sharey=True)

for axdx, ax in enumerate(axs):
    data = np.loadtxt(join(source_dir, f"run-lr1e3_cosS_bs16_aggradd_noPre_fold{axdx+1}_logs-tag-val_loss_avg.csv"), delimiter=',', skiprows=1, usecols=(1,2))
    ax.plot(np.arange(1,51,1), data[:,1], color='r', alpha=0.8, linewidth=2)
    data = np.loadtxt(join(source_dir, f"run-lr1e3_cosS_bs16_aggradd_fold{axdx+1}_logs-tag-val_loss_avg.csv"), delimiter=',', skiprows=1, usecols=(1,2))
    ax.plot(np.arange(1,51,1), data[:,1], color='g', alpha=0.8, linewidth=2)
    #ax.set_title(f"Fold {axdx+1}", y=1.0, pad=-20)
    ax.set_title(f"Fold {axdx+1}")
    ax.set_xticks([0,25,50])
axs[0].set_ylabel("Mean validation loss")
axs[2].set_xlabel("Epoch number")
m_s = []
for color, label in zip(['g','r'], ['Pre-training', 'No pre-training']):
    m_s.append(mlines.Line2D([],[], color=color, marker='s', linestyle='None', mew=0, mec='k', markersize=15, label=label))
#axs[4].legend(handles=m_s, fontsize="14", bbox_to_anchor=(0.5,0.5,0.5,0.5), loc="lower right", framealpha=1,bbox_transform=ax.transAxes)
axs[4].legend(handles=m_s, fontsize="14", loc="upper right", framealpha=1)
plt.subplots_adjust(top=0.987,bottom=0.142,left=0.096,right=0.984,hspace=0.2,wspace=0.1)
plt.show()

