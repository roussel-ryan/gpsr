import numpy as np
import matplotlib.pyplot as plt

def compare_screens(measured_images,
                    predicted_images,
                    bins,
                    quad_strengths=None,
                    test_ids=None
                    ):

    xx = np.meshgrid(bins, bins, indexing='ij')

    fig, ax = plt.subplots(2,20, figsize=(20,2))

    for i in range(len(measured_images)):
        ax[0, i].pcolormesh(xx[0]*1e3, xx[1]*1e3,
                            measured_images[i])
        ax[1, i].pcolormesh(xx[0]*1e3, xx[1]*1e3,
                            predicted_images[i])
        
        ax[0,i].set_aspect('equal')
        ax[1,i].set_aspect('equal')

        ax[0,i].tick_params(bottom=False, left=False,
                            labelbottom=False, labelleft=False)
        ax[1,i].tick_params(bottom=False, left=False,
                            labelbottom=False, labelleft=False)
        
        ax[0,0].set_ylabel('measured')
        ax[1,0].set_ylabel('predicted')
        
        if quad_strengths is not None:
            ax[0,i].set_title(f'{quad_strengths[i][0]:.1f}')

        if test_ids is not None:
            if i in test_ids:
                for spine in ax[0,i].spines.values():
                    spine.set_edgecolor('orange')
                    spine.set_linewidth(2)
                for spine in ax[1,i].spines.values():
                    spine.set_edgecolor('orange')
                    spine.set_linewidth(2)

    plt.subplots_adjust(wspace=0.1,
                        hspace=0.1)
    
    plt.show()