from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

#error_metrics = ['edge rmse', 'edge max', 'zonal rmse', 'meridional rmse', 'zonal max', 'meridional max']
error_metrics = ['edge rmse', 'zonal rmse', 'meridional rmse']
#lines = ['-', '--', '-', '-', '--', '--']
lines = ['-', '-', '-']
#colors = ['tab:blue', 'tab:blue', 'tab:orange', 'tab:green', 'tab:orange', 'tab:green']
colors = ['tab:blue', 'tab:orange', 'tab:green']
#order = ['first', None, None, 'second', None, None]
order = ['first', None, 'second',None,'third'] 

runs = OrderedDict()
# Reconstruct at cell center
#runs['4km->128km']  = {
#    'edge rmse': 0.0021671833958852877,
#    'edge max': 0.01445705133470676,
#    'zonal rmse': 0.0482388834104607,
#    'meridional rmse': 0.031874316848447805,
#    'zonal max': 0.09539109902659892,
#    'meridional max': 0.05821511396736412}
#
#runs['4km->64km'] = {
#    'edge rmse': 0.0004946524117206881,
#    'edge max': 0.0060734829573523275,
#    'zonal rmse': 0.012356052893066873,
#    'meridional rmse': 0.008046357902932148,
#    'zonal max': 0.02465931691315615,
#    'meridional max': 0.02113444613425286}
#
#runs['4km->32km'] = {
#    'edge rmse': 0.00039823678178451684,
#    'edge max': 0.0021026727060370343,
#    'zonal rmse': 0.002655615802144595,
#    'meridional rmse': 0.0019824700520184564,
#    'zonal max': 0.005905086975347418,
#    'meridional max': 0.003496279560777804}
#
#runs['4km->16km'] = {
#    'edge rmse': 0.00016073550542934523,
#    'edge max': 0.0023078691919243832,
#    'zonal rmse': 0.0008168237735517267,
#    'meridional rmse': 0.0005486519563447298,
#    'zonal max': 0.004993817938962475,
#    'meridional max': 0.007348222387513426}

# Reconstruct off center
#runs['4km->128km']  = {
#    'edge rmse': 0.0021671833958852877,
#    'edge max': 0.01445705133470676,
#    'zonal rmse': 0.041242084560763635,
#    'meridional rmse': 0.02807784896116916,
#    'zonal max': 0.08647804810676885,
#    'meridional max': 0.06943004272016662}
#
#runs['4km->64km'] = {
#    'edge rmse': 0.0004946524117206881,
#    'edge max': 0.0060734829573523275,
#    'zonal rmse': 0.010571322241902922,
#    'meridional rmse': 0.0069056811372729685,
#    'zonal max': 0.026464246465415897,
#    'meridional max': 0.024144957335867634}
#
#runs['4km->32km'] = {
#    'edge rmse': 0.00039823678178451684,
#    'edge max': 0.0021026727060370343,
#    'zonal rmse': 0.0023127077732377124,
#    'meridional rmse': 0.00189495995471102,
#    'zonal max': 0.006442613861199931,
#    'meridional max': 0.0044126949407213845}
#
#runs['4km->16km'] = {
#    'edge rmse': 0.00016073550542934523,
#    'edge max': 0.0023078691919243832,
#    'zonal rmse': 0.0007552871317225694,
#    'meridional rmse': 0.0005331792979846265,
#    'zonal max': 0.006639838683589083,
#    'meridional max': 0.009769996288563343}

# 15 quadrature points in remap, center reconstruct
runs['4km->128km']  = {
	'edge rmse': 0.7020444895399857,
	'edge max': 5.473209102272643,
	'zonal rmse': 0.017551828204150723,
	'meridional rmse': 0.011718620225501906,
	'zonal max': 0.046717776492988916,
	'meridional max': 0.027672045784650212}

runs['4km->64km'] = {
	'edge rmse': 0.5089695829266325,
	'edge max': 5.5485367706774085,
	'zonal rmse': 0.0044910427742152655,
	'meridional rmse': 0.002930915338176544,
	'zonal max': 0.010701458812085951,
	'meridional max': 0.008003860549508524}

runs['4km->32km'] = {
	'edge rmse': 0.34617002959723825,
	'edge max': 5.508379025683199,
	'zonal rmse': 0.0010043080829062996,
	'meridional rmse': 0.0007235318189643327,
	'zonal max': 0.0021278019860426767,
	'meridional max': 0.0010476794044393944}

runs['4km->16km'] = {
    'edge rmse': 0.251051486691703,
               # 0.00016073550542934523
    'edge max': 5.410878009698531,
    'zonal rmse': 0.0003447267697141242,
    'meridional rmse': 0.00022437041035580646,
    'zonal max': 0.0006812753497642543,
    'meridional max': 0.00047161624530733803}

resolutions = []
for key, value in runs.items():

    res = key.split('->')[1].replace('km','')
    resolutions.append(float(res))

fig = plt.figure()
ax = fig.add_subplot(111)
for i, metric in enumerate(error_metrics):

    errors = []
    for key, value in runs.items():
        errors.append(runs[key][metric])

    ax.loglog(resolutions, errors, label=metric, linestyle=lines[i], color=colors[i])
    ax.scatter(resolutions, errors, color=colors[i])
    if order[i] is not None:
        order_error = [errors[0]-0.1*errors[0]]
        if order[i] == 'first':
            factor = 2
            alpha = 0.75
        if order[i] == 'second':
            factor = 4
            alpha = 0.25
        for j in range(len(resolutions)-1):
            order_error.append(order_error[j]/factor)

        ax.loglog(resolutions, order_error, label=f'{order[i]} order', color='k', alpha=alpha)
    ax.legend()
    ax.set_xlabel('resolution')
    ax.set_ylabel('error')


fig.savefig('rbf_errors.png')

        


   

