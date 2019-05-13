import numpy as np
import pandas
from scipy.stats import rankdata

from bayesian_benchmarks.database_utils import Database
from bayesian_benchmarks.data import regression_datasets, classification_datasets
from bayesian_benchmarks.data import _ALL_REGRESSION_DATATSETS, _ALL_CLASSIFICATION_DATATSETS

_ALL_DATASETS = {}
_ALL_DATASETS.update(_ALL_REGRESSION_DATATSETS)
_ALL_DATASETS.update(_ALL_CLASSIFICATION_DATATSETS)

def sort_data_by_N(datasets):
    Ns = [_ALL_DATASETS[dataset].N for dataset in datasets]
    order = np.argsort(Ns)
    return list(np.array(datasets)[order])

regression_datasets = sort_data_by_N(regression_datasets)
classification_datasets = sort_data_by_N(classification_datasets)

database_path = 'results/results.db'

def rank_array(A):
    res = []
    for a in A.reshape([np.prod(A.shape[:-1]), A.shape[-1]]):
        a[np.isnan(a)] = -1e10
        res.append(rankdata(a))
    res = np.array(res)
    return res.reshape(A.shape)


dataset_colors = {
    'challenger':   [0,0],
    'fertility':    [0,0],
    'concreteslump':[0,0],
    'autos':        [0,0],
    'servo':        [0,0],
    'breastcancer': [0,0],
    'machine':      [0,0],
    'yacht':        [0,0],
    'autompg':      [0,0],
    'boston':       [0,0],
    'forest':       [1,0],
    'stock':        [0,0],
    'pendulum':     [0,0],
    'energy':       [0,0],
    'concrete':     [0,0],
    'solar':        [1,0],
    'airfoil':      [0,0],
    'winered':      [0,0],
    'gas':          [0,0],
    'skillcraft':   [0,0],
    'sml':          [0,1],
    'winewhite':    [0,0],
    'parkinsons':   [0,0],
    'kin8nm':       [0,1],
    'pumadyn32nm':  [0,0],
    'power':        [1,0],
    'naval':        [0,0],
    'pol':          [1,1],
    'elevators':    [0,0],
    'bike':         [1,1],
    'kin40k':       [0,1],
    'protein':      [0,0],
    'tamielectric': [0,0],
    'keggdirected': [1,1],
    'slice':        [0,1],
    'keggundirected':[1,0],
    '3droad':       [0,0],
    'song':         [0,0],
    'buzz':         [0,0],
    'nytaxi':       [0,0],
    'houseelectric':[1,1]
}


def read(datasets, models, splits, table, field, extra_text='', highlight_max=True, highlight_non_gaussian=True, use_error_bars=True):
    results = []
    results_test_shapiro_W_median = []

    with Database(database_path) as db:
        for dataset in datasets:
            for dd in models:
                for split in splits:
                    d = {'dataset': dataset,
                         'split' : split}
                    d.update({'iterations':100000})
                    d.update({k:dd[k] for k in ['configuration', 'mode']})

                    if True:# _ALL_REGRESSION_DATATSETS[dataset].N < 1000:
                        res = db.read(table, [field, 'test_shapiro_W_median'], d)
                    else:
                        res = []

                    if len(res) > 0:
                        try:
                            results.append(float(res[0][0]))
                            results_test_shapiro_W_median.append(float(res[0][1]))

                        except:
                            print(res, d, dataset)
                            # results.append(np.nan)
                            # results_test_shapiro_W_median.append(np.nan)
                    else:
                        results.append(np.nan)
                        results_test_shapiro_W_median.append(np.nan)

    results = np.array(results).reshape(len(datasets), len(models), len(splits))
    results_test_shapiro_W_median = np.array(results_test_shapiro_W_median).reshape(len(datasets), len(models), len(splits))
    results_test_shapiro_W_median = np.average(results_test_shapiro_W_median, -1)

    results_mean = np.nanmean(results, -1)
    results_std_err = np.nanstd(results, -1)/float(len(splits))**0.5

    argmax = np.argmax(results_mean, 1)
    lower_pts = [m[a]-e[a] for m, e, a in zip(results_mean, results_std_err, argmax)]
    high_pts = results_mean + results_std_err
    argmaxes = [np.where(h>l)[0] for h, l in zip(high_pts, lower_pts)]

    rs = rank_array(np.transpose(results, [0, 2, 1]))

    rs_flat = rs.reshape(len(datasets) * len(splits), len(models))
    avg_ranks = np.average(rs_flat, 0)
    std_ranks = np.std(rs_flat, 0) / float(len(datasets) * len(splits))**0.5
    r = ['{:.2f} ({:.2f})'.format(m, s) for m, s in zip(avg_ranks, std_ranks)]

    res_combined = []
    for i, (ms, es, Ws) in enumerate(zip(results_mean, results_std_err, results_test_shapiro_W_median)):
        for j, (m, e, W) in enumerate(zip(ms, es, Ws)):
            if field == 'test_shapiro_W_median':
                if m < 0.999:
                    res_combined.append('{:.4f}'.format(m))
                else:
                    res_combined.append(r' ')


            else:
                if m > -1000:
                    if use_error_bars:
                        if m > -10:
                            t = '{:.2f} ({:.2f})'.format(m, e)
                        else:
                            t = '{:.0f} ({:.0f})'.format(m, e)
                    else:
                        if m > -10:
                            t = '{:.2f}'.format(m)
                        else:
                            t = '{:.0f}'.format(m)

                    if highlight_max and (j in argmaxes[i]):
                        t = r'\textbf{' + t + '}'
                    if highlight_non_gaussian and (W<0.99):
                        t = r'\textit{' + t + '}'
                    res_combined.append(t)
                else:
                    res_combined.append('$-\infty$')

    results_pandas = np.array(res_combined).reshape(results_mean.shape)

    extra_fields = []
    extra_fields.append('Avg ranks')
    results_pandas = np.concatenate([results_pandas, np.array(r).reshape(1, -1)], 0)

    extra_fields.append('Median diff from gp')
    ind = np.where(np.array([mm['nice_name'] for mm in models])=='G')[0][0]

    median = np.nanmedian(np.transpose(results - results[:, ind, :][:, None, :], [0, 2, 1]).reshape(len(datasets)*len(splits), len(models)), 0)
    median = ['{:.2f}'.format(m) for m in median]
    results_pandas = np.concatenate([results_pandas, np.array(median).reshape(1, -1)], 0)

    _datasets = []
    for d in datasets:
        if 'wilson' in d:
            nd = d[len('wilson_'):]
        else:
            nd = d

        if (dataset_colors[nd][0] == 0) and  (dataset_colors[nd][1] == 0):
            _d = nd

        elif (dataset_colors[nd][0] == 1) and  (dataset_colors[nd][1] == 0):
            _d = r'{\color{myAcolor} \textbf{' + nd + '}\myAcolormarker}'

        elif (dataset_colors[nd][0] == 0) and  (dataset_colors[nd][1] == 1):
            _d = r'{\color{myBcolor} \textbf{' + nd + '}\myBcolormarker}'

        elif (dataset_colors[nd][0] == 1) and  (dataset_colors[nd][1] == 1):
            _d = r'{\color{myCcolor} \textbf{' + nd + '}\myCcolormarker}'

        _datasets.append(_d)

    res = pandas.DataFrame(data=results_pandas, index=_datasets + extra_fields, columns=[m['nice_name'] for m in models])
    res.insert(0, 'N', [_ALL_DATASETS[dataset].N for dataset in datasets] + [' ',] * len(extra_fields))
    res.insert(1, 'D', [_ALL_DATASETS[dataset].D for dataset in datasets] + [' ',] * len(extra_fields))

    if hasattr(_ALL_DATASETS[datasets[0]], 'K'):
        res.insert(2, 'K', [_ALL_DATASETS[dataset].K for dataset in datasets] + [' ',] * len(extra_fields))


    pandas.DataFrame.to_csv(res, 'results_{}_{}{}.csv'.format(table, field, extra_text))#, float_format='%.6f')
    with pandas.option_context("max_colwidth", 1000):
        latex = pandas.DataFrame.to_latex(res, escape=False)

    with open('results_{}_{}{}.tex'.format(table, field, extra_text), 'w') as f:
        f.writelines(latex)


    return results

splits = range(5)

models = [
    {'mode':'CVAE', 'configuration':'', 'nice_name':'Linear'},
    {'mode':'CVAE', 'configuration':'50', 'nice_name':'CVAE 50'},
    # {'mode':'CVAE', 'configuration':'50_50', 'nice_name':'CVAE $50-50$'},
    {'mode':'CVAE', 'configuration':'100_100', 'nice_name':'CVAE $100-100$'},
    # {'mode': 'CVAE', 'configuration': '100_100_100', 'nice_name': '$100-100-100$'},
    {'mode': 'VI', 'configuration': '', 'nice_name': 'G'},
    {'mode': 'VI', 'configuration': 'G5', 'nice_name': 'GG'},
    {'mode': 'VI', 'configuration': 'G5_G5', 'nice_name': 'GGG'},
    {'mode': 'SGHMC', 'configuration': '', 'nice_name': 'G (SGHMC)'},
    {'mode': 'SGHMC', 'configuration': 'G5', 'nice_name': 'GG (SGHMC)'},
    {'mode': 'SGHMC', 'configuration': 'G5_G5', 'nice_name': 'GGG (SGHMC)'},
    {'mode': 'VI', 'configuration': 'L1', 'nice_name':'LG'},
    {'mode':'IWAE', 'configuration':'L1', 'nice_name':'LG (IW)'},
    {'mode': 'VI', 'configuration': 'L1_G5', 'nice_name':'LGG'},
    {'mode':'IWAE', 'configuration':'L1_G5', 'nice_name':'LGG (IW)'},
    {'mode': 'VI', 'configuration': 'L1_G5_G5', 'nice_name': 'LGGG'},
    {'mode':'IWAE', 'configuration':'L1_G5_G5', 'nice_name':'LGGG (IW)'},
]


res_test_loglik = read(regression_datasets, models, splits, 'conditional_density_estimation', 'test_loglik')
# res_test_loglik = read(regression_datasets, models, splits, 'conditional_density_estimation', 'test_rmse')
res_test_shapiro_W_median = read(regression_datasets, models, splits, 'conditional_density_estimation', 'test_shapiro_W_median')


models = [
    {'mode': 'VI', 'configuration': '', 'nice_name': 'G'},
    {'mode': 'VI', 'configuration': 'G5', 'nice_name': 'GG'},
    {'mode': 'VI', 'configuration': 'G5_G5', 'nice_name': 'GGG'},
    {'mode': 'VI', 'configuration': 'L1', 'nice_name':'LG'},
    {'mode':'IWAE', 'configuration':'L1', 'nice_name':'LG (IW)'},
    {'mode': 'VI', 'configuration': 'L1_G5', 'nice_name':'LGG'},
    {'mode':'IWAE', 'configuration':'L1_G5', 'nice_name':'LGG (IW)'},
    {'mode': 'VI', 'configuration': 'L1_G5_G5', 'nice_name': 'LGGG'},
    {'mode':'IWAE', 'configuration':'L1_G5_G5', 'nice_name':'LGGG (IW)'},
]


res_test_loglik = read(regression_datasets, models, splits, 'conditional_density_estimation', 'test_loglik', '_gp_only')
