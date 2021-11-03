import pandas as pd
import altair as alt
import copy
import numpy as np

#
# Data transformation and Aggregation functions
#

def normalizeDatBySize(dat, normalize_by=None, sqrt=True, inplace=False):
    if not inplace:
        dat = copy.deepcopy(dat)
        
    if normalize_by is None:
        return dat
    
    if sqrt:
        dat.error *= np.sqrt(dat[normalize_by])
    else:
        dat.error *= dat[normalize_by]
        
    return dat

def cleanByClause(by):
    by = [x for x in by if x != alt.Undefined and x is not None]
    return by

def populationVar(x):
    return np.var(x, ddof=0)

def rawAggregateExperimentResults(dat, by=['qid', 'query_idx', 'query', 'sketch_name']):
    by = cleanByClause(by)
    aggdat = dat.groupby(by, as_index=False).agg(
        bias=('error', np.mean),
        std=('error', np.std),
        rmse=('error', lambda x: np.linalg.norm(x)),
        q5=('error',lambda x: np.quantile(x, 0.05)),
        q25=('error',lambda x: np.quantile(x, 0.25)),
        q75=('error',lambda x: np.quantile(x, 0.75)),
        q95=('error',lambda x: np.quantile(x, 0.95)),
        size=('sketch_size', np.mean),
        size_bytes=('sketch_bytes', np.mean),
        observed_max_size=('sketch_size', np.max),
        count=('error', 'count'),
    )
        
    # two level calculation for variance
    vardat_by_workload = dat.groupby(by + ['data_seed'], as_index=False).agg(
        var=('error', populationVar),
        bias=('error', np.mean),
        count=('error','count'),
    )
    vardat_by_workload['var'] = vardat_by_workload['var'] / np.sqrt(vardat_by_workload['count']-1 + 1e-6)    
    
    vardat = vardat_by_workload.groupby(by, as_index=False).agg(
        var_expectation=('bias', populationVar),
        expected_var=('var', np.mean),
        count_by_seed=('bias', 'count')
    )
    vardat['var_bias'] = (vardat['var_expectation'] + vardat['expected_var']) / (vardat['count_by_seed']-1 + 1e-6)
    aggdat = aggdat.merge(vardat)
    
    aggdat['bias_lower95'] = aggdat['bias'] - 1.96 * np.sqrt(aggdat['var_bias']) 
    aggdat['bias_upper95'] = aggdat['bias'] + 1.96 * np.sqrt(aggdat['var_bias']) 
    
    return aggdat

def meltRawAggregateExperimentResults(aggdat, by=['qid', 'query_idx', 'query', 'sketch_name']):
    by = cleanByClause(by)
    return aggdat.melt(id_vars=by, value_vars=['bias', 'std', 'q5', 'q25','q75', 'q95', 'size', 'size_bytes', 
                                               'bias_lower95', 'bias_upper95'])
    
def aggregateExperimentResults(dat, by=['qid', 'query_idx', 'query', 'sketch_name']):
    aggdat = rawAggregateExperimentResults(dat, by=by)
    return meltRawAggregateExperimentResults(aggdat, by=by)


########################################################################################################


# Filter the melted dataset to pick out the stats useful for each plotting task

def filterErrorStats(aggdat):
    return aggdat[aggdat.variable.isin(['bias', 'std', 'q5', 'q25','q75', 'q95'])]

def filterBias(dat):
    return dat[dat.variable == 'bias']

def filterBiasCI(dat):
    return dat[dat.variable.isin(['bias_lower95', 'bias_upper95'])]

########################################################################################################

color_coding_dict = {
    'bias': 'black',
#    'std': None,
    'q5': 'red',
    'q95': 'red',
    'q25': 'orange',
    'q75': 'orange'
}

########################################################################################################
#
# Plotting functions
# 


def makePlotLabels(column=alt.Undefined, row=alt.Undefined, 
                   normalize_by=None, normalize_by_sqrt=False,
                   base_title="Error"):
    # make title
    if column != alt.Undefined and row != alt.Undefined:
        title_group = f" by {column} x {row}"
    elif column != alt.Undefined:
        title_group = f" by {column}"
    elif row != alt.Undefined:
        title_group = f" by {row}"
    else:
        title_group = ""
    
    title = f"{base_title}{title_group}"
    
    if normalize_by is None:
        yaxis = 'Error'
    elif normalize_by_sqrt:
        yaxis = 'Error * sqrt(size)'
    else:
        yaxis = 'Error * size'

    return title, yaxis

def makeListLike(dat, unit):
    if pd.api.types.is_list_like(unit):
        return unit
    else:
        return [unit]
        
def plotExperimentResults(dat, x='query_idx', unit='qid', 
                          column=alt.Undefined, row=alt.Undefined, 
                          normalize_by=None, normalize_by_sqrt=False,
                          base_title="Error"):
    unit = makeListLike(unit)
    dat = normalizeDatBySize(dat, normalize_by=normalize_by, sqrt=normalize_by_sqrt)
    aggdat = aggregateExperimentResults(dat, by=[x, column, row] + unit)
    
    title, yaxis = makePlotLabels(column=column, row=row, normalize_by=normalize_by, normalize_by_sqrt=normalize_by_sqrt)
        
    # make actual plot   
    chart = alt.Chart(filterErrorStats(aggdat)).mark_line().encode(
        x=x,
        y=alt.Y('value', axis=alt.Axis(title=yaxis)), 
        color=alt.Color('variable', scale=alt.Scale(domain=list(color_coding_dict.keys()), range=list(color_coding_dict.values()))),
        #)'color_coding'),
        detail="variable"
    ).facet(column=column, row=row).properties(
        title=title
    )
    
    return (chart, aggdat)

def plotBias(dat, x='query_idx', unit='qid', column=alt.Undefined, row=alt.Undefined, normalize_by=None, normalize_by_sqrt=False):
    unit = makeListLike(unit)
    raw_aggdat = rawAggregateExperimentResults(dat, by=[x, column, row] + unit)
    
    title, yaxis = makePlotLabels(column=column, row=row, normalize_by=normalize_by, normalize_by_sqrt=normalize_by_sqrt, base_title="Bias")
    
    chart = alt.Chart().mark_line().encode(
        x=f"{x}:Q", 
        y=alt.Y('bias:Q', title="bias"), 
    )
    
    error_bars = alt.Chart().mark_area(opacity=0.3).encode(
        x=f"{x}:Q", 
        y=alt.Y('bias_lower95:Q', title="bias"),
        y2='bias_upper95:Q', 
    )
    
    combined_chart = alt.layer(chart, error_bars, data = raw_aggdat).facet(
        column=column, row=row
    ).properties(
        title=title
    )
    return (combined_chart, raw_aggdat)

def plotErrorVsSize(dat, x='sketch_bytes', unit='qid'):
    unit = makeListLike(unit)
    raw_aggdat = rawAggregateExperimentResults(dat, by=[x, 'sketch_name'] + unit)
    print(raw_aggdat.shape)
    chart = alt.Chart(raw_aggdat).mark_point(size=10, opacity=0.3).encode(
        x=f"{x}:Q", 
        y=alt.Y('rmse:Q', title="rmse"), 
        color=alt.Color('sketch_name'),
    )
    chart = chart + chart.transform_loess(f'{x}', f'rmse', groupby=['sketch_name']).mark_line(size=4)
    return chart, raw_aggdat
    
def plotSizeVsParams(dat, x, size="sketch_size"):
    chart = alt.Chart(dat).mark_point(size=10, opacity=0.3).encode(
        x=f"{x}:Q", 
        y=alt.Y(f'{size}:Q', title=size), 
        color=alt.Color('sketch_name'),
    )
    chart = chart + chart.transform_loess(f'{x}', f'{size}', groupby=['sketch_name']).mark_line(size=4)
    return chart
