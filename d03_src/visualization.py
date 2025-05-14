############################################################################################
# Functions to visualize flows
#     1. Functions to read Census spatial data
#     2. Functions to read Census yearly state to state migration data
#     3. Functions to read ACS 5-year county to county migration data

############################################################################################

import numpy as np
import pandas as pd
from tqdm import tqdm
import copy

from census import Census
from pygris import counties, zctas
import pygris
import scipy.sparse as ss
from scipy.stats import pearsonr, spearmanr
import statsmodels.stats.weightstats as smw

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch, ConnectionPatch
from matplotlib.colors import TwoSlopeNorm,LogNorm,SymLogNorm, ListedColormap, BoundaryNorm
import matplotlib.lines as mlines
import matplotlib.cm as cm
import matplotlib.ticker as mticker
plt.rcParams['font.family'] = 'Helvetica Light'
plt.rc('axes', unicode_minus=False)

import geopandas as gpd
from pygris.utils import erase_water, shift_geometry

import sys
sys.path.append('../d03_src/')
import vars
import os
import process_census as prc

#####################################################################################################################################
#1. RMSE bar plot:

def plot_RMSE_reduction(estimates, CIs=None, colors=None, ax=None,
                        title=None, title_fontsize=25, labels=None, textsize=20,
                        y_label=True, x_label=True, legend=True, invert=True):
    #Create an axis:
    if ax is None: fig, ax = plt.subplots(figsize=(2,4))
    #Get widths and y_values:
    widths=estimates.values
    y=labels if labels is not None else estimates.index
    #Plot horizontal bars:
    _ = ax.barh(y=y, width=widths,
                label=y,
                xerr=CIs,
                error_kw={'ecolor':'k', 'lw':3, 'capsize':6, 'zorder':10},
                left=0, height=.9, color=colors,
                hatch=None, edgecolor='white', lw=0)
    #Configure x-axis:
    _ = ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    _ = ax.tick_params(axis='x', labelsize=textsize)
    _ = ax.set_xlim(0,100)
    if x_label: _ = ax.set_xlabel('Reduction in RMSE', fontsize=textsize, ha='right', x=1)
    #Configure y-axis:
    _ = ax.tick_params(axis='y', length=0)
    _ = ax.tick_params(axis='y', labelsize=textsize) if y_label else ax.set_yticklabels([])
    #Configure axis:    
    _ = ax.spines[['right', 'top']].set_visible(False)
    if title:  _ = ax.set_title(title, fontsize=title_fontsize, x=0, ha='left', y=1 if not legend else 1.15, va='bottom')
    if legend: _ = ax.legend(bbox_to_anchor=(.5, 1.), loc='lower center', borderaxespad=0, ncols=4, fontsize=textsize, frameon=False)
    if invert: _ = _ = ax.invert_yaxis()
    return ax

#####################################################################################################################################
#2. Correlation plot:

def plot_correlation(x, y, w=None,
                     title=None, title_fontsize=25,
                     y_label=None, x_label=None, negative_x=False, rate=False,
                     ax=None, colors=None, markersize=15, markeralpha=0.4,
                     report_corr=True, corr_val=None, corr_type='pearson', textsize=20,
                     legend=True, unit=None, max_value=None, ticksize=15):
    #Create an axis:
    if ax is None: fig, ax = plt.subplots(figsize=(4,4))

    if unit is not None:
        x = x/unit
        y = y/unit

    #Plot:
    _ = ax.axline((0,0), slope=1, linestyle='--', color='k', alpha=0.5, zorder=-1)
    _ = ax.scatter(x,y, color=colors, s=markersize, alpha=markeralpha)

    #Compute the correlation and annotate:
    if report_corr:
        if corr_val is None:
            if w is None:
                assert corr_type in ['pearson', 'spearman']
                corr_val, _ = pearsonr(x, y) if corr_type == 'pearson' else spearmanr(x,y)
            else:
                assert corr_type == 'pearson', 'If using weights, must use Pearson corr.'
                weighted_stats = smw.DescrStatsW(np.column_stack((x, y)), weights=w)
                corr_val = weighted_stats.corrcoef[0,1]
        _ = ax.annotate(fr'$\rho$ = {corr_val:.3f}',
                        xy=(.1, .95), size=textsize, ha='left', va='top',
                        horizontalalignment='left',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=.5'),
                        xycoords=ax.transAxes)

    #Configure axis:
    _ = ax.set_aspect('equal', adjustable='box')    
    _ = ax.spines[['right', 'top']].set_visible(False)
    if title:  _ = ax.set_title(title, fontsize=title_fontsize, x=0, ha='left', y=1.1, va='bottom')

    #Find max value and round, along with ticks:
    if max_value is None:
        max_value = np.max([np.max(abs(x)), np.max(abs(y))]) if not rate else 1
        max_value = np.round(max_value, -1) + (10 if max_value%1 > .1 else 0)
    ticks = np.arange(-max_value if negative_x else 0, max_value+(10 if max_value>10 else 1), 150 if max_value > 500 else (50 if max_value > 200 else (40 if max_value > 120 else (20 if max_value > 50 else (10 if max_value > 10 else 2)))))
    
    # Configure x-axis:
    if unit == 1_000_000: _ = ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x}M"))
    if unit == 1_000: _ = ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x}k"))
    _ = ax.tick_params(axis='x', labelsize=ticksize)
    _ = ax.set_xlim(-max_value if negative_x else 0, max_value)
    _ = ax.set_xticks(ticks if negative_x else ticks[1:])
    if x_label and not legend:
        _ = ax.set_xlabel(x_label, fontsize=textsize)
        
    # Configure y-axis:
    if unit == 1_000_000: _ = ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x}M"))
    if unit == 1_000: _ = ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x}k"))
    _ = ax.tick_params(axis='y', labelsize=ticksize)
    _ = ax.set_ylim(-max_value if negative_x else 0, max_value)
    _ = ax.set_yticks(ticks if negative_x else ticks[1:])
    if y_label: _ = ax.set_ylabel(y_label, fontsize=textsize)
 
    #Add legend:
    patch = Rectangle((0, 0), 1, 1, facecolor=colors, alpha=1.)
    if legend: _ = ax.legend([patch], [x_label if x_label else ''], fontsize=textsize, frameon=False, loc='upper center', bbox_to_anchor=(.5, -.1 if x_label else (-.2 if unit is None else -3)))

    return ax

#####################################################################################################################################
#3. County population error:

def map_county_population_error(error_df,
                                cmap='PuOr',
                                vmax=80, plot_inset=True, plot_boundary=True,
                                textsize=20, title_fontsize=30, title='Infutor',
                                ax=None,
                                us_gdf=None, county_idx=None, ignore_PR=True):
    
    #Collect the US geodataframe:
    if us_gdf is None:
        us_gdf = shift_geometry(gpd.read_file(f'{vars._census_spatial_dir}processed/COUNTY.gpkg'))
        if county_idx is None: county_idx = prc.get_geography_indices('county', ignore_PR=ignore_PR)
        us_gdf['idx'] = us_gdf.GEOID.astype(int).map(county_idx).astype('Int32')
        us_gdf = us_gdf.dropna(axis=0, subset='idx').sort_values('idx')
        assert us_gdf.idx.iloc[-1] == len(us_gdf)-1, 'Missing indices!'
    if plot_boundary: boundary = gpd.GeoDataFrame(geometry=[us_gdf.geometry.unary_union], crs=us_gdf.crs)

    #Create axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
    else:
        fig = ax.get_figure()
        
    #Create colorbar:
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    #Plot:
    _ = us_gdf.plot(error_df['Infutor'], ax=ax, legend=False, cmap=cmap, norm=norm)

    #Plot inset:
    if plot_inset:
        ax_inset = ax.inset_axes([0.9, 0, 0.3, 0.3])
        _ = us_gdf.plot(error_df['MIGRATE'], ax=ax_inset, legend=False, cmap=cmap, norm=norm)
        _ = ax_inset.set_title('MIGRATE', fontsize=textsize)
        _ = ax_inset.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

    #Configure legend:
    cbar_ax = ax.inset_axes([0, 1., 1., 0.1])
    _ = cbar_ax.set_anchor('S')
    ticks = np.linspace(-vmax, vmax, 9)
    cb = fig.colorbar(cbar, ax=cbar_ax, orientation='horizontal', fraction=1, aspect=40, ticks=ticks,
                      format=mticker.FixedFormatter([f'< {int(ticks[0])}%'] + [f'{int(t)}%' for t in ticks[1:-1]] + [f'> {int(ticks[-1])}%']),
                      extend='both')
    _ = cb.ax.tick_params(labelsize=textsize)
    _ = cbar_ax.axis('off')
    _ = cbar_ax.set_title('County population error in Infutor', fontsize=title_fontsize)

    #Configure axes:
    if title: _ = ax.set_title(title, fontsize=title_fontsize)
    _ = ax.axis('off')
    if plot_boundary:
        _ = boundary.boundary.plot(ax=ax, color='k', alpha=1, linewidth=.5)
        if plot_inset: _ = boundary.boundary.plot(ax=ax_inset, color='k', alpha=1, linewidth=.5)

    return ax

def plot_county_population_error(x_values, y_values,
                                 x_labels=None, x_lims=None, x_ticks=None,
                                 spacing=.3, vmax=80, cmap='PuOr',
                                 markersize=20, markeralpha=.75, textsize=20,
                                 ticksize=15, y_label='County population error in Infutor',
                                 report_corr=True, corr_val=None, corr_type='pearson'):

    N_plots = x_values.shape[1]
    fig, Axes = plt.subplots(figsize=((6+spacing)*N_plots-spacing,6), ncols=N_plots, gridspec_kw={'wspace':spacing}, sharey=True)
    
    #Get colors:
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    colors = cm.get_cmap(cmap)(norm(y_values))
    
    #Do one axis at at a time:
    for ax_idx, ax in enumerate(Axes):
        _ = ax.scatter(x_values[:, ax_idx], y_values, color=colors, s=markersize, alpha=markeralpha)
    
        #Configure axis:
        _ = ax.axhline(0, linestyle='--', color='k', alpha=1, zorder=-3)
    
        #Configure axis:
        _ = ax.tick_params(labelsize=textsize)
        _ = ax.spines[['right', 'top']].set_visible(False)
    
        #Configure x-axis:
        if np.all(x_values[:,ax_idx] <= 1.):
            _ = ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        else:
            _ = ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x}y"))
        if x_lims: _ = ax.set_xlim(x_lims[ax_idx])
        if x_labels: _ = ax.set_xlabel(x_labels[ax_idx], fontsize=textsize)
        if x_ticks: _ = ax.set_xticks(x_ticks[ax_idx])
    
        #Configure y-axis:
        _ = ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        if ax_idx == 0 and y_label: _ = ax.set_ylabel(y_label, fontsize=textsize)
        _ = ax.tick_params(axis='both', labelsize=ticksize)
    
        #Include correlation:
        if report_corr:
            assert corr_type in ['pearson', 'spearman']
            if corr_val is None:
                rho, _ = pearsonr(x_values[:, ax_idx], y_values) if corr_type == 'pearson' else spearmanr(x_values[:, ax_idx], y_values)
            else:
                rho = corr_val[ax_idx] 
            _ = ax.annotate(fr'$\rho$ = {rho:.3f}',
                            xy=(.1, .95), size=textsize, ha='left', va='top',
                            horizontalalignment='left',
                            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=.5'),
                            xycoords=ax.transAxes)
    return Axes
    
def plot_CBG_population_error(est_df, std_df,
                              spacing=.15, title_fontsize=25, textsize=20,
                              dataset_colors = {'Infutor':'thistle', 'MIGRATE':'lightseagreen'},
                              renames = {'Under 18 years': '< 18',
                                       '18 to 24 years': '18-24',
                                       '25 to 44 years': '25-44',
                                       '45 to 64 years': '45-64',
                                       '65 years and over': '> 65',
                                       'Owner occupied': 'Owner',
                                       'Renter occupied': 'Renter'},
                              demographics_to_plot=None, base_width=2, height=4, 
                              y_label='Bias in population estimate', grid=False, grid_kwargs=None):

    #Process dfs:
    if renames is not None:
        est_df = est_df.rename(index=renames, level=1)
        std_df = std_df.rename(index=renames, level=1)
    if demographics_to_plot is None:
        demographics_to_plot = est_df.index.get_level_values(0).unique()
    if grid_kwargs is None:
        grid_kwargs = {'linestyle': '--', 'linewidth': 1, 'color': 'gray', 'alpha': 0.7, 'zorder':-100}
    #Create axis:
    widths = est_df.groupby(level=0).count().iloc[:,0].loc[demographics_to_plot].values
    fig, Axes = plt.subplots(figsize=(base_width*est_df.shape[0] + (len(demographics_to_plot)-1)*spacing, height),
                             ncols=len(demographics_to_plot),
                             gridspec_kw={'wspace': spacing},
                             sharey=True, width_ratios=widths)
    
    #Go one group at a time:
    for demographic_group, ax in zip(demographics_to_plot, Axes):
        
        #Get bar estimates and standard deviations:
        est = est_df.loc[demographic_group][dataset_colors.keys()]
        std = std_df.loc[demographic_group][dataset_colors.keys()]
    
        #Plot:
        ax = est.plot(kind='bar',
                      yerr=std, error_kw={'ecolor':'k', 'lw':2, 'capsize':4, 'zorder':2},
                      width=.8, hatch=None, edgecolor=None, lw=0, color=dataset_colors.values(),
                      ax=ax, legend=False, zorder=1)    
        
        #Configure ax:
        _ = ax.spines[['right', 'top']].set_visible(False)
        _ = ax.set_title(demographic_group.split(' (')[0], ha='left', x=0, y=1.06, fontsize=title_fontsize)
        if y_label: _ = ax.set_ylabel(y_label, fontsize=textsize)
    
        #Configure ticks and labels:
        _ = ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        _ = ax.tick_params(axis='both', length=5)
        _ = ax.tick_params(axis='both', labelsize=textsize, rotation=0)


    #Plot a horizontal line:
    for ax in Axes:
        if grid: _ = ax.yaxis.grid(True, **grid_kwargs)
        if ax.get_ylim()[0] < 0: _ = ax.axhline(y=0, linestyle='--', color='k', zorder=2, lw=1)
    
    return Axes


#####################################################################################################################################

def map(values,
        cmap='RdPu',
        two_sided=False, vmin=0, vmax=100,
        plot_boundary=True, boundary_gdf=None, legend=True, n_ticks=6,
        textsize=20, title_fontsize=25, ax=None,
        us_gdf=None, CBG_idx=None, ignore_PR=True,
        indices=None, clear_water=True,
        title='Average domestic out-migration rate (per 1,000 people)',
        annotate=False, user_centroids={'36081': (-73.8, 40.72)}, annotate_size=15, annotate_format=lambda x: x, include_label_in_annotation=False,
        axis_off=True, colorbar_placement=[.1, -.1, .8, .1], extend_above=True,
        mask=None, mask_color='lightgrey'):
    
    #Collect the US geodataframe:
    if us_gdf is None:
        us_gdf = shift_geometry(gpd.read_file(f'{vars._census_spatial_dir}processed/BLOCK_GROUP.gpkg'))
        if CBG_idx is None: CBG_idx = prc.get_geography_indices('blockgroup', ignore_PR=ignore_PR)
        us_gdf['idx'] = us_gdf.GEOID.astype(int).map(CBG_idx).astype('Int32')
        us_gdf = us_gdf.dropna(axis=0, subset='idx').sort_values('idx')
        assert us_gdf.idx.iloc[-1] == len(us_gdf)-1, 'Missing indices!'

    #Create a plot column:
    us_gdf['plot'] = values
    
    #Select indices:
    if indices is not None:
        selected_gdf = us_gdf[us_gdf['idx'].isin(indices)].to_crs('EPSG:4326')
        if boundary_gdf is not None: boundary_gdf = boundary_gdf.to_crs('EPSG:4326')
        if mask is not None: mask = mask.to_crs('EPSG:4326')
        if clear_water: selected_gdf = erase_water(selected_gdf)
        us_gdf = selected_gdf
    
    if plot_boundary:
        if boundary_gdf is None: boundary_gdf = us_gdf
        boundary = gpd.GeoDataFrame(geometry=[boundary_gdf.geometry.unary_union], crs=boundary_gdf.crs)

    #Create axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
    else:
        fig = ax.get_figure()

    #Create colorbar:
    norm = TwoSlopeNorm(vmin=min(vmin,-vmax), vcenter=0, vmax=vmax) if two_sided else plt.Normalize(vmin=vmin, vmax=vmax)
    cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    #Plot:
    _ = us_gdf.plot('plot', ax=ax, legend=False, cmap=cmap, norm=norm, missing_kwds={'color':mask_color})

    #Configure axis:
    if indices is None:
        _ = ax.set_xlim(left=-3.25*1e6, right=2.3*1e6)
        _ = ax.set_ylim(top=1.75*1e6)
    if axis_off:
        _ = ax.axis('off')
    else:
        _ = ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
    if plot_boundary: _ = boundary.boundary.plot(ax=ax, color='k', alpha=1, linewidth=.5)
    
    #Configure legend:
    if legend:
        #Get ticks and labels:
        ticks = np.linspace(min(vmin, -vmax) if two_sided else vmin, vmax, n_ticks)
        tick_labels = [annotate_format(t) for t in ticks]
        if (two_sided or extend_above and vmin > 0): tick_labels[0] = f'< {tick_labels[0]}'
        if extend_above: tick_labels[-1] = f'> {tick_labels[-1]}'
        #Plot colorbar:
        colorbar_anchor = 'SW' if colorbar_placement == 'NY' else 'S'
        if type(colorbar_placement)==str and colorbar_placement == 'NY': colorbar_placement = [0.3, .1, .7, .05]
        cbar_ax = ax.inset_axes(colorbar_placement)
        _ = cbar_ax.set_anchor(colorbar_anchor)
        cb = fig.colorbar(cbar,
                          ax=cbar_ax,
                          orientation='horizontal',
                          fraction=1,
                          aspect=40,
                          ticks=ticks,
                          format=mticker.FixedFormatter(tick_labels),
                          extend='both' if (two_sided or extend_above and vmin > 0) else ('max' if extend_above else 'neither'))
        _ = cb.ax.tick_params(labelsize=textsize)
        _ = cbar_ax.axis('off')
        _ = cbar_ax.set_title(title, fontsize=title_fontsize)

    #Mask:
    if mask is not None:
        _ = mask.plot(ax=ax, color=mask_color)

    #Annotate with values:
    def annotate_with_values(row, _ax=ax, include_label=include_label_in_annotation):
        xy = user_centroids.get(row.GEOID, row.geometry.centroid.coords[0])
        text = annotate_format(row['plot'])
        if include_label: text = row['label'] + '\n' + text
        _ = _ax.annotate(text=text, xy=xy,
                         ha='center', color='black', fontsize=annotate_size,
                         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=.5'))
        return None
    if annotate: _ = us_gdf.apply(annotate_with_values, axis=1)

    return ax

#####################################################################################################################################
#NATIONAL SUMMARIES:

def plot_stacked_bars(df, row_groups=None, colors='summer_r',
                      ax=None, figsize=(18, 8), show=False,
                      group_spacing=0.5,
                      title=None,
                      fontsize=20, fontsize_xlabel=20, fontsize_text=15, fontsize_title=25,
                      discretized_colorbar_legend=False, distance_unit_label=None, distance_unit_label_position='right', colorbar_top=False):
    """
    Plots a horizontal stacked bar chart from a row-normalized DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame where each row corresponds to a group and each column is a category of movers.

    row_groups : dict or None, default=None
        Dictionary where keys are group names and values are lists of row labels in the desired order.
        If None, all rows are included as a single group.

    colors : str or list, default='summer_r'
        Either a colormap name (e.g., 'tab10', 'viridis', 'summer_r') or a list of colors to use for the columns.

    ax : matplotlib.axes.Axes or None, default=None
        Optional axis to plot on. If None, a new figure and axis are created.

    figsize : tuple, default=(18, 8)
        Size of the figure if `ax` is None.

    show : bool, default=False
        Whether to immediately show the plot using `plt.show()`.

    group_spacing : float, default=0.5
        Vertical spacing between groups in y-axis units.

    title : str or None, default=None
        Optional title to display above the plot.

    fontsize : int, default=20
        Font size for axis tick labels.

    fontsize_xlabel : int, default=20
        Font size for the x-axis label.

    fontsize_text : int, default=15
        Font size for text annotations inside bars.

    fontsize_title : int, default=25
        Font size for the plot title.
        
    discretized_colorbar_legend : bool, default=False
        If True, formats the legend to resemble a discretized colorbar with ordinal bins.

    distance_unit_label : str or None, default=None
        Optional unit label to display below the legend (e.g., 'miles') when
        `discretized_colorbar_legend=True`.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the stacked horizontal bar plot.
    """
    #Collect groups:
    if row_groups is None: row_groups = {'All': list(df.index)}

    # Flatten row order and compute y positions with spacing
    row_order = sum(row_groups.values(), [])
    group_sizes = [len(g) for g in row_groups.values()]
    offsets = np.cumsum([0] + [s + group_spacing for s in group_sizes[:-1]])
    y_positions = np.hstack([np.arange(s) + off for s, off in zip(group_sizes, offsets)])

    #Normalize and re-order:
    df = df.loc[row_order]
    df = df.div(df.sum(axis=1), axis=0)

    #Get colors:
    if isinstance(colors, str): colors = [plt.get_cmap(colors)(i/len(df.columns)) for i in range(len(df.columns))]
        
    #Get axis:
    if ax is None: fig, ax = plt.subplots(figsize=figsize)

    #Plot one column at a time:
    left = np.zeros(len(df))
    for i, col in enumerate(df.columns):
        values = df[col].values
        bars = ax.barh(y_positions, values, left=left, height=0.8, color=colors[i], label=col)
        #Annotate:
        for j, (val, bar) in enumerate(zip(values, bars)):
            text_x = left[j] + val / 2
            text = f"{df.iloc[j, i]:.0%}"
            if df.iloc[j, i] >.01: _ = ax.text(text_x, y_positions[j], text, ha='center', va='center', color='black', fontsize=fontsize_text)
        left += values

    #Configure the y axis:
    _ = ax.set_yticks(y_positions)
    _ = ax.set_yticklabels(row_order)
    _ = ax.invert_yaxis()
    
    #Configure the x axis:
    _ = ax.set_xlim(0, 1)
    _ = ax.set_xlabel("Share of Movers", fontsize=fontsize_xlabel)
    _ = ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    #Legend:
    if not discretized_colorbar_legend:
        _ = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.21),
                      ncol=len(df.columns), frameon=False, fontsize=fontsize)
    else:
        #Get the colors:
        cmap = ListedColormap(colors)
        bounds = np.arange(len(df.columns) + 1)
        norm = BoundaryNorm(bounds, cmap.N)
        #Process the colorbar:
        cb_ax = ax.inset_axes([0.1, 1.05, 0.8, 0.05] if colorbar_top else [0.1, -0.18, 0.8, 0.05])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cb = plt.colorbar(sm, cax=cb_ax, orientation='horizontal', ticks=np.arange(len(df.columns)) + 0.5)
        #Set labels:
        labels = list(df.columns) if not distance_unit_label else [label.replace(f" {distance_unit_label}", "") for label in df.columns]
        _ = cb.ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=fontsize)
        if distance_unit_label: _ = cb.ax.set_xlabel(distance_unit_label, fontsize=fontsize, labelpad=10, loc=distance_unit_label_position)

    #Configure ax:
    _ = ax.tick_params(axis='both', labelsize=fontsize)
    _ = ax.spines[['right', 'top']].set_visible(False)
    if colorbar_top and title:
        _ = cb.ax.set_title(title, fontsize=fontsize_title, pad=10)
    else:
        _ = ax.set_title(title, fontsize=fontsize_title)
    
    if show: plt.show()

    return ax

def plot_stacked_bars_vertical(df, colors='summer_r',
                                ax=None, figsize=(14, 8), show=False,
                                title=None,
                                fontsize=20, fontsize_ylabel=20, fontsize_text=15, fontsize_title=25,
                                discretized_colorbar_legend=False, distance_unit_label=None, distance_unit_label_position='right', colorbar_top=False,
                                plot_yearly_value=False, yearly_values=None, yearly_label='Average Distance Moved (Miles)', yearly_color='teal', yearly_linewidth=3, yearly_range=None):
    """
    Plots a vertical stacked bar chart from a row-normalized DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame where each row corresponds to a year and each column is a category.

    colors : str or list, default='summer_r'
        Either a colormap name or a list of colors to use for the columns.

    ax : matplotlib.axes.Axes or None, default=None
        Optional axis to plot on. If None, a new figure and axis are created.

    figsize : tuple, default=(18, 8)
        Size of the figure if `ax` is None.

    show : bool, default=False
        Whether to immediately show the plot using `plt.show()`.

    title : str or None, default=None
        Optional title to display above the plot.

    fontsize : int, default=20
        Font size for axis tick labels.

    fontsize_ylabel : int, default=20
        Font size for the y-axis label.

    fontsize_text : int, default=15
        Font size for text annotations inside bars.

    fontsize_title : int, default=25
        Font size for the plot title.

    discretized_colorbar_legend : bool, default=False
        If True, formats the legend to resemble a discretized colorbar with ordinal bins.

    distance_unit_label : str or None, default=None
        Optional unit label to display below the legend (e.g., 'miles') when
        `discretized_colorbar_legend=True`.
        
    colorbar_top : bool, default=False
        If True, places the colorbar horizontally above the plot instead of as a vertical legend.

    plot_yearly_value : bool, default=False
        If True, overlays a secondary y-axis line plot representing `yearly_values`.

    yearly_values : array-like or None, default=None
        An array of values to plot over time (must match number of rows in `df`).

    yearly_label : str, default='Total Movers'
        Label for the secondary y-axis and line.

    yearly_color : str, default='black'
        Color for the secondary y-axis line and label.

    yearly_linewidth : float, default=3
        Line width for the secondary y-axis line.

    yearly_range : tuple or None, default=None
        Tuple (ymin, ymax) to manually set the y-limits for the secondary y-axis. If None, the axis limits are set automatically.


    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the stacked vertical bar plot.
    """
    # Normalize:
    df = df.div(df.sum(axis=1), axis=0)

    # Get colors:
    if isinstance(colors, str): colors = [plt.get_cmap(colors)(i / len(df.columns)) for i in range(len(df.columns))]

    # Get axis:
    if ax is None: fig, ax = plt.subplots(figsize=figsize)

    # Plot:
    bottom = np.zeros(len(df))
    x_positions = np.arange(len(df))
    for i, col in enumerate(df.columns):
        values = df[col].values
        bars = ax.bar(x_positions, values, bottom=bottom, width=0.8, color=colors[i], label=col)
        for j, (val, bar) in enumerate(zip(values, bars)):
            text_y = bottom[j] + val / 2
            text = f"{df.iloc[j, i]:.0%}"
            if df.iloc[j, i] > 0.01:
                _ = ax.text(x_positions[j], text_y, text, ha='center', va='center', color='black', fontsize=fontsize_text)
        bottom += values

    # Configure x-axis:
    _ = ax.set_xticks(x_positions)
    _ = ax.set_xticklabels(df.index)
    
    # Configure y-axis:
    _ = ax.set_ylim(0, 1)
    _ = ax.set_ylabel("Share of Movers", fontsize=fontsize_ylabel)
    _ = ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # Legend:
    if not discretized_colorbar_legend:
        _ = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
                      ncol=len(df.columns), frameon=False, fontsize=fontsize)
    else:
        cmap = ListedColormap(colors)
        bounds = np.arange(len(df.columns) + 1)
        norm = BoundaryNorm(bounds, cmap.N)
        cb_ax = ax.inset_axes([0.1, 1.075, 0.8, 0.05] if colorbar_top else [1.05, 0.2, 0.03, 0.6])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cb = plt.colorbar(sm, cax=cb_ax,
                          orientation='horizontal' if colorbar_top else 'vertical',
                          ticks=np.arange(len(df.columns)) + 0.5)
        labels = list(df.columns) if not distance_unit_label else [label.replace(f" {distance_unit_label}", "") for label in df.columns]
        if colorbar_top:
            _ = cb.ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=fontsize)
            if distance_unit_label:
                _ = cb.ax.set_xlabel(distance_unit_label, fontsize=fontsize, labelpad=10, loc=distance_unit_label_position)
        else:
            _ = cb.ax.set_yticklabels(labels, fontsize=fontsize)
            if distance_unit_label:
                _ = cb.ax.set_ylabel(distance_unit_label, fontsize=fontsize, labelpad=10, loc=distance_unit_label_position)


    #Configure ax:
    _ = ax.tick_params(axis='both', labelsize=fontsize)
    _ = ax.spines[['right', 'top']].set_visible(False)
    _ = ax.set_title(title, fontsize=fontsize_title)

    #Yearly values?
    if plot_yearly_value:
        if yearly_values is None or len(yearly_values) != len(df):
            raise ValueError("`yearly_values` must be provided and have the same length as the number of years (rows of df).")
    
        # Create twin axis
        ax2 = ax.twinx()
    
        # Align with bar positions
        x_positions = np.arange(len(df))
    
        # Plot line on top of bars
        _ = ax2.plot(x_positions, yearly_values, color=yearly_color, linewidth=yearly_linewidth, marker='o', label=yearly_label, zorder=-1, alpha=0.6)
        _ = ax2.set_ylabel(yearly_label, fontsize=fontsize_ylabel, color=yearly_color)
        _ = ax2.tick_params(axis='y', labelsize=fontsize, colors=yearly_color)
    
        # Color spine and axis label
        _ = ax2.spines['right'].set_color(yearly_color)
        _ = ax2.spines[['bottom', 'left', 'top']].set_visible(False)
        _ = ax2.yaxis.label.set_color(yearly_color)
        if yearly_range is not None: _ = ax2.set_ylim(*yearly_range)
    
    if show: plt.show()
        

    return ax

def plot_mobility_trends(yearly_trends,
                         baseline=False,
                         ax=None, figsize=(10, 10), show=False,
                         colors='Dark2',
                         legend_loc='lower right',
                         title=None,
                         legend_title='Income Quartile of Origin CBG',
                         ylabel='Average Distance Moved (miles)', yrange=None,
                         markers=True,
                         fontsize=20, fontsize_xlabel=20,
                         fontsize_ylabel=20, fontsize_legend=20, fontsize_title=25):
    """
    Plots mobility yearly trends for different demographic groups.

    Parameters
    ----------
    yearly_trends : pd.DataFrame
        dataframe whose rows are groups and columns are years

    baseline : bool or string, default=False
        Whether to plot a dashed black line representing a baseline, which is a row.
        
    ax : matplotlib.axes.Axes or None, default=None
        Optional axis to plot on. If None, a new figure and axis are created.

    figsize : tuple, default=(10, 6)
        Size of the figure if `ax` is None.

    show : bool, default=False
        Whether to immediately show the plot using `plt.show()`.

    title : str, default=None
        Title to display above the plot.

    fontsize : int, default=20
        Font size for axis tick labels.

    fontsize_xlabel : int, default=20
        Font size for the x-axis label.

    fontsize_ylabel : int, default=20
        Font size for the y-axis label.

    fontsize_legend : int, default=15
        Font size for legend text.

    fontsize_title : int, default=25
        Font size for the plot title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    # Get axis
    if ax is None: fig, ax = plt.subplots(figsize=figsize)

    #Get colors:
    if isinstance(colors, str): colors = [plt.get_cmap(colors)(i/len(yearly_trends)) for i in range(len(yearly_trends))]

    # Extract x values from column names (assumed numeric years)
    x = yearly_trends.columns.astype(int)

    # Plot each group
    for i, (group, values) in enumerate(yearly_trends.iterrows()):
        y = values.values
        is_baseline = (baseline is not False and group == baseline)
        _ = ax.plot(x, y,
                    label=group,
                    color='grey' if is_baseline else colors[i],
                    linestyle='-',
                    linewidth=6 if is_baseline else 2,
                    alpha=0.8,
                    zorder=-1 if is_baseline else 1,
                    marker='o' if markers else None)

    # Axis labels and formatting
    _ = ax.set_xlabel("Year", fontsize=fontsize_xlabel)
    _ = ax.set_ylabel(ylabel, fontsize=fontsize_ylabel)
    _ = ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
    if yrange is not None: _ = ax.set_ylim(*yrange)

    # Legend
    if legend_loc == 'outside top right':
        _ = ax.legend(title=legend_title,
                      fontsize=fontsize_legend, title_fontsize=fontsize,
                      frameon=False,
                      loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    else:
        _ = ax.legend(title=legend_title,
                      fontsize=fontsize_legend, title_fontsize=fontsize,
                      frameon=False,
                      loc=legend_loc)

    # Title and axes
    _ = ax.set_title(title, fontsize=fontsize_title)
    _ = ax.spines[['right', 'top']].set_visible(False)
    _ = ax.tick_params(axis='both', labelsize=fontsize)

    if show: plt.show()

    return ax


def plot_mobility_trends_log(yearly_trends,
                              ax=None, figsize=(10, 6), show=False,
                              colors='Dark2',
                              legend_loc='lower right',
                              title=None,
                              legend_title=None,
                              ylabel='Homophily relative to all movers', vmin=1, vmax=17,
                              fontsize=20, fontsize_xlabel=20,
                              fontsize_ylabel=20, fontsize_legend=20, fontsize_title=25):
    """
    Plots log2-scaled yearly trends for different groups relative to a baseline.

    Parameters
    ----------
    yearly_trends : pd.DataFrame
        DataFrame where rows are groups and columns are years. Values should be ratios
        (e.g., 1.5 means 1.5x the baseline).

    ax : matplotlib.axes.Axes or None, default=None
        Optional axis to plot on. If None, a new figure and axis are created.

    figsize : tuple, default=(10, 6)
        Size of the figure if `ax` is None.

    show : bool, default=False
        Whether to immediately show the plot using `plt.show()`.

    colors : str or list, default='Dark2'
        Either a colormap string or a list of color values to use for the lines.

    legend_loc : str, default='lower right'
        Location of the legend. Can be any valid matplotlib legend location string,
        or "outside top right" for placing outside the plot.

    title : str, default=None
        Title to display above the plot.

    legend_title : str, default=None
        Title for the legend.

    ylabel : str, default='Relative Ratio (log2)'
        Label for the y-axis.

    yvmin : float or None, default=None
        Minimum y-axis value in the original ratio scale (not log2). For example,
        if vmin=0.5, the y-axis will start at -1 (log2(0.5)) corresponding to -2x.

    fontsize : int, default=20
        Font size for axis tick labels.

    fontsize_xlabel : int, default=20
        Font size for the x-axis label.

    fontsize_ylabel : int, default=20
        Font size for the y-axis label.

    fontsize_legend : int, default=20
        Font size for legend text.

    fontsize_title : int, default=25
        Font size for the plot title.

    Returns
    -------
    matplotlib.axes.Axes
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Get colors
    if isinstance(colors, str):
        colors = [plt.get_cmap(colors)(i / len(yearly_trends)) for i in range(len(yearly_trends))]

    x = yearly_trends.columns.astype(int)

    for i, (group, values) in enumerate(yearly_trends.iterrows()):
        y = np.log2(values.values)
        _ = ax.plot(x, y,
                    label=group,
                    color=colors[i],
                    linestyle='-',
                    linewidth=2,
                    alpha=0.9,
                    marker='o')

    # Horizontal reference line at log2(1) = 0
    _ = ax.axhline(0, linestyle='--', color='black', linewidth=1)

    # Y-axis ticks as relative ratios
    ymin = np.log2(vmin)
    ymax = np.log2(vmax)
    log_ticks = np.arange(np.floor(ymin), ymax + 1)
    _ = ax.set_yticks(log_ticks)
    _ = ax.set_yticklabels([f"-{int(2**abs(t))}x" if t < 0 else f"{int(2**t)}x" if t > 0 else "1x" for t in log_ticks])
    _ = ax.set_ylim(ymin, ymax)

    # Axes
    _ = ax.set_xlabel("Year", fontsize=fontsize_xlabel)
    _ = ax.set_ylabel(ylabel, fontsize=fontsize_ylabel)
    _ = ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
    _ = ax.set_title(title, fontsize=fontsize_title)
    _ = ax.tick_params(axis='both', labelsize=fontsize)
    _ = ax.spines[['right', 'top']].set_visible(False)

    # Legend
    if legend_loc == 'outside top right':
        _ = ax.legend(title=legend_title,
                      fontsize=fontsize_legend, title_fontsize=fontsize,
                      frameon=False,
                      loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    else:
        _ = ax.legend(title=legend_title,
                      fontsize=fontsize_legend, title_fontsize=fontsize,
                      frameon=False,
                      loc=legend_loc)

    if show: plt.show()

    return ax

def plot_upwards_mobility(aggregated_rates_per_group, number_of_quantiles=10,
                          plot_random_baseline=True,
                          ax=None, figsize=(10, 10), show=False,
                          colors='Dark2',
                          legend_loc='lower left',
                          title='Homophily in Migration',
                          legend_title='Predominant Race in CBG',
                          ylabel='Share of Movers Moving to Higher-Income CBGs',
                          x_quantity='Income', markers=True,
                          fontsize=20, fontsize_xlabel=20,
                          fontsize_ylabel=20, fontsize_legend=20, fontsize_title=25):
    """
    Plots upward mobility according to an ordinal variable
        (quantiles) for different demographic groups.

    Parameters
    ----------
    aggregated_rates_per_group : dict
        Dictionary where keys are group names (e.g., race categories) and values are
        dictionaries mapping origin quantile (int) to share of movers satisfying a criteria.

    number_of_quantiles : int, default=10
        The number of quantiles (e.g., 10 for deciles, 100 for percentiles).
        Controls x-axis labeling and tick positioning.

    plot_random_baseline : bool, default=False
        Whether to plot a dashed black line representing a random baseline.
        
    ax : matplotlib.axes.Axes or None, default=None
        Optional axis to plot on. If None, a new figure and axis are created.

    figsize : tuple, default=(10, 6)
        Size of the figure if `ax` is None.

    show : bool, default=False
        Whether to immediately show the plot using `plt.show()`.

    title : str, default='Upward Mobility and Household Income'
        Title to display above the plot.

    fontsize : int, default=20
        Font size for axis tick labels.

    fontsize_xlabel : int, default=20
        Font size for the x-axis label.

    fontsize_ylabel : int, default=20
        Font size for the y-axis label.

    fontsize_legend : int, default=15
        Font size for legend text.

    fontsize_title : int, default=25
        Font size for the plot title.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the upward mobility plot.
    """
    # Get axis
    if ax is None: fig, ax = plt.subplots(figsize=figsize)

    #Get colors:
    if isinstance(colors, str): colors = [plt.get_cmap(colors)(i/len(aggregated_rates_per_group)) for i in range(len(aggregated_rates_per_group))]

    # Plot all groups
    x = np.arange(1, number_of_quantiles + 1)
    for i, (group, shares) in enumerate(aggregated_rates_per_group.items()):
        y = [shares.get(d, np.nan) for d in x]
        _ = ax.plot(x, y, label=group, marker='o' if markers else None,
                    zorder=-1 if 'All' in group else 1,
                    color='grey' if 'All' in group else colors[i], alpha=0.8, 
                    linewidth=6 if 'All' in group else 2)

    # Optional: plot the random baseline line
    if plot_random_baseline:
        y_baseline = 1 + (1 -2*x)/(2*number_of_quantiles)
        baseline_line, = ax.plot(x, y_baseline, linestyle='--', color='k', label='Random baseline', zorder=-1, alpha=0.6)

    # Configure x axis
    x_label = f"{x_quantity} {'Decile' if number_of_quantiles == 10 else ('Percentile' if number_of_quantiles == 100 else 'Group')} of Origin CBG"
    _ = ax.set_xlabel(x_label, fontsize=fontsize_xlabel)
    if number_of_quantiles == 10: _ = ax.set_xticks(x)
    _ = ax.set_xlim(0.75, number_of_quantiles + 0.25)

    # Configure y axis
    _ = ax.set_ylabel(ylabel, fontsize=fontsize_ylabel)
    _ = ax.set_ylim(0, 1)
    _ = ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    # Legend
    _ = ax.legend(title=legend_title, fontsize=fontsize_legend, title_fontsize=fontsize, frameon=False, loc=legend_loc)

    # Configure axis
    _ = ax.set_title(title, fontsize=fontsize_title)
    _ = ax.spines[['right', 'top']].set_visible(False)
    _ = ax.tick_params(axis='both', labelsize=fontsize)

    if show: plt.show()

    return ax


def plot_mover_heatmap(df, group_blocks,
                       normalize=False, cmap='YlOrRd',
                       ax=None, figsize=(10, 8),
                       title=None,
                       fontsize=20, fontsize_xlabel=20, fontsize_text=12, fontsize_title=25,
                       cbar_label="Proportion of movers",
                       group_spacing=0.5,
                       random_baseline=False,
                       all_str='All CBGs (observed destination)',
                       random_str='Population share',
                       relative_to_baseline=False):

    """
    Plots a heatmap of mover rates between origin and destination groups, with spacing between groups.

    Parameters
    ----------
    df : pd.DataFrame
        Square or rectangular DataFrame with origin groups as rows and destination groups as columns.

    group_blocks : dict
        Dictionary where keys are group names and values are lists of group labels, applied to both rows and columns.

    normalize : bool, default=False
        Whether to normalize each row to sum to 1.

    cmap : str or Colormap, default='YlOrRd'
        Colormap used for heatmap shading.

    ax : matplotlib.axes.Axes or None, default=None
        Axis to plot on. If None, a new figure and axis are created.

    figsize : tuple, default=(10, 8)
        Size of the figure if ax is None.

    title : str or None, default=None
        Optional title to display above the plot.

    fontsize : int, default=20
        Font size for tick labels.

    fontsize_xlabel : int, default=20
        Font size for axis labels.

    fontsize_text : int, default=12
        Font size for value annotations in heatmap cells.

    fontsize_title : int, default=25
        Font size for plot title.

    cbar_label : str, default="Proportion of movers"
        Label for the colorbar (only used if relative_to_baseline is False).

    group_spacing : float, default=0.5
        Visual spacing between row/column groups in axis units.

    random_baseline : bool or dict, default=False
        If True, adds a row labeled `random_str` that contains the uniform baseline expectation across all columns.
        If a dictionary, uses the values as expected proportions per column (must sum to 1).
        This row will be added directly after the `all_str` row and must be present in the appropriate group in `group_blocks`.

    all_str : str, default='All CBGs'
        The label used to identify the summary row for all movers in both `df` and `group_blocks`.
        Required if `relative_to_baseline=True` to compute relative ratios.

    random_str : str, default='Population share'
        The label used for the added random baseline row. Must be distinct from other rows.

    relative_to_baseline : bool, default=False
        If True, computes values as ratios relative to the row labeled `all_str`.
        The `all_str` row will not be plotted.
        Colors and labels will show overrepresentation as multiples of expected (e.g., "2x") and underrepresentation as negative multiples (e.g., "-2x"),
        with a log2-based color scale centered at 1.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the heatmap.
    """

    # Copy group_blocks so we don't mutate the original and insert random baseline if wanted
    group_blocks = copy.deepcopy(group_blocks)
    baseline_block = next((k for k, v in group_blocks.items() if all_str in v), None)
    if random_baseline:
        group_blocks[baseline_block].append(random_str)

    row_order_full = sum(group_blocks.values(), [])
    col_order = [col for col in row_order_full if col in df.columns]

    df = df.loc[[r for r in row_order_full if r in df.index], col_order]
    if normalize:
        df = df.div(df.sum(axis=1), axis=0)

    #Include the random baseline:
    if random_baseline and random_str not in df.index:
        if isinstance(random_baseline, dict):
            random_row = pd.Series([random_baseline.get(col, 0) for col in col_order], index=col_order, name=random_str)
        elif random_baseline is True:
            uniform_val = 1 / len(col_order)
            random_row = pd.Series([uniform_val for _ in col_order], index=col_order, name=random_str)
        else:
            raise ValueError("random_baseline must be a dict or True if requested")
        df = pd.concat([df, pd.DataFrame([random_row])])

    #Normalize if needed:
    if relative_to_baseline:
        if all_str not in df.index:
            raise ValueError(f"relative_to_baseline=True but '{all_str}' not found in df.index")
        baseline = df.loc[all_str]
        df = df.drop(index=all_str)
        row_order_full = [r for r in row_order_full if r != all_str]
        df = df.div(baseline + 1e-12, axis=1)

    #Collect row and column sizes, positions, and spacings:
    row_order = [r for r in row_order_full if r in df.index]
    row_sizes = []
    for g in group_blocks.values():
        rows_in_df = [r for r in g if r in df.index]
        row_sizes.append(len(rows_in_df))
    row_offsets = np.cumsum([0] + [s + group_spacing for s in row_sizes[:-1]])
    y_positions = np.hstack([np.arange(s) + off for s, off in zip(row_sizes, row_offsets)])
    col_sizes = [len([c for c in g if c in col_order]) for g in group_blocks.values()]
    col_blocks = [[c for c in g if c in col_order] for g in group_blocks.values()]
    col_offsets = np.cumsum([0] + [s + group_spacing for s in col_sizes[:-1]])
    x_positions = np.hstack([np.arange(s) + off for s, off in zip(col_sizes, col_offsets)])
    col_order_spaced = sum(col_blocks, [])

    #Create ax:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    #Plot one cell at a time:
    for i, y in enumerate(y_positions):
        for j, x in enumerate(x_positions):
            row_label = row_order[i]
            val = df.at[row_label, col_order_spaced[j]] if row_label in df.index else np.nan
            if relative_to_baseline and not np.isnan(val):
                log_val = np.log2(val) if val > 0 else np.nan
                color = cm.get_cmap(cmap)((log_val + 2) / 4) if not np.isnan(log_val) else (1, 1, 1, 0)
            else:
                color = cm.get_cmap(cmap)(val) if not np.isnan(val) else (1, 1, 1, 0)
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                       facecolor=color, edgecolor='white', linewidth=0.5))
            if not np.isnan(val):
                if relative_to_baseline:
                    ratio = val
                    if ratio >= 1:
                        label = f"{round(ratio, 1)}x"
                    else:
                        label = f"-{round(1/ratio, 1)}x"
                else:
                    label = "< 1%" if val < 0.005 else f"{int(round(val * 100))}%"
                style = 'italic' if row_label == random_str else 'normal'
                ax.text(x, y, label, ha='center', va='center',
                        fontsize=fontsize_text, color='black', style=style)

    #Cofnigure ax:
    ax.set_xticks(x_positions)
    ax.set_xticklabels(col_order_spaced, rotation=45, ha='right', fontsize=fontsize)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(row_order, fontsize=fontsize)
    ax.set_xlim(x_positions[0] - 0.5, x_positions[-1] + 0.5)
    ax.set_ylim(y_positions[-1] + 0.5, y_positions[0] - 0.5)
    ax.set_xlabel("Destination", fontsize=fontsize_xlabel)
    ax.set_ylabel("Origin", fontsize=fontsize_xlabel)
    ax.set_title(title, fontsize=fontsize_title)
    ax.tick_params(axis='both', length=0, labelsize=fontsize)
    ax.set_aspect('equal')
    [s.set_visible(False) for s in ax.spines.values()]

    #Colorbar:
    if relative_to_baseline:
        from matplotlib.colors import Normalize
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=-2, vmax=2))
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, orientation='vertical', extend='both')
        cbar.set_label(cbar_label, fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.set_ticks([-2, -1, 0, 1, 2])
        cbar.set_ticklabels(["-4x", "-2x", "1x", "2x", "4x"])
    else:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, orientation='vertical')
        cbar.set_label(cbar_label, fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])

    return ax


#######################################################################################################################
#WILDFIRES:

def plot_california(california_gdf, fires_gdf,
                    fire_perimeter=None, ax=None,
                    fire_color='maroon', plot_all_fires=False, axis_off=True):
    #Select the ax:
    if ax is None: fig, ax = plt.subplots(figsize=(6,7))
        
    #Plot the basemap:
    _ = california_gdf.plot(ax=ax, color='lightgray', edgecolor='k', linewidth=.5)
    
    #If we are plotting all the fires:
    if plot_all_fires: _ = fires_gdf.clip(california_gdf).plot(ax=ax, color=fire_color, alpha=.5)
    
    #Plot the studied fires:
    if fire_perimeter is not None:
        for perimeter in fire_perimeter.values(): _ = perimeter.clip(california_gdf).plot(ax=ax, color=fire_color, alpha=1)
    
    #Configure axis:
    if axis_off: _ = ax.axis('off')
    return ax


def map_fire_migration(fire_gdf,
                       fire_name=None, fire_year=None, fire_counties=None,
                       base_map=None, bounds=None,
                       fire_perimeter=None,
                       fire_color='maroon',
                       area_threshold=0.1,
                       cmap='YlOrRd', ax=None, plot_colorbar=True, plot_legend=False, show=False, vmax=50, ax_linewidth=1,
                       show_name=False, per_thousand=True, outmover=True,
                       textsize=15, title_fontsize=20, legend_loc='upper left', legend_bbox=(1,1)):

    #Get the axis:
    if ax is None: fig, ax = plt.subplots(figsize=(6,6))
    
    #Prepare the colorbar:
    norm = plt.Normalize(vmin=0, vmax=vmax)
    cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    #Plot the CBG migration:
    col_to_plot = 'movers' if outmover else 'incoming'
    _ = fire_gdf.plot(f'{col_to_plot}_pth' if per_thousand else f'{col_to_plot}_pct', ax=ax, edgecolor='k', cmap=cmap, norm=norm, linewidth=0)

    #Plot the fire boundary:
    if fire_perimeter is not None:
        _ = fire_perimeter.boundary.plot(ax=ax, color=fire_color, linestyle='--', alpha=1, linewidth=1.5)

    #Plot the basemap:
    if base_map is not None:
        _ = base_map.boundary.plot(ax=ax, edgecolor='k', linewidth=.4)

    #Adjust the bounds:
    if bounds is None:
        assert fire_counties is not None, 'Must either pass counties or bounds'
        bounds = base_map[base_map.NAME.isin(fire_counties)].dissolve().buffer(1_000).bounds.iloc[0].values
    _ = ax.set_xlim(bounds[0], bounds[2])
    _ = ax.set_ylim(bounds[1], bounds[3])
    _ = ax.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)


    #Configure the colorbar:
    if plot_colorbar:
        cbar_ax = fig.add_axes([0.1, .75, 0.8, 0.1])
        ticks = np.linspace(0, vmax, 6)
        cb = fig.colorbar(cbar, ax=cbar_ax, orientation='horizontal', fraction=1, aspect=40,
                          ticks=ticks,
                          format=mticker.FixedFormatter([f'{int(t)}%' for t in ticks]), extend='max')
        _ = cb.ax.tick_params(labelsize=textsize)
        _ = cbar_ax.axis('off')
        _ = cbar_ax.set_title(f"Domestic {'out' if outmover else 'in'}-migration rate in {fireyear+1} (per {1000 if per_thousand else 100} people)", fontsize=title_fontsize)

    #Configure the categorical legend:
    if plot_legend:
        legend_elements = [Patch(facecolor='white', edgecolor=fire_color,  linestyle='--', linewidth=1.5,  label='Fire perimeter'),
                           Patch(facecolor='white', edgecolor='k',   linewidth=1, label='County boundary'),]
        _ = ax.legend(handles=legend_elements,
                      loc=legend_loc, bbox_to_anchor=legend_bbox, borderaxespad=0,
                      fontsize=textsize,frameon=False)

    #Configure the legend:
    if show_name and fire_name is not None:
        _ = ax.annotate(f'{fire_name} fire'+ f' ({fire_year})' if fire_year is not None else '',
                        size=textsize,
                        xy=(0, .95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round", fc="white"))
    #Ajust axis:
    for axis in ['top','bottom','left','right']:
        _ = ax.spines[axis].set_linewidth(ax_linewidth)
    #_ = ax.spines['left'].set_position(('outward', 0))
    #_ = ax.spines['bottom'].set_position(('outward', 0))
        
    if show: plt.show()
        
    return ax
    

def connect_ax(ax, parent_ax,
               bounds=None, end_coords=(-0.05, .95),
               fig=None, start_above=True):

    #Get figure:
    if fig is None: fig = parent_ax.get_figure()

    #Collect the bounds of the fire area:
    if bounds is None:
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        width, height = x_max - x_min, y_max - y_min
    else:
        x_min, y_min  = bounds[0], bounds[1]
        width, height = bounds[2] - x_min, bounds[3] - y_min

    #Plot a rectangle in the parent ax:
    rectangle = Rectangle((x_min, y_min), width, height, edgecolor='k', facecolor='none', linewidth=1.5)
    _ = parent_ax.add_patch(rectangle)

    #Create an arrow connector:
    xA= rectangle.get_x() + rectangle.get_width()/2
    yA = rectangle.get_y() + (rectangle.get_height() if start_above else 0)
    arrow = ConnectionPatch(xyA=(xA, yA),
                            coordsA=parent_ax.transData,
                            xyB=end_coords, coordsB=ax.transAxes,
                            arrowstyle='->', color='k', connectionstyle='angle,angleA=90', linewidth=1.5)
    _ = fig.add_artist(arrow)
    
    return fig

def plot_colorbar(vmax, fig, cbar_ax,
                  one_sided=True, cmap='viridis', n_ticks=6,
                  label_size=15, title_size=20,
                  title=None, title_below=True, title_y=1,
                  cbar_anchor=(.5, 1), cbar_panchor=(.5,-10), cbar_aspect=50, cbar_fraction=1):

    #Create the norm:
    vmin = 0 if one_sided else -vmax
    norm = plt.Normalize(vmin=vmin, vmax=vmax) if one_sided else TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    #Get ticks and formatter:
    ticks = np.linspace(0, vmax, n_ticks)
    ticks_fmt = [f'{int(t)}' for t in ticks[:-1]]+[f'> {int(ticks[-1])}']
    cb = fig.colorbar(cbar, ax=cbar_ax, orientation='horizontal', fraction=cbar_fraction, aspect=cbar_aspect,
                      ticks=ticks, format=mticker.FixedFormatter(ticks_fmt),
                      extend='max',
                      anchor=cbar_anchor, panchor=cbar_panchor, pad=0)
    
    #Configure:
    _ = cb.ax.tick_params(labelsize=label_size)
    if title_below:
        _ = cb.set_label(title, size=title_size)
    else:
        _ = cbar_ax.set_title(title, fontsize=title_size, y=title_y, va='bottom')
    _ = cbar_ax.axis('off')
    
    return cbar_ax

def plot_fire_lines(indices_per_group, migration_estimates,
                    fire_name=None, fire_month=None, fire_year=None, fire_counties=None,
                    ax=None, show_legend=False, vmax=100, tick_step=100, cmap='Set1', fire_color='maroon',
                    label_y=True, label_x=False, x_range=range(2015,2020), x_offset=.3,
                    label_size=15, treatment_label=None, control_label='Outside fire perimeter:',
                    two_column_legend=True,
                    per_thousand=True, five_year_data=False,
                    outmover=True, passing_populations=False,
                    county_inset=False, verbose=False, print_values=False,
                    title_fontsize=25, title=None, tick_size=15):
        
    #Get the axis and temporal range:
    if ax is None: fig, ax = plt.subplots(figsize=(8,4))
    x_vals=list(x_range)
    
    #If we didn't specify the treatment group, assume the first key:
    if treatment_label is None: treatment_label = list(indices_per_group.keys())[0]
    N_groups = len(indices_per_group)
    
    #Iterate over groups to create a summary DataFrame over years:
    group_values = {}
    for col_idx, (group, indices) in enumerate(indices_per_group.items()):
        is_control = False if group == treatment_label else True
        df = pd.DataFrame()
        
        #Populate every year:
        for year, M in migration_estimates.items():
            #Collect outflows:
            M_in_group = M[indices]
            #Select the rows (or entries) of the arrays pertaining to the group:
            if outmover:
                assert not passing_populations, 'Must pass migration matrices to assess outmovers'
                df.loc[:,f'{year-1}_population'] = np.array(M_in_group.sum(axis=1)).flatten()
                df.loc[:,f'{year}_stayers']      = M_in_group[:,indices].diagonal()
                df.loc[:,f'{year}_outmovers']    = df[f'{year-1}_population'] - df[f'{year}_stayers']
            #Populate the DataFrame with populations:
            else:
                df.loc[:,f'{year-1}_population'] = np.array(M_in_group.sum(axis=1)).flatten() if not passing_populations else migration_estimates.get(year-1, migration_estimates[year])[indices]
                df.loc[:,f'{year}_arrivers']  = np.array(M.sum(axis=0)).flatten()[indices] if not passing_populations else M[indices]
                df.loc[:,f'{year}_popchange'] = df.loc[:,f'{year}_arrivers'] - df.loc[:,f'{year-1}_population']
                
        #Collect the y-values by grouping:
        numerator = df[[f"{year}_{'outmovers' if outmover else 'popchange'}" for year in x_vals]].values.T.sum(axis=1)
        y_vals = numerator/df[[f'{year-1}_population' for year in x_vals]].values.T.sum(axis=1)
        y_vals = 1_000*y_vals if per_thousand else 100*y_vals
        group_values[group] = y_vals
        if print_values: print(group, [f'{year}: {y:.1f}' for year,y in zip(x_vals,y_vals)])
                
        #Plot:
        _ = ax.plot(x_vals, y_vals,
                    linewidth=1 if is_control else 3,
                    alpha=.7 if is_control else 1,
                    label=group,
                    color= plt.get_cmap(cmap)(col_idx/N_groups) if is_control else fire_color,
                    marker='o' if is_control else 's',
                    markersize=10,
                    zorder=1 if is_control else 10)
    
    #Configure axis:
    _ = ax.spines[['right', 'top']].set_visible(False)
    _ = ax.tick_params(labelsize=tick_size)
    if not outmover: _ = ax.spines['bottom'].set_position('center')
    if title is not None: _ = ax.set_title(title, fontsize=title_fontsize, x=0, y=1.1, ha='left', va='bottom')
    
    #Configure y-axis:
    _ = ax.set_ylim(bottom=0 if outmover else -vmax, top=vmax)
    if label_y:
        label = 'Out-migration' if outmover else 'Population change'
        if per_thousand:
            label += '\n(per 1,000)'
        else:
             _ = ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        _ = ax.set_ylabel(label, fontsize=label_size, labelpad=10)
    if outmover: _ = ax.set_yticks(np.arange(0,vmax+1,tick_step)[1:], np.arange(0,vmax+1,tick_step)[1:])
    
    #Configure x-axis:
    if label_x:
        label = '5-Year period' if five_year_data else 'Year'
        label += ' when move was reported'
        _ = ax.set_xlabel(label, fontsize=label_size, labelpad=10)
    _ = ax.set_xlim(left=x_vals[0]-x_offset, right=x_vals[-1]+x_offset)
    if five_year_data: _ = ax.set_xticks(x_vals, [f'{year-4}-{str(year)[-2:]}' for year in x_vals], rotation=0)
    
    #Include a vertical bar for the fire date:
    if fire_year is not None:
        _ = ax.axvline(fire_year, linestyle='--', color='k', zorder=-1)
        _ = ax.text(fire_year - 0.1, 0.95*vmax,
                    f'{fire_name} fire' if fire_name is not None else 'Fire',
                    fontsize=label_size, va='top', ha='right')
    
    #Legend:
    if show_legend:
        #Separate control and treatment labels:
        handles = ax.get_lines()
        treat_handles, treat_labels = [handles[0]], [treatment_label]
        contr_handles, contr_labels =  handles[1:], [group for group in indices_per_group.keys() if group != treatment_label]
        #Create a two-column legend, padding treatment:
        if two_column_legend:
            handles = treat_handles + (N_groups-1)*[plt.Line2D([],[], color='none')] + contr_handles
            labels  = treat_labels  + (N_groups-1)*['']                              + contr_labels
            leg = ax.legend(handles, labels,
                            loc='upper center', bbox_to_anchor=(0.5, -0.2 if not label_x else -0.5), ncols=2,
                            frameon=False, fontsize=label_size)
        else:
            handles = treat_handles + contr_handles
            labels  = treat_labels  + contr_labels
            leg = ax.legend(handles, labels,
                            loc='upper center', bbox_to_anchor=(0.5, -0.2 if not label_x else -0.5), ncols=1,
                            frameon=False, fontsize=label_size)
            
        for lh in leg.legend_handles: lh.set_alpha(1)

    #Print summary:
    if verbose:
        print(f'\nThe year following {fire_name} fire ({fire_year+1}), {treatment_label} had an out-migration rate:')
        
        fire_idx = x_vals.index(fire_year+1)
        y_treat = group_values[treatment_label][fire_idx]
        y_treat_previous = group_values[treatment_label][fire_idx-1]
        print(f'...of {y_treat:.1f} per 1,000 people')
        print(f'...about {y_treat/y_treat_previous-1:.2f}x that of {fire_year}')
        for group, control_values in group_values.items():
            if group != treatment_label:
                control_value = control_values[fire_idx]
                print(f'...about {y_treat/control_value:.2f}x that of {group}')        
    
    return ax

def correlation_pairlot(value_list,
                        value_names=None,
                        plot_correlations=True,
                        correlation_fontsize=12,
                        label_fontsize=30,
                        alpha=0.5):

    #From values and their names, do a DataFrame:
    plot_df = pd.DataFrame(data=value_list, index=value_names).T

    #Plot:
    sns.set_context('paper', rc={'axes.labelsize':label_fontsize})
    graph = sns.pairplot(plot_df, corner=True, plot_kws={'alpha':alpha})

    #Apply correlations:
    if plot_correlations:
        
        def corrfunc(x, y, ax=None, **kwargs):
            r, _ = spearmanr(x, y, nan_policy='omit')
            ax = ax or plt.gca()
            ax.annotate(fr'$\rho$ = {r:.2f}', xy=(.6, .1),
                        size=correlation_fontsize,
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=.5'),
                        xycoords=ax.transAxes)
            
        _ = graph.map_lower(corrfunc)
                            
    return graph