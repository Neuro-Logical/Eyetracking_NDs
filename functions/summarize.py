from scipy.stats import ranksums, kruskal, spearmanr, ttest_ind, pearsonr, kendalltau
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from copy import deepcopy
import seaborn as sns
import numpy as np
import sys
import gc
import os
import pandas as pd

try:
    from eyetracking.statannotations import Annotator
except:
    from ..statannotations import Annotator

def get_statistics( group_i, group_j, group_text='', pvalue_threshold=0.05):
    '''Returns dictionary containing statistics about the statistical independence of two groups.

    Multiple tests are run, including:
        * Kruskal
        * Ranksums
        * T-test

    An ROC_AUC score is also calculated.

    Args:
        group_i (array): Array containing one data metric from one group
        group_j (array): Array containing one data metric from another group
        group_text (str, optional): A descriptor to be added to a line of text summarizing the results.
        pvalue_threshold (float, optional): A threshold which decides whether the line will be print to the screen.
    Returns
        Dictionary summarizing statistical test results.
        String containign a summary of the statisitical tests, along with the mean and standard deviation of each group's data.
    '''
    try:
        result_kruskal, pvalue_kruskal = kruskal(  group_i, group_j)
        result_ranksum, pvalue_ranksum = ranksums( group_i, group_j)
        result_ttest,   pvalue_ttest   = ttest_ind( group_i, group_j)
        roc_auc                        = roc_auc_score( [0] * len(group_i) + [1] * len(group_j), np.concatenate((group_i, group_j), axis=0), average='micro')

        output_line = '\t\t' + group_text + '  :  ranksum p = {:.3f}\tkruskal p = {:.3f}\troc_auc = {:.3f}\tu={:.1f}vs{:.1f}, std={:.2f}vs{:.2f}\t(ttest p={:.3f})'.format( pvalue_ranksum, pvalue_kruskal, roc_auc, np.mean(group_i), np.mean(group_j), np.std(group_i), np.std(group_j), pvalue_ttest)

        if (pvalue_kruskal < pvalue_threshold) and (pvalue_ranksum < pvalue_threshold):
            print(output_line)

    except Exception as e:
        pvalue_kruskal = 0.5
        pvalue_ranksum = 0.5
        pvalue_ttest   = 0.5
        roc_auc        = 0.5
        output_line = '\t\t' + group_text + '  :  ERROR CALCULATING STATS - ' + str(e)

    return {'ranksum': pvalue_ranksum, 'kruskal': pvalue_kruskal, 'roc_auc': roc_auc, 'avg': '{:.1f}vs{:.1f}'.format(group_i.mean(), group_j.mean()), 'std': '{:.2f}vs{:.2f}'.format(group_i.std(), group_j.std())}, output_line



def get_boxplots( df_data, col_data, trial=None, order=None, plot_points=True, path_output='./', save_outputs=True, show_plots=False):
    '''Function to generate and save boxplots, using data from the column of a dataframe.

    Args:
        df_data (pandas dataframe):    Dataframe containing all of the data
        col_data (str):                Identifier for the column of data in the dataframe to use when generating boxplots.
        order (list, optional):        The order in which the groups should appear on the y axis.
        plot_points (bool, optional):  Whether to also plot the raw data points for each trial
        path_output (str, optional):   Path to the output folder to save outputs to.
        save_outputs (bool, optional): Whether to save outputs
        show_plots (bool, optional):   Whether to show the outputs after generation.
    
    Returns:
        None. A figure is generated.
    '''
    print('Plotting...', end='')
    sys.stdout.flush()

    color_palette = sns.color_palette()
    color_palette = [ (0.17254901960784313, 0.6274509803921569, 0.17254901960784313), (0.12156862745098039, 0.4666666666666667, 0.7058823529411765), (1.0, 0.4980392156862745, 0.054901960784313725), (0.8392156862745098, 0.15294117647058825, 0.1568627450980392), (0.5803921568627451, 0.403921568627451, 0.7411764705882353), (0.5490196078431373, 0.33725490196078434, 0.29411764705882354), (0.8901960784313725, 0.4666666666666667, 0.7607843137254902), (0.4980392156862745, 0.4980392156862745, 0.4980392156862745), (0.7372549019607844, 0.7411764705882353, 0.13333333333333333), (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)]

    if order is not None:
        display_order = []
        for g in order:
            groups = df_data.loc[ df_data['group'].str.match(g), 'group'].unique()
            display_order.extend( sorted( list(groups)))
    else:
        display_order = None

    # find all possible pairs
    pairs = [(a, b) for idx, a in enumerate(order) for b in order[idx + 1:]]
    print(pairs)
    plt.figure()
    if trial:
        if trial == "smoothPursuit3":
            trial = "Smooth Pursuit (trial 3)"
        if trial == "ReadRainbow1":
            trial = "Rainbow Passage (reading out loud)"
        if trial == "ReadRainbow2":
            trial = "Rainbow Passage (silent reading)"
        if trial == "smoothPursuit2":
            trial = "Smooth Pursuit (trial 2)"
        if trial == "smoothPursuit1":
            trial = "Smooth Pursuit (trial 1)"
        if trial == "smoothPursuit4":
            trial = "Smooth Pursuit (trial 4)"
        if trial == "smoothPursuit5":
            trial = "Smooth Pursuit (trial 5)"
        if trial == "smoothPursuit6":
            trial = "Smooth Pursuit (trial 6)"
        if trial == "smoothPursuit7":
            trial = "Smooth Pursuit (trial 7)"
        if trial == "cookieThief":
            trial = "Cookie Thief"
        if trial == "Prosaccade_horizontal" or trial == "prosac_h":
            trial = "Prosaccade Horizontal"
        if trial == "Prosaccade_vertical" or trial == "prosac_v":
            trial = "Prosaccade Vertical"
        if trial == "Antisaccade_horizontal" or trial == "antisac_h":
            trial = "Antisaccade Horizontal"
        if trial == "Antisaccade_vertical" or trial == "antisac_v":
            trial = "Antisaccade Vertical"
        if trial == "Antisac_vigor_horizontal":
            trial = "Antisaccade Vigor Horizontal"
        plt.title(trial)
    g = sns.boxenplot(x=col_data, y='group', data=df_data, order=order, k_depth='full', palette=color_palette)
    annot = Annotator.Annotator(g, pairs, data=df_data, x=col_data, y='group', order=order, orient='h', hide_non_significant=True)
    annot.configure(test='Kruskal', text_format='star', loc='inside', verbose=1)
    annot.apply_test().annotate(line_offset_to_group=0.06)
    # annot.apply_and_annotate()
    # g.set_axis_labels(fontsize=10)
    plt.tight_layout()#pad=3)
    #plt.show()
    str_col = str(col_data)
    str_col = str_col.replace('/', '_')
    if save_outputs:
        if not os.path.exists(os.path.join(path_output, trial)):
            os.mkdir(os.path.join(path_output, trial))
        plt.savefig(os.path.join(path_output, trial, trial+str_col+'-box.png'))
        print(str(os.path.join(path_output, trial, trial+str_col+'-box.png')))
    if not show_plots:
        plt.clf()
        plt.close()
        gc.collect()


    if plot_points:
        print('Plotting pts')
        trials_ALL = df_data['trial'].unique()
        num_trials = len(trials_ALL)
        plt.figure(figsize=(15,5))
        for i, t in enumerate( trials_ALL):
            df_data_singleTrial = df_data[ df_data['trial'] == t]
            # boolean = df_data_singleTrial[col_data].duplicated().any()
            plt.subplot(1,num_trials,i+1)
            g = sns.stripplot(x=col_data, y='group', data=df_data_singleTrial, order=order, palette=color_palette, alpha=0.8)
            g.spines['top'].set_visible(False)
            g.spines['right'].set_visible(False)
            plt.title(t)
            if i > 0:
                plt.yticks([],[])
                plt.ylabel('')
        
        if save_outputs:
            plt.savefig(os.path.join(path_output, str_col+'-pts.png'))
        if not show_plots:
            plt.clf()
            plt.close()
            gc.collect()



def get_correlations( df_data, col_metric, metric_compare, description='', normalize=False, pvalue_threshold=0.05, path_output='./', save_outputs=True, show_plots=False):
    '''Function to generate and record correlation characteristics between many metrics

    Args:
        df_data (pandas dataframe):    Pandas dataframe containing all of the data
        col_metric (str):              Identifier of the column in `df_data` containing the data metric of interest
        metric_compare (str or list):  Identifier or list of identifiers of columns in `df_data` to correlate with col_metric.
        description (str, optional):   Descriptor of the data, to be added to saved filenames for easy reference.
        normalize (bool, optional):    Whether to normaliez the data of interest to have mean of 0 and stdev of 1
        pvalue_threshold (float, optional): threshold deciding which decides whether to generate a plot or skip. Set to 1 to plot everything.
        path_output (str, optional):   Path to the output folder to save outputs to.
        save_outputs (bool, optional): Whether to save outputs
        show_plots (bool, optional):   Whether to show the outputs after generation.
        
    '''
    metric_compare_use = deepcopy(metric_compare)
    for m in metric_compare:
        if df_data[m].notna().sum() == 0:
            metric_compare_use.remove(m)

    # g = sns.PairGrid(df_data.reset_index(), hue='group', palette='husl', vars=list(metric_compare_use)+[col_metric])
    # g.map_lower(sns.scatterplot)
    # g.map_diag(sns.histplot)
    # g.map_upper(sns.kdeplot)

    # if save_outputs:
    #     plt.savefig(os.path.join(path_output, description + '_' + str(col_metric) + '_corr.png'))
    # if not show_plots:
    #     plt.clf()
    #     plt.close()
    #     gc.collect()
    # return
    correlation_test = ['Pearson'] # 'Spearman', 'Kendall',
    for metric_comp in metric_compare_use:
        temp_df = pd.DataFrame()
        # Need to redo this every time, since different rows may get dropped
        data_corr = df_data[[col_metric,metric_comp]].replace([np.inf, -np.inf], np.nan).dropna(axis=0)

        np.seterr(divide='ignore', invalid='ignore')

        if len(data_corr) > 2:

            for corr_test in correlation_test:

                if corr_test == 'Spearman':
                    corr, pvalue = spearmanr( data_corr[col_metric], data_corr[metric_comp])
                elif corr_test == 'Kendall':
                    corr, pvalue = kendalltau(data_corr[col_metric], data_corr[metric_comp])
                elif corr_test == 'Pearson':
                    corr, pvalue = pearsonr(data_corr[col_metric], data_corr[metric_comp])

                if pvalue < pvalue_threshold:
                    label = corr_test + ': ' + str(col_metric) + ' VS ' + str(metric_comp) + '\n r={:.2f}'.format(corr)
                    print('\t\t' + corr_test + ' ' + metric_comp + '\tr={:.2f} (p={:.3f})'.format(corr, pvalue))

                    temp_df = pd.DataFrame()
                    if normalize:
                        temp_df[col_metric] = (data_corr[col_metric] - data_corr[col_metric].mean()) / data_corr[col_metric].std()
                        temp_df[metric_comp] = (data_corr[metric_comp] - data_corr[metric_comp].mean()) / data_corr[metric_comp].std()
                    else:
                        temp_df[col_metric] = data_corr[col_metric]
                        temp_df[metric_comp] = data_corr[metric_comp]

                    import warnings
                    from statsmodels.tools.sm_exceptions import ConvergenceWarning
                    warnings.simplefilter('ignore', ConvergenceWarning)

                    sns.lmplot(x=col_metric, y=metric_comp, data=temp_df, height=6.2, aspect=1.5, scatter=True, palette="husl", robust=True)
                    plt.title( description + '\n' + label + ' (p={:.4f}) '.format(pvalue))
                    plt.tight_layout(h_pad=2)

                    path_out_final = os.path.join(path_output, metric_comp)
                    if not( os.path.exists( path_out_final)) or not( os.path.isdir(path_out_final)):
                        os.mkdir(path_out_final)
                    if save_outputs:
                        plt.savefig(os.path.join(path_out_final, col_metric) + '_' + description.replace('/', '-') + '_p={:.3f}.png'.format(pvalue))
                    if not show_plots:
                        plt.clf()
                        plt.close()
                        gc.collect()
        
    #     biomarker   = pd.Series(data=[label]*len(values_1), index=values_1.index)
    #     temp_df     = pd.DataFrame({col_metric: values_1, metric_comp: values_2, 'Biomarker': biomarker})
    #     correlate_final = pd.concat([correlate_final, temp_df])
    
    # plt.figure()
    # sns.lmplot(x=col_metric, y=metric_comp, hue="Biomarker", data=correlate_final, height=6.2, aspect=1.5, scatter=True, palette="husl", robust=True)

    # if save_outputs:
    #     plt.savefig(os.path.join(path_output, 'corr_' + str(col_metric) + '-VS-all.png'))
    # if not show_plots:
    #     plt.clf()
    #     plt.close()
    #     gc.collect()
