from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys
import os

sns.set(font_scale=1.5)
sns.set_style('whitegrid')

try:
    import eyetracking.functions.extract   as extract
    import eyetracking.functions.summarize as summarize
except:
    import functions.extract   as extract
    import functions.summarize as summarize

def main():
    path_data              = '/Users/remus/Documents/jhu/PD_project/all_data/updated5/prosac_v'
    # path_data              = '/home/trevor-debian/Documents/datasets/data_eyetracking'
    # path_data              = '/export/b15/tmeyer16/datasets/eyelink'
    # path_output            = '/Users/trevor/Dropbox/Mac (2)/Documents/outputs/eyelink'
    path_output            = '/Users/remus/Documents/jhu/PD_project/all_data/updated5/prosac_v'
    # path_output            = '/home/trevor-debian/Documents/outputs/eyetracking_output'
    # path_output            = '/export/b15/tmeyer16/outputs/eyelink'
    # path_processed         = os.path.join(path_output, 'data_processed')
    path_processed         = '/Users/remus/Documents/jhu/PD_project/all_data/visual/cookie'
    # path_metadata          = os.path.join( path_processed, 'metadata_combined.csv')
    # path_metadata          = './metadata/metadata_combined.csv'

    use_trialContaining  = 'Antisacca_Vert_12'
    subgroups            = ''
    subgroup_thresholds  = None
    # subgroups            = 'H&Y'
    # subgroup_thresholds  = ['2.5','99']
    # subgroup_thresholds  = ['2','99']
    # subgroups            = 'UPDRS3'
    # subgroup_thresholds  = ['25', '99']
    # subgroups            = 'certainty'
    # subgroup_thresholds  = None
    # subgroups              = 'moca'
    # subgroup_thresholds    = ['20', '99']
    subgroup_stratify    = None
    
    recalculate_metricSummary = False
    include_trialDifferences  = False
    calculate_statistics      = False
    plot_summaryBoxplots      = True
    plot_rawBoxplots          = False
    plot_correlations         = False
    export_wordTimes          = False
    show_plots                = False
    save_outputs              = True
    min_statGroupSize         = 5
    pvalue_threshold          = 0.05

    test_suffix            = '_ad-pdALL'
    hdf_key_dir            = 'tdm'
    folder_output_stats    = 'stats'
    folder_output_corr     = 'correlation'
    folder_output_boxplots = 'boxplots'
    folder_output_export   = 'metrics'

    boxplot_order        = ['CTRL', 'AD/MCI', 'PD', 'PDM']
    show_rawPlotsWith    = ['mean', 'std', 'med', 'count', 'time']
    show_corrPlotsWith   = ['moca', 'UPDRS'] # 'Age', 'cdr',  'HY'


    # summary_ALLsac, summary_ALLfix, summary_ALLblk, summary_ALLgaz, summary_ALLwrd, summary_ALLsub = extract.get_processed(path_processed, hdf_key_dir)
    # df_metadata = pd.read_csv(path_metadata).set_index('subject')
    # df_metadata['subject'] = df_metadata.index
    summary_ALLsac, summary_ALLfix, summary_ALLblk, summary_ALLgaz, summary_ALLwrd = None, None, None, None, None
    summary_ALLsub = pd.read_csv(path_data+'/saccade_data_summary.csv')
    # summary_ALLsub = pd.read_csv(path_data + '/summary_selected.csv')
    # summary_ALLsub = pd.read_csv(path_data + '/data_summary.csv')
    # summary_ALLsub = pd.read_csv(path_data + '/ratio_pro.csv')
    # summary_ALLsub['trial'] = 'Prosaccade Ratio'
    # summary_ALLsub = pd.read_csv(path_data + '/pupil_summary.csv', index_col=False)
    # summary_ALLsub.drop(columns=summary_ALLsub.columns[0], axis=1, inplace=True)
    summary_ALLsub.dropna(subset=['trial'], inplace=True)

    meta_df = pd.read_excel('/Users/remus/Documents/jhu/PD_project/all_data/Book3_updated.xlsx', sheet_name='Sheet1')
    meta_df2 = pd.read_excel('/Users/remus/Documents/jhu/PD_project/all_data/LABELS_LAUREANO.xlsx', sheet_name='NeuroLogicalSignalsN-NLSAgesAnd')
    meta_df.rename(columns={"Participant I.D.": "subject", "MoCA Score 1 visit": "moca"}, inplace=True)
    meta_df2.rename(columns={"Record ID": "subject"}, inplace=True)
    meta_df = meta_df.merge(meta_df2, how='left', on='subject')
    meta_df.loc[:, 'moca'] = meta_df['moca'].fillna(meta_df['MOCA_VISIT1'])



    # meta_df = pd.read_csv('/Users/remus/Documents/jhu/PD_project/all_data/0.metadata.csv')
    # summary_ALLsub = summary_ALLsub[:-5]

    summary_ALLsub.loc[summary_ALLsub['group'] == 'AD', 'group'] = 'AD/MCI'
    # summary_ALLsub.loc[summary_ALLsub['group'] == 'CTL', 'group'] = 'CTRL'
    summary_ALLsub = summary_ALLsub[ summary_ALLsub['subject'].str.contains("PEC_005") == False ]
    summary_ALLsub = summary_ALLsub[ summary_ALLsub['subject'].str.contains("NLS_123") == False ]
    summary_ALLsub = summary_ALLsub[ summary_ALLsub['subject'].str.contains("NLS_057") == False]
    # summary_ALLsub = summary_ALLsub[ summary_ALLsub['subject'].str.contains("AD_021") == False ]
    # summary_ALLsub = summary_ALLsub[ summary_ALLsub['subject'].str.contains("AD_007") == False ]
    # summary_ALLsub = summary_ALLsub[ summary_ALLsub['subject'].str.contains("AD_001") == False ]
    # summary_ALLsub = summary_ALLsub[summary_ALLsub['subject'].str.contains("AD_014") == False]


    summary_ALLsub = summary_ALLsub[ summary_ALLsub['subject'].str.contains("NLS_129") == False ]

    # summary_ALLsub.drop_duplicates(subset="subject", keep="last", inplace=True)
    summary_ALLsub.reset_index(inplace=True, drop=True)

    if recalculate_metricSummary:
        summary_ALLsub = get_newSummary( summary_ALLsac, summary_ALLfix, summary_ALLblk, summary_ALLgaz, summary_ALLwrd)

    # Incorporate metadata into summary_ALLsub
    # for mdata in df_metadata.columns:
    #     if mdata == 'subject':
    #         continue
    #     summary_ALLsub[mdata] = summary_ALLsub.apply( lambda row: df_metadata.loc[ row['subject'], mdata] if row['subject'] in df_metadata.index else np.nan, axis=1)

    # if len(subgroups) > 0:
    #     folder_output_stats    += '_' + subgroups
    #     folder_output_boxplots += '_' + subgroups
    #     folder_output_export   += '_' + subgroups
    #
    #     summary_ALLsub = make_subgroups(summary_ALLsub, df_metadata, subgroup=subgroups, thresholds=subgroup_thresholds)
    #     if subgroup_stratify is not None:
    #         summary_ALLsub = summary_ALLsub.loc[ summary_ALLsub['group'].str.contains(subgroup_stratify), :]

    trials_ALL = [t for t in summary_ALLsub['trial'].unique()] #if use_trialContaining in t]
    # trials_ALL = ["smoothPursuit4"]
    groups_ALL = summary_ALLsub['group'].unique().tolist()

    def in_metricsToSkip(metric):
        # return not( metric in ['fix_one_count', 'focus_length', 'gaz_distance_sacInternal_one_mean', 'gaz_duration_total_one_mean', 'gaz_num_fixations_one_mean', 'gaz_num_visits_one_mean', 'gaz_pos_variance_one_mean', 'gaz_velocityMax_sacInternalMean_one_mean', 'gaz_velocityAvg_sacInternalMean_one_mean', 'sac_one_count', 'time_until_line_0', 'time_until_line_1', 'time_until_line_2', 'time_until_line_3', 'time_until_line_4', 'time_until_line_5'])
        return  ('left' in metric) or ('right' in metric) or ('perc-' in metric) or ('timestamp' in metric) or \
                ('gaze_line' in metric) or ('gaze_word' in metric) or ('gaze_index' in metric) or \
                ('subject' in metric) or ('group' in metric) or ('trial' in metric) or ('test' in metric) or \
                (('pos' in metric) and (('x' in metric) or ('y' in metric))) or \
                ('line-' in metric.lower()) or ('word-' in metric.lower()) or ('session' in metric.lower()) or ('median_no_reaction' in metric) or \
                ('mode' in metric) or ('fix_num_blinks_one_median' in metric) #or ('no_reaction' in metric)


    # Boxplots for trial summary data
    path_out_stats = os.path.join(path_output, folder_output_stats+test_suffix)
    if not os.path.exists(path_out_stats):
        os.mkdir(path_out_stats)
    path_out_corr = os.path.join(path_output, folder_output_corr+test_suffix)
    if not os.path.exists(path_out_corr):
        os.mkdir(path_out_corr)
    path_out_boxplot = os.path.join(path_output, folder_output_boxplots+test_suffix)
    if not os.path.exists(path_out_boxplot):
        os.mkdir(path_out_boxplot)
    path_out_boxSummary = os.path.join(path_out_boxplot, 'summary')
    if not os.path.exists(path_out_boxSummary):
        os.mkdir(path_out_boxSummary)
    path_out_boxDifferences = os.path.join(path_out_boxplot, 'trialChanges')
    if not os.path.exists(path_out_boxDifferences):
        os.mkdir(path_out_boxDifferences)
    path_out_boxRaw = os.path.join(path_out_boxplot, 'raw')
    if not os.path.exists(path_out_boxRaw):
        os.mkdir(path_out_boxRaw)

    df_pvalues = pd.DataFrame()
    # Analyze all the metrics separately
    for metric in summary_ALLsub.columns:

        if metric == "smooth_distance_one_mean":
            metric = "Smooth pursuit distance [pixels] (\u03BC)"
            summary_ALLsub.rename(columns={"smooth_distance_one_mean": metric}, inplace=True)
        if metric == "sac_distance_one_median":
            metric = "Saccade distance [pixels] (median)"
            summary_ALLsub.rename(columns={"sac_distance_one_median": metric}, inplace=True)
        if metric == "sac_velocity_avg_one_median":
            metric = "Saccade velocity average [deg/s] (median)"
            summary_ALLsub.rename(columns={"sac_velocity_avg_one_median": metric}, inplace=True)
        if metric == "sac_distance_one_stdev":
            metric = "Saccade distance [pixels] (\u03C3)"
            summary_ALLsub.rename(columns={"sac_distance_one_stdev": metric}, inplace=True)
        if metric == "fix_one_count":
            metric = "Fixation count [#]"
            summary_ALLsub.rename(columns={"fix_one_count": metric}, inplace=True)
        if metric == "fix_duration_one_mean":
            metric = "Fixation duration [ms] (\u03BC)"
            summary_ALLsub.rename(columns={"fix_duration_one_mean": metric}, inplace=True)
        if metric == "sac_one_count":
            metric = "Saccade count [#]"
            summary_ALLsub.rename(columns={"sac_one_count": metric}, inplace=True)
        if metric == "sac_duration_one_mean":
            metric = "Saccade duration [ms] (\u03BC)"
            summary_ALLsub.rename(columns={"sac_duration_one_mean": metric}, inplace=True)
        if metric == "fix_duration_one_stdev":
            metric = "Fixation duration [ms] (\u03C3)"
            summary_ALLsub.rename(columns={"fix_duration_one_stdev": metric}, inplace=True)
        if metric == "sac_duration_one_stdev":
            metric = "Saccade duration [ms] (\u03C3)"
            summary_ALLsub.rename(columns={"sac_duration_one_stdev": metric}, inplace=True)
        if metric == "relative_change_mean":
            metric = "Pupil size relative change (\u03BC)"
            summary_ALLsub.rename(columns={"relative_change_mean": metric}, inplace=True)
        if metric == "mean_over_undershoot_mean":
            metric = "Over_undershoot [pixels] (\u03BC)"
            summary_ALLsub.rename(columns={"mean_over_undershoot_mean": metric}, inplace=True)
        if metric == "max_over_undershoot_max":
            metric = "Over_undershoot [pixels] (max)"
            summary_ALLsub.rename(columns={"max_over_undershoot_max": metric}, inplace=True)
        if metric == "mean_adjustment":
            metric = "Saccade count after [#] (\u03BC)"
            summary_ALLsub.rename(columns={"mean_adjustment": metric}, inplace=True)
        if metric == "mean_wrong_direction":
            metric = "Saccades in wrong direction [#] (\u03BC)"
            summary_ALLsub.rename(columns={"mean_wrong_direction": metric}, inplace=True)
        if metric == "max_wrong_direction":
            metric = "Saccades in wrong direction [#] (max)"
            summary_ALLsub.rename(columns={"max_wrong_direction": metric}, inplace=True)
        if metric == "mean_no_reaction":
            metric = "Errors of omission [#] (\u03BC)"
            summary_ALLsub.rename(columns={"mean_no_reaction": metric}, inplace=True)
        if metric == "mean_delay_mean":
            metric = "Latency [ms] (\u03BC)"
            summary_ALLsub.rename(columns={"mean_delay_mean": metric}, inplace=True)
        if metric == "gain_mean":
            metric = "Pursuit gain [%] (\u03BC)"
            summary_ALLsub.rename(columns={"gain_mean": metric}, inplace=True)
        if metric == "difference_mean":
            metric = "Difference [pixels] (\u03BC)"
            summary_ALLsub.rename(columns={"difference_mean": metric}, inplace=True)
        if metric == "difference_max":
            metric = "Difference [pixels] (max)"
            summary_ALLsub.rename(columns={"difference_max": metric}, inplace=True)
        if metric == "max_delay_max":
            metric = "Latency [ms] (max)"
            summary_ALLsub.rename(columns={"max_delay_max": metric}, inplace=True)
        if metric == "std_delay_time_mean":
            metric = "Latency [ms] (\u03C3)"
            summary_ALLsub.rename(columns={"std_delay_time_mean": metric}, inplace=True)
        if metric == "std_deviation_other_std":
            metric = "Offset in the non-changing axis [pixels] (\u03C3)"
            summary_ALLsub.rename(columns={"std_deviation_other_std": metric}, inplace=True)
        if metric == "std_deviation_other_mean":
            metric = "Mean offset in the non-changing axis [pixels] (\u03C3)"
            summary_ALLsub.rename(columns={"std_deviation_other_mean": metric}, inplace=True)
        if metric == "average_deviation_other_std":
            metric = "Std offset in the non-changing axis [pixels] (\u03BC)"
            summary_ALLsub.rename(columns={"average_deviation_other_std": metric}, inplace=True)
        if metric == "mean_deviation_other_mean":
            metric = "Square-wave jerks [pixels] (\u03BC)"
            summary_ALLsub.rename(columns={"mean_deviation_other_mean": metric}, inplace=True)
        if metric == "ratio_mean_delay_mean":
            metric = "Latency ratio [%] (\u03BC)"
            summary_ALLsub.rename(columns={"ratio_mean_delay_mean": metric}, inplace=True)
        if metric == "sac_velocity_avg_one_mean":
            metric = "Saccade velocity [deg/s] (\u03BC)"
            summary_ALLsub.rename(columns={"sac_velocity_avg_one_mean": metric}, inplace=True)

        if in_metricsToSkip(metric):
            continue

        output_text  = []

        if True:
        # try:
            if pd.api.types.is_numeric_dtype( summary_ALLsub[metric]):
                print('Analyzing ', metric)
                output_text.append(metric)

                for trial in trials_ALL:
                    output_text.append('\t' + str(trial))
                    summary_header = False
                    summary_ALLsub_trial = summary_ALLsub.loc[ summary_ALLsub['trial'] == trial]
                    # Compare all the trials to one another
                    for i in range(len(groups_ALL)):
                        group_i = summary_ALLsub_trial.loc[ summary_ALLsub_trial['group'] == groups_ALL[i]]
                        if plot_correlations and (len(group_i) >= min_statGroupSize):
                            group_i = group_i.merge(meta_df, how='inner', on='subject')
                            if trial == "Prosaccade_horizontal" or trial == "prosac_h":
                                trial = "Prosaccade Horizontal"
                            if trial == "Prosaccade_vertical" or trial == "prosac_v":
                                trial = "Prosaccade Vertical"
                            if trial == "Antisaccade_horizontal" or trial == "antisac_h":
                                trial = "Antisaccade Horizontal"
                            if trial == "Antisaccade_vertical" or trial == "antisac_v":
                                trial = "Antisaccade Vertical"
                            summarize.get_correlations( group_i, metric, show_corrPlotsWith, trial + ' ' + groups_ALL[i], path_output=path_out_corr, save_outputs=save_outputs, show_plots=show_plots)

                        group_i = group_i.loc[:, metric].dropna().to_numpy()
                        if calculate_statistics:
                            for j in range(i+1, len(groups_ALL)):
                                group_j = summary_ALLsub_trial.loc[ summary_ALLsub_trial['group'] == groups_ALL[j], metric].dropna().to_numpy()

                                if (len(group_i) >= min_statGroupSize) and (len(group_j) >= min_statGroupSize):
                                    group_text  = str(groups_ALL[i]).upper() + '(' + str(len(group_i)) + ') vs ' + str(groups_ALL[j]).upper()+ '(' + str(len(group_j)) + ')'
                                    group_text += ''.join([' '] * (34 - len(group_text)))

                                    if not summary_header:
                                        print('\t' + str(trial))
                                        summary_header = True

                                    pvalues, output_line = summarize.get_statistics( group_i, group_j, group_text=group_text, pvalue_threshold=pvalue_threshold)
                                    output_text.append(output_line)
                                    if (pvalues['ranksum'] < pvalue_threshold) and (pvalues['kruskal'] < pvalue_threshold):
                                        pvalues.update( {'metric': metric, 'trial': trial, 'groups': group_text})
                                        df_pvalues = df_pvalues.append( pd.DataFrame( pvalues, index=[0]))


                if plot_summaryBoxplots:
                    if metric== "Square-wave jerks [pixels] (\u03BC)":
                        for trial in trials_ALL:
                            summary_ALLsub_ALLtrials = summary_ALLsub[ summary_ALLsub['trial'].str.contains(trial)].copy()
                            summarize.get_boxplots(summary_ALLsub_ALLtrials, metric, trial=trial, order=boxplot_order, path_output=path_out_boxSummary, save_outputs=save_outputs, show_plots=show_plots)


                if include_trialDifferences:
                    output_text.append('\n\nTRIAL DIFFERENCES\n')
                    data_plotBoxplot = pd.DataFrame()

                    for a in range(len(trials_ALL)):
                        for b in range(a+1, len(trials_ALL)):
                            trial_difference = str(trials_ALL[a]) + '_MINUS_' + str(trials_ALL[b])

                            summary_ALLsub_trial_a = summary_ALLsub.loc[ summary_ALLsub['trial'] == trials_ALL[a]]
                            summary_ALLsub_trial_b = summary_ALLsub.loc[ summary_ALLsub['trial'] == trials_ALL[b]]
                            # Only keep subjects which have both trials
                            subjects_keep = summary_ALLsub_trial_a.loc[ summary_ALLsub_trial_a['subject'].isin(summary_ALLsub_trial_b['subject']), 'subject']
                            summary_ALLsub_trial_a = summary_ALLsub_trial_a[ summary_ALLsub_trial_a['subject'].isin(subjects_keep)]
                            summary_ALLsub_trial_b = summary_ALLsub_trial_b[ summary_ALLsub_trial_b['subject'].isin(subjects_keep)]

                            summary_header = False
                            output_text.append('\t' + trial_difference)
                            for i in range(len(groups_ALL)):
                                group_i_a = summary_ALLsub_trial_a.loc[ summary_ALLsub_trial_a['group'] == groups_ALL[i]].drop_duplicates(subset='subject', keep='first')
                                group_i_b = summary_ALLsub_trial_b.loc[ summary_ALLsub_trial_b['group'] == groups_ALL[i]].drop_duplicates(subset='subject', keep='first')
                                group_i = (group_i_a[metric] - group_i_b[metric])

                                # if plot_correlations and (len(group_i) >= min_statGroupSize):
                                #     get_correlations( group_i, metric, show_corrPlotsWith, trial_difference + ' ' + groups_ALL[i], path_out=path_out_corr, save_outputs=save_outputs, show_plots=show_plots)

                                df_temp = group_i_a.loc[:,['subject','group']].copy()
                                df_temp['trial'] = trial_difference
                                df_temp[metric] = group_i
                                data_plotBoxplot = data_plotBoxplot.append(df_temp)

                                group_i = group_i.dropna().to_numpy()

                                if len(group_i) >= min_statGroupSize:
                                    for j in range(i+1, len(groups_ALL)):
                                        group_j_a = summary_ALLsub_trial_a.loc[ summary_ALLsub_trial_a['group'] == groups_ALL[j]]
                                        group_j_b = summary_ALLsub_trial_b.loc[ summary_ALLsub_trial_b['group'] == groups_ALL[j]]
                                        group_j = (group_j_a[metric] - group_j_b[metric]).dropna().to_numpy()

                                        if len(group_j) >= min_statGroupSize:
                                            group_text  = str(groups_ALL[i]).upper() + '(' + str(len(group_i)) + ') vs ' + str(groups_ALL[j]).upper()+ '(' + str(len(group_j)) + ')'
                                            group_text += ''.join([' '] * (40 - len(group_text)))

                                            if calculate_statistics:
                                                if not summary_header:
                                                    print('\t' + str(trial_difference))
                                                    summary_header = True
                                                pvalues, output_line = summarize.get_statistics( group_i, group_j, group_text=group_text, pvalue_threshold=pvalue_threshold)
                                                output_text.append( output_line)
                                                if (pvalues['ranksum'] < pvalue_threshold) and (pvalues['kruskal'] < pvalue_threshold):
                                                    pvalues.update( {'metric': metric, 'trial': trial_difference, 'groups': group_text})
                                                    df_pvalues = df_pvalues.append( pd.DataFrame( pvalues, index=[0]))

                    if plot_summaryBoxplots:
                        summarize.get_boxplots(data_plotBoxplot, metric, order=boxplot_order, path_output=path_out_boxDifferences, save_outputs=save_outputs, show_plots=show_plots)

                if calculate_statistics and save_outputs:
                    str_metric = str(metric)
                    str_metric = str_metric.replace('/', '_')
                    with open( os.path.join( path_out_stats, str_metric+'.txt'), 'w') as f:
                        f.writelines([line + '\n' for line in output_text])
    if calculate_statistics:
        _, df_pvalues['ranksum_fdrcorrected'] = fdrcorrection( df_pvalues['ranksum'])
        _, df_pvalues['kruskal_fdrcorrected'] = fdrcorrection( df_pvalues['kruskal'])
        df_pvalues_keep = df_pvalues
        # df_pvalues_keep = df_pvalues[ (df_pvalues['ranksum_fdrcorrected'] < pvalue_threshold) & (df_pvalues['kruskal_fdrcorrected'] < pvalue_threshold)]

        if save_outputs:
            df_pvalues_keep.to_csv( os.path.join( path_out_stats, '!summary.csv'))


    if plot_rawBoxplots:
        for df_info, desc in zip( [summary_ALLsac, summary_ALLfix, summary_ALLblk, summary_ALLgaz, summary_ALLwrd], ['sac', 'fix', 'blk', 'gaz', 'wrd']):
            print(desc,end=', ')
            sys.stdout.flush()

            # Plot each trial separately
            for trial in trials_ALL:
                if use_trialContaining in trial:
                    df_plot = df_info[ df_info['trial'] == trial]

                    # Plot all the metrics separately
                    for metric in df_info.columns:
                        # Do not plot left/right eye, only plot summary including both eyes
                        if in_metricsToSkip(metric):
                            continue
                    
                        try:
                            if pd.api.types.is_numeric_dtype( df_plot[metric]) and any([part in metric for part in show_rawPlotsWith]):
                                # print('plotting')
                                s = sns.catplot(x=metric, y='subject', col='group', data=df_plot, kind='boxen', k_depth='full', linewidth=0, showfliers=False)
                                plt.gcf().set_size_inches(16, 9)
                                plt.suptitle(trial)
                                # plt.tight_layout(pad=3)
                                    
                                if show_plots:
                                    plt.show()
                                if save_outputs:
                                    plt.savefig(os.path.join(path_out_boxRaw, '_'.join((trial, desc, str(metric)+'.png'))), dpi=300)
                                plt.clf()
                                plt.close()

                        except Exception as e:
                            print('ERROR: Unable to create boxplots for ', metric)
                            print('\t', e)
                            continue


    if export_wordTimes:
        
        columns_save = [ 'group', 'subject', 'trial', 'trial_index', 'timestamp_start', 'timestamp_end', 'focus', 'gaze_line', 'gaze_word', 'gaze_text', 'gaze_color']

        path_out_wordTimes = os.path.join(path_output, folder_output_export)
        if not os.path.exists(path_out_wordTimes):
            os.mkdir(path_out_wordTimes)
        path_out_wordTimes_all   = os.path.join( path_out_wordTimes, 'fixations-all_audioAlighned.csv')
        path_out_wordTimes_first = os.path.join( path_out_wordTimes, 'fixations-first_audioAlighned.csv')
        path_out_wordTimes_all  = './fixations-all_audioAligned.csv'
        path_out_wordTimes_first = './fixations-first_audioAligned.csv'
        
        trials_annotation = { 'stroop':            [    'Word_Color_long',   'Word_Color_long_END',                  'WordColor.wav'],
                            'stroop_onlyText':     ['Colors_preliminary1','Colors_preliminaryEnd1', 'Secuence_stroop_Previous_1.wav'], 
                            'stroop_onlyColors':   ['Colors_preliminary2','Colors_preliminaryEnd2', 'Secuence_stroop_Previous_2.wav']
                            # 'cookieThief':       ['Exploration_Cookie', 'Exploration_CookieEnd']
                            }

        df_export_all   = pd.DataFrame()
        df_export_first = pd.DataFrame()

        for group in summary_ALLfix['group'].unique():
            data_group_fixations = summary_ALLfix[ (summary_ALLfix['group'] == group)]

            for subject in data_group_fixations['subject'].unique():
                data_sub_fixations = data_group_fixations[ (data_group_fixations['subject'] == subject)]
                df_export_all_subject   = pd.DataFrame()
                df_export_first_subject = pd.DataFrame()

                for trial, trial_messages in trials_annotation.items():
                    data_trial_fixations = data_sub_fixations[(data_sub_fixations['trial'] == trial)]

                    for filename in data_trial_fixations['filename'].unique():
                        filename_notes      = os.path.splitext( filename)[0] + '.hdf5'
                        data_eye_annotation = extract.hdf2df( os.path.join(path_data, group, subject, filename_notes), 'eyelink_annotations')

                        # Find the trial starting and ending row
                        if 'row_data' in data_eye_annotation.columns:
                            # Extract the raw data rows of interest
                            start_trial = data_eye_annotation.loc[ data_eye_annotation.iloc[:,2] == trial_messages[0], 'row_data']
                            end_trial   = data_eye_annotation.loc[ data_eye_annotation.iloc[:,2] == trial_messages[1], 'row_data']
                            start_audio = data_eye_annotation.loc[ (data_eye_annotation.iloc[:,4] == 'ARECSTART') & (data_eye_annotation.iloc[:,6] == trial_messages[2]), 'row_data']
                            end_audio   = data_eye_annotation.loc[ (data_eye_annotation.iloc[:,4] == 'ARECSTOP' ) & (data_eye_annotation.iloc[:,6] == trial_messages[2]), 'row_data']
                        else:
                            # Extract the timestamps of interest
                            start_trial = data_eye_annotation.loc[ data_eye_annotation.iloc[:,2] == trial_messages[0]].iloc[:,1]
                            end_trial   = data_eye_annotation.loc[ data_eye_annotation.iloc[:,2] == trial_messages[1]].iloc[:,1]
                            start_audio = data_eye_annotation.loc[ (data_eye_annotation.iloc[:,4] == 'ARECSTART') & (data_eye_annotation.iloc[:,6] == trial_messages[2])].iloc[:,1]
                            end_audio   = data_eye_annotation.loc[ (data_eye_annotation.iloc[:,4] == 'ARECSTOP' ) & (data_eye_annotation.iloc[:,6] == trial_messages[2])].iloc[:,1]

                        # For all the identified trial start/stop indexes (there may be multiple runs of a single trial. Often there is only one.)
                        for index_trial, (timestamp_start, timestamp_end) in enumerate( zip( start_trial, end_trial)):
                            description_trial = subject + '_' + trial + '-' + str(index_trial)

                            timestamp_startAudio = start_audio[ (start_audio > timestamp_start) & (start_audio < timestamp_end)].values
                            if len(timestamp_startAudio) == 0:
                                print('NO AUDIO TIMESTAMPS FOUND FOR ', description_trial)
                                continue

                            data_fixations = data_trial_fixations[ (data_trial_fixations['filename'] == filename) & (data_trial_fixations['trial_index'] == index_trial)]

                            data_save_all = data_fixations.loc[:, columns_save]
                            data_save_all.loc[:,'timestamp_start'] = data_save_all['timestamp_start'] - float(timestamp_startAudio[0])
                            data_save_all.loc[:,'timestamp_end'  ] = data_save_all['timestamp_end'  ] - float(timestamp_startAudio[0])

                            data_findFirst = data_save_all[ data_save_all['focus'] == True]
                            index_first = []
                            for line in sorted( data_findFirst['gaze_line'].unique()):
                                if line < 0:
                                    continue
                                for word in sorted( data_findFirst['gaze_word'].unique()):
                                    if word < 0:
                                        continue
                                    index_word = data_findFirst[ (data_findFirst['gaze_line'] == line) & (data_findFirst['gaze_word'] == word)].index
                                    if len(index_word) == 0:
                                        print('ERROR FINDING WORD FOR ', description_trial, '  :  Word (', line, ', ', word, ')')
                                        continue
                                    index_first.append( index_word[0])
                            data_save_first = data_findFirst.loc[ index_first, columns_save]
                            data_save_first = data_save_first.drop('focus', axis=1, inplace=False)

                            df_export_all_subject   = df_export_all_subject.append(   data_save_all)
                            df_export_first_subject = df_export_first_subject.append( data_save_first)

                df_export_all_subject.to_hdf( os.path.join( path_processed, group, subject, subject+'_info.hdf'), key=hdf_key_dir+'/alignAudio_fixations-all', mode='a')
                df_export_first_subject.to_hdf( os.path.join( path_processed, group, subject, subject+'_info.hdf'), key=hdf_key_dir+'/alignAudio_fixations-first', mode='a')
                df_export_all   = df_export_all.append(  df_export_all_subject)
                df_export_first = df_export_first.append(df_export_first_subject)

        df_export_all.to_csv(path_out_wordTimes_all)
        df_export_first.to_csv(path_out_wordTimes_first)



def get_newSummary( summary_ALLsac, summary_ALLfix, summary_ALLblk, summary_ALLgaz, summary_ALLwrd):
    '''Function to generate a new summary dataframe from the raw data output dataframes.

    Args:
        summary_ALLsac (pandas dataframe): Dataframe containting information about individual saccades.
        summary_ALLfix (pandas dataframe): Dataframe containting information about individual fixations.
        summary_ALLblk (pandas dataframe): Dataframe containting information about individual blinks.
        summary_ALLgaz (pandas dataframe): Dataframe containting information about gaze.
        summary_ALLwrd (pandas dataframe): Dataframe containting information about spoken word timings.
    
    Returns:
        Dataframe containing summary data for each subject for each trial.
    '''
    try:
        import eyetracking.functions.analyze as analyze
    except:
        import functions.analyze as analyze

    summary_ALLsub = pd.DataFrame()

    # Compile summary statistics for the trial.
    for subject in summary_ALLsac['subject'].unique():
        summary_subject = pd.DataFrame()
        groups = summary_ALLsac.loc[ summary_ALLsac['subject'] == subject, 'group']
        if len(groups.unique()) == 1:
            group = groups.unique()[0]
        else:
            print('ERROR: MULTIPLE GROUPS DETECTED FOR ', subject, '\n\tSKIPPING SUBJECT')

        for trial in summary_ALLsac['trial'].unique():
            info_saccade   = summary_ALLsac[ (summary_ALLsac['subject'] == subject) & (summary_ALLsac['trial'] == trial)]
            info_fixation  = summary_ALLfix[ (summary_ALLfix['subject'] == subject) & (summary_ALLfix['trial'] == trial)]
            info_blink     = summary_ALLblk[ (summary_ALLblk['subject'] == subject) & (summary_ALLblk['trial'] == trial)]
            info_gaze      = summary_ALLgaz[ (summary_ALLgaz['subject'] == subject) & (summary_ALLgaz['trial'] == trial)]
            info_wordBegin = summary_ALLwrd[ (summary_ALLwrd['subject'] == subject) & (summary_ALLwrd['trial'] == trial)]

            # Get trial duration
            if info_saccade is not None:
                timestamp_firstSaccade = info_saccade['timestamp_start'].min()
                timestamp_lastSaccade = info_saccade['timestamp_end'].max()
            if info_fixation is not None:
                timestamp_firstFixation = info_fixation['timestamp_start'].min()
                timestamp_lastFixation = info_fixation['timestamp_end'].max()
                if info_saccade is None:
                    timestamp_startTrial = timestamp_firstFixation
                    timestamp_stopTrial  = timestamp_lastFixation
                else:
                    timestamp_startTrial = min( timestamp_firstSaccade, timestamp_firstFixation)
                    timestamp_stopTrial = max( timestamp_lastSaccade,  timestamp_lastFixation)
            durationTrial = (timestamp_stopTrial - timestamp_startTrial) / 1000

            summary = {}
            for df_info, desc in zip( [info_saccade, info_fixation, info_blink, info_gaze, info_wordBegin], ['sac', 'fix', 'blk', 'gaz', 'wrd']):
                if len(df_info) > 0:
                    summary.update( analyze.get_summaryStats( df_info, durationTrial=durationTrial, prefix=desc))
 
            summary.update( analyze.get_trialStats( None, info_saccade, info_fixation, info_blink, info_gaze, 'timestamp', 'focus', 'gaze_line', 'gaze_line_start', 'gaze_line_end', 'gaze_word', 'gaze_word_start', 'gaze_word_end', 'timestamp_start', 'timestamp_end'))
            summary['trial']   = trial
            summary['subject'] = subject
            summary['group']   = group

            summary_subject  = summary_subject.append( pd.DataFrame(summary, index=[0]))
        summary_ALLsub = summary_ALLsub.append( summary_subject)
    
    return summary_ALLsub



def make_subgroups( df_data, df_metadata, col_subgroup, thresholds=[30,40,50,60,70,99]):
    '''Create new groups based on a metadata category

    Args:
        df_data (pandas dataframe):     Dataframe containing all data of interest.
        df_metadata (pandas dataframe): Dataframe containing metadata to use when generating new groups.
        col_subgroup (str):             Identifier for the column in df_metadata containing the information used to create new subgroups
        thresholds (list, optional):    List of thresholds to use when separating subroups must be split based on numerical metrics.

    Returns:
        df_data with new 'group' column replaced with new groups.  A column named ['group_og'] is created to maintain the original group assignment.
    '''
    df_data['group_og'] = df_data['group'].copy()

    if not( col_subgroup in df_metadata.columns) and (('proc_'+col_subgroup) in df_metadata.columns):
        col_subgroup = 'proc_'+col_subgroup

    if col_subgroup == 'gender':
        for g, gender in zip([1,2], ['m','f']):
            subjects_group = df_metadata.loc[ df_metadata[col_subgroup] == g, 'subject']
            for sub in subjects_group:
                df_data.loc[ df_data['subject'] == sub, 'group'] = df_data.loc[ df_data['subject'] == sub, 'group_og'] + '_' + gender
    else:
        if thresholds is None:
            subgroup_all = df_metadata[col_subgroup].unique()
            for g in subgroup_all:
                subjects_group = df_metadata.loc[ df_metadata[col_subgroup] == g, 'subject']
                for sub in subjects_group:
                    df_data.loc[ df_data['subject'] == sub, 'group'] = df_data.loc[ df_data['subject'] == sub, 'group_og'] + '_' + str(g)
        elif isinstance( thresholds, str):
            for thresh in thresholds:
                subjects_group = df_metadata.loc[ df_metadata[col_subgroup] == thresh, 'subject']
                for sub in subjects_group:
                    df_data.loc[ df_data['subject'] == sub, 'group'] = df_data.loc[ df_data['subject'] == sub, 'group_og'] + '_' + str(thresh)
        else:
            for thresh in sorted( thresholds, reverse=True):
                subjects_group = df_metadata.loc[ df_metadata[col_subgroup].astype('float') <= float(thresh), 'subject']
                for sub in subjects_group:
                    df_data.loc[ df_data['subject'] == sub, 'group'] = df_data.loc[ df_data['subject'] == sub, 'group_og'] + '_le' + str(thresh)

    return df_data



if __name__ == '__main__':
    main()
    print('\n\nFin.')
