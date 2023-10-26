from tracemalloc import start
import matplotlib.pyplot as plt
# from datetime import datetime
from numpy import linspace
from copy import deepcopy
import pandas as pd
import numpy as np
import warnings
import sys
import os
import re

try:
    import eyetracking.functions.extract as extract
    import eyetracking.functions.analyze as analyze
    import eyetracking.functions.annotation as annotation
    import eyetracking.functions.plot as fplot
except:
    import functions.extract as extract
    import functions.analyze as analyze
    import functions.annotation as annotation
    import functions.plot as fplot


def main():
    
    # path_data              = '/Users/trevor/Dropbox/Mac (2)/Documents/datasets/eyelink_test'
    # path_data              = '/home/trevor-debian/Documents/datasets/data_eyetracking'
    # path_data              = '/Users/remus/Documents/jhu/PD_project/all_data/eyelink_test'

    path_data              = '/Users/remus/Documents/jhu/PD_project/all_data/S'
    path_output            = '/Users/remus/Documents/jhu/PD_project/all_data/S_output'

    # path_output            = '/Users/trevor/Dropbox/Mac (2)/Documents/outputs/eyelink_test'
    # path_output            = '/home/trevor-debian/Documents/outputs/eyetracking_output'
    # path_output            = '/Users/remus/Documents/jhu/PD_project/all_data/visual/smooth_pursuit_test'

    notes_sheet            = 'tdm'
    hdf_key_dir            = 'tdm'
    hdf_key_note           = ''

    reextract              = False
    reprocess              = True
    trim_trial             = False
    save_processed         = False
    save_csv               = False
    add_annotation         = False
    save_plots             = False
    show_plots             = False

    trials = {  # 'stroop':            [    'Word_Color_long',   'Word_Color_long_END',                  'WordColor'],
        # 'stroop_onlyText':   ['Colors_preliminary1','Colors_preliminaryEnd1', 'Secuence_stroop_Previous_1'],
        # 'stroop_onlyColors': ['Colors_preliminary2','Colors_preliminaryEnd2', 'Secuence_stroop_Previous_2'],
        # 'cookieThief':       ['Exploration_Cookie', 'Exploration_CookieEnd',                 'CookieThief']
        'ReadRainbow1':    ['ReadRainbow1', 'ReadRainbowEnd1', 'NA'],
        'ReadRainbow2':    ['ReadRainbow2', 'ReadRainbowEnd2', 'NA'],
        # 'smoothPursuit1':      ['SmoothPur_1', 'SmoothPur_End1', 'NA'],
        # 'smoothPursuit2':      ['SmoothPur_2', 'SmoothPur_End2', 'NA'],
        # 'smoothPursuit3':      ['SmoothPur_3', 'SmoothPur_End3', 'NA'],
        # 'smoothPursuit4':      ['SmoothPur_4', 'SmoothPur_End4', 'NA'],
        # 'smoothPursuit5':      ['SmoothPur_5', 'SmoothPur_End5', 'NA'],
        # 'smoothPursuit6':      ['SmoothPur_6', 'SmoothPur_End6', 'NA'],
        # 'smoothPursuit7':      ['SmoothPur_7', 'SmoothPur_End7', 'NA'],

        # 'ProsacVigorNaming1': ['Prosaccades_1', 'Prosaccadesend_1', 'VC_89'],  # Prosac_Vigor_1.wav
        # 'ProsacVigorNaming2': ['Prosaccades_2', 'Prosaccadesend_2', 'VC_90'],
        # 'ProsacVigorNaming3': ['Prosaccades_3', 'Prosaccadesend_3', 'VC_91'],
        # 'ProsacVigorNaming4': ['Prosaccades_4', 'Prosaccadesend_4', 'VC_92'],
        # 'ProsacVigorNaming5': ['Prosaccades_5', 'Prosaccadesend_5', 'VC_93'],
        # 'ProsacVigorNaming6': ['Prosaccades_6', 'Prosaccadesend_6', 'VC_94'],
        # 'ProsacVigorNaming7': ['Prosaccades_7', 'Prosaccadesend_7', 'VC_95'],
        # 'ProsacVigorNaming8': ['Prosaccades_8', 'Prosaccadesend_8', 'VC_96'],

        # 'ProsacVigor1_1': ['Prosaccades_1', 'Prosaccadesend_1', 'VC_98'],
        # 'ProsacVigor1_2': ['Prosaccades_2', 'Prosaccadesend_2', 'VC_99'],
        # 'ProsacVigor1_3': ['Prosaccades_3', 'Prosaccadesend_3', 'VC_100'],
        # 'ProsacVigor1_4': ['Prosaccades_4', 'Prosaccadesend_4', 'VC_101'],
        # 'ProsacVigor1_5': ['Prosaccades_5', 'Prosaccadesend_5', 'VC_102'],
        # 'ProsacVigor1_6': ['Prosaccades_6', 'Prosaccadesend_6', 'VC_103'],
        # 'ProsacVigor1_7': ['Prosaccades_7', 'Prosaccadesend_7', 'VC_104'],
        # 'ProsacVigor1_8': ['Prosaccades_8', 'Prosaccadesend_8', 'VC_105'],
        #
        # 'ProsacVigor2_1': ['Prosaccades_1', 'Prosaccadesend_1', 'VC_106'],
        # 'ProsacVigor2_2': ['Prosaccades_2', 'Prosaccadesend_2', 'VC_107'],
        # 'ProsacVigor2_3': ['Prosaccades_3', 'Prosaccadesend_3', 'VC_108'],
        # 'ProsacVigor2_4': ['Prosaccades_4', 'Prosaccadesend_4', 'VC_109'],
        # 'ProsacVigor2_5': ['Prosaccades_5', 'Prosaccadesend_5', 'VC_110'],
        # 'ProsacVigor2_6': ['Prosaccades_6', 'Prosaccadesend_6', 'VC_111'],
        # 'ProsacVigor2_7': ['Prosaccades_7', 'Prosaccadesend_7', 'VC_112'],
        # 'ProsacVigor2_8': ['Prosaccades_8', 'Prosaccadesend_8', 'VC_113'],
        #
        # 'ProsacVigor3_1': ['Prosaccades_1', 'Prosaccadesend_1', 'VC_114'],
        # 'ProsacVigor3_2': ['Prosaccades_2', 'Prosaccadesend_2', 'VC_115'],
        # 'ProsacVigor3_3': ['Prosaccades_3', 'Prosaccadesend_3', 'VC_116'],
        # 'ProsacVigor3_4': ['Prosaccades_4', 'Prosaccadesend_4', 'VC_117'],
        # 'ProsacVigor3_5': ['Prosaccades_5', 'Prosaccadesend_5', 'VC_118'],
        # 'ProsacVigor3_6': ['Prosaccades_6', 'Prosaccadesend_6', 'VC_119'],
        # 'ProsacVigor3_7': ['Prosaccades_7', 'Prosaccadesend_7', 'VC_120'],
        # 'ProsacVigor3_8': ['Prosaccades_8', 'Prosaccadesend_8', 'VC_121'],
        #
        # 'Antisacca_Horiz_vigor1': ['Antisacc_Horiz_1', 'Antisacc_Horizend_1', 'VC_122'],
        # 'Antisacca_Horiz_vigor2': ['Antisacc_Horiz_2', 'Antisacc_Horizend_2', 'VC_123'],
        # 'Antisacca_Horiz_vigor3': ['Antisacc_Horiz_3', 'Antisacc_Horizend_3', 'VC_124'],
        # 'Antisacca_Horiz_vigor4': ['Antisacc_Horiz_4', 'Antisacc_Horizend_4', 'VC_125'],
        # 'Antisacca_Vert_vigor5': ['Antisacc_Vert_5', 'Antisacc_Vert_END5', 'VC_126'],
        # 'Antisacca_Vert_vigor6': ['Antisacc_Vert_6', 'Antisacc_Vert_END6', 'VC_127'],
        # 'Antisacca_Vert_vigor7': ['Antisacc_Vert_7', 'Antisacc_Vert_END7', 'VC_128'],
        # 'Antisacca_Vert_vigor8': ['Antisacc_Vert_8', 'Antisacc_Vert_END8', 'VC_129'],
        # 'Prosaccade1_h': ['Prosaccades_1', 'Prosaccadesend_1', 'VC_3'],
        # 'Prosaccade2_h': ['Prosaccades_2', 'Prosaccadesend_2', 'VC_4'],
        # 'Prosaccade3_v': ['Prosaccades_3', 'Prosaccadesend_3', 'VC_5'],
        # 'Prosaccade4_h': ['Prosaccades_4', 'Prosaccadesend_4', 'VC_6'],
        # 'Prosaccade5_v': ['Prosaccades_5', 'Prosaccadesend_5', 'VC_7'],
        # 'Prosaccade6_h': ['Prosaccades_6', 'Prosaccadesend_6', 'VC_8'],
        # 'Prosaccade7_h': ['Prosaccades_7', 'Prosaccadesend_7', 'VC_9'],
        # 'Prosaccade8_v': ['Prosaccades_8', 'Prosaccadesend_8', 'VC_10'],
        # 'Prosaccade9_h': ['Prosaccades_9', 'Prosaccadesend_9', 'VC_11'],
        # 'Prosaccade10_v': ['Prosaccades_10', 'Prosaccadesend_10', 'VC_12'],
        # 'Prosaccade11_v': ['Prosaccades_11', 'Prosaccadesend_11', 'VC_13'],
        # 'Prosaccade12_v': ['Prosaccades_12', 'Prosaccadesend_12', 'VC_14'],
        # 'Prosaccade13_h': ['Prosaccades_13', 'Prosaccadesend_13', 'VC_15'],
        # 'Prosaccade14_v': ['Prosaccades_14', 'Prosaccadesend_14', 'VC_16'],
        # 'Prosaccade15_v': ['Prosaccades_15', 'Prosaccadesend_15', 'VC_17'],
        # 'Prosaccade16_h': ['Prosaccades_16', 'Prosaccadesend_16', 'VC_18'],
        # 'Antisacca_Horiz1': ['Antisacc_Horiz_1', 'Antisacc_Horizend_1', 'VC_21'],
        # 'Antisacca_Horiz2': ['Antisacc_Horiz_2', 'Antisacc_Horizend_2', 'VC_22'],
        # 'Antisacca_Horiz3': ['Antisacc_Horiz_3', 'Antisacc_Horizend_3', 'VC_23'],
        # 'Antisacca_Horiz4': ['Antisacc_Horiz_4', 'Antisacc_Horizend_4', 'VC_24'],
        # 'Antisacca_Horiz5': ['Antisacc_Horiz_5', 'Antisacc_Horizend_5', 'VC_25'],
        # 'Antisacca_Horiz6': ['Antisacc_Horiz_6', 'Antisacc_Horizend_6', 'VC_26'],
        # 'Antisacca_Horiz7': ['Antisacc_Horiz_7', 'Antisacc_Horizend_7', 'VC_27'],
        # 'Antisacca_Horiz8': ['Antisacc_Horiz_8', 'Antisacc_Horizend_8', 'VC_28'],
        # 'Antisacca_Horiz9': ['Antisacc_Horiz_9', 'Antisacc_Horizend_9', 'VC_29'],
        # 'Antisacca_Horiz10': ['Antisacc_Horiz_10', 'Antisacc_Horizend_10', 'VC_30'],
        # 'Antisacca_Vert_11': ['Antisacc_Vert_11', 'Antisacc_Vertend_11', 'VC_31'],
        # 'Antisacca_Vert_12': ['Antisacc_Vert_12', 'Antisacc_Vertend_12', 'VC_32'],
        # 'Antisacca_Vert_13': ['Antisacc_Vert_13', 'Antisacc_Vertend_13', 'VC_33'],
        # 'Antisacca_Vert_14': ['Antisacc_Vert_14', 'Antisacc_Vertend_14', 'VC_34'],
        # 'Antisacca_Vert_15': ['Antisacc_Vert_15', 'Antisacc_Vertend_15', 'VC_35'],
        # 'Antisacca_Vert_16': ['Antisacc_Vert_16', 'Antisacc_Vertend_16', 'VC_36'],
        # 'Antisacca_Vert_17': ['Antisacc_Vert_17', 'Antisacc_Vertend_17', 'VC_37'],
        # 'Antisacca_Vert_18': ['Antisacc_Vert_18', 'Antisacc_Vertend_18', 'VC_38'],
        # 'Antisacca_Vert_19': ['Antisacc_Vert_19', 'Antisacc_Vertend_19', 'VC_39'],
        # 'Antisacca_Vert_20': ['Antisacc_Vert_20', 'Antisacc_Vertend_20', 'VC_40'],
    }

    anti = False
    smooth = False
    analysis_constants = {
                        'closest_sac':       5, # ms           default=20
                        'closest_blink':    60, # ms           default=50
                        'closest_pur':      20,
                        'threshold_fixDist':    40, # pixels,      default=20
                        'threshold_fixVel' :    60, # deg/sec,     default=30
                        'threshold_fixAcc' :  6000, # deg/sec/sec, default=3000
                        'threshold_purDist': 50,
                        'threshold_purVel': 5,
                        'threshold_purAcc': 200,
                        'gaze_tolerance_x' :   100, # pixelss
                        'gaze_tolerance_y' :  None # pixels
                        }

    # Initialize final output dataframes
    summary_ALLsac = pd.DataFrame()
    summary_ALLfix = pd.DataFrame()
    summary_ALLblk = pd.DataFrame()
    summary_ALLgaz = pd.DataFrame()
    summary_ALLwrd = pd.DataFrame()
    summary_ALLsub = pd.DataFrame()
    summary_status = pd.DataFrame()
    summary_ALLpupil = pd.DataFrame()
    status_template = {**{'edf':[False],'group':[False]}, **{t:[False] for t in trials.keys()}, **{t+'_mvmt':[False] for t in trials.keys()}, **{t+'_gaze':[False] for t in trials.keys()}, **{t+'_wordAlign':[False] for t in trials.keys()}, **{t+'_wordBegin':[False] for t in trials.keys()}}
    # Create data extraction output folder
    path_output_data = os.path.join( path_output, 'data_processed')
    if not os.path.exists(path_output_data):
        os.mkdir(path_output_data)
    if len(hdf_key_note) > 0:
        hdf_key_suffix = '_' + hdf_key_note
    else:
        hdf_key_suffix = ''

    # Iterate through each experimental group
    for group in sorted( os.listdir(path_data), reverse=False):
        path_group = os.path.join( path_data, group)
        if os.path.isdir(path_group):
            print('\n\n----------\n', group)
            path_out_group = os.path.join(path_output_data, group)
            if not os.path.exists(path_out_group):
                os.mkdir(path_out_group)

            # Iterate through each subject in the experimental group
            for subject in sorted( os.listdir(path_group)):
                if subject != 'NLS_089_ses1':
                    continue
                if subject in ['NLS_089', 'PEC_014', 'AD_006', 'AD_010']:
                    continue
                path_subject = os.path.join( path_group, subject)
                print(path_subject)
                if os.path.isdir(path_subject):
                    path_out_subject = os.path.join( path_out_group, subject)
                    if not os.path.exists(path_out_subject):
                        os.mkdir(path_out_subject)

                    # Initialize output dataframes for this subject
                    summary_saccade   = pd.DataFrame()
                    summary_fixation  = pd.DataFrame()
                    summary_blink     = pd.DataFrame()
                    summary_gaze      = pd.DataFrame()
                    summary_wordBegin = pd.DataFrame()
                    summary_subject   = pd.DataFrame()
                    summary_pupil     = pd.DataFrame()
                    status_subject    = deepcopy( status_template)
                    status_subject['group'] = group

                    # Check if output was alrady extracted, or if it must be (re)processed
                    path_output_processed = os.path.join( path_out_subject, subject+'_info.hdf')
                    hdfstore_dir_exists = False
                    if os.path.isfile(path_output_processed):
                        store = pd.HDFStore(path_output_processed)
                        hdfstore_dir_exists = any( [s.startswith('/'+hdf_key_dir+'/') for s in store.keys()])
                        store.close()
                    if reprocess or not(hdfstore_dir_exists):
                        # Extract subject notes
                        notes_subject = None #extract.get_subjectNotes(path_subject, notes_sheet)
                        # Find all the edf files in the folder
                        session_count = 0
                        for filename in sorted( os.listdir(path_subject)):
                            if filename.lower().endswith('.edf'):
                                print('Subject:\t', '\t\t'.join([subject,group,filename]))
                                session_count += 1
                                # session = filename.split('_')[2]
                                # print('Session: ', session)
                                status_subject['edf'] = [True]

                                # Extract the raw data from the edf file, if necessary
                                path_raw     = os.path.join( path_subject, filename)
                                path_extract = os.path.splitext( path_raw)[0] + '.hdf5'
                                if not( os.path.exists(path_extract)) or reextract:
                                    path_intermediate = os.path.splitext( path_raw)[0] + '.asc'
                                    if not( os.path.exists(path_intermediate)) or reextract:
                                        extract.edf2asc(path_raw, path_intermediate)
                                    extract.asc2hdf(path_intermediate, path_extract)

                                # Extract data from available files
                                notes_file = notes_subject.loc[ notes_subject['filename'] == filename]  if notes_subject is not None else None
                                data_eye_annotation = extract.hdf2df( path_extract, 'eyelink_annotations')
                                data_eye_samples    = extract.hdf2df( path_extract, 'eyelink_samples')
                                # messages_all = data_eye_annotation.loc[ data_eye_annotation.iloc[:,0] == 'MSG']
                                # message_options = messages_all.iloc[:,2].unique()
                                
                                # str_date = '-'.join( data_eye_annotation.loc[1, 3:].dropna())
                                # time_eyeFile = datetime.strptime(str_date, '%b-%d-%H:%M:%S-%Y')
                                # session_date = '{:04d}-{:02d}-{:02d}'.format(time_eyeFile.year, time_eyeFile.month, time_eyeFile.day)

                                # Reference Validation row:
                                #       0        1     2           3    4   5      6     7      8     9     10     11   12      13    14    15          16    17    18
                                # idx  MSG  6917588  !CAL  VALIDATION  HV9  LR   LEFT  POOR  ERROR  3.97  avg.  17.41  max  OFFSET  2.86  deg.  -2.4,120.1  pix.  None
                                try:
                                    msg_validation  = data_eye_annotation[ data_eye_annotation.loc[:,3] == 'VALIDATION']
                                    index_lowAvgErr = msg_validation.loc[:,9].astype(float).idxmin()
                                    lowAvgErr = msg_validation.loc[:, 9].astype(float).min()
                                    eye_lowValError = msg_validation.loc[index_lowAvgErr, 6]
                                except:
                                    print('\t\tno validation...using right')
                                    eye_lowValError = 'NoVal'
                                    lowAvgErr = np.nan
                                
                                # Iterate through all the trials of interest, defined in dictionary above
                                for trial, trial_messages in trials.items():
                                    print('\t', trial)
                                    sys.stdout.flush()

                                    # Find the trial starting and ending timestamp
                                    # start_eye = pd.Series(dtype=np.int64)
                                    # end_eye = pd.Series(dtype=np.int64)
                                    # start_eye_index = None
                                    # end_eye_index = None
                                    #
                                    # start_eye_all   = data_eye_annotation.loc[(data_eye_annotation.iloc[:,2] == trial_messages[0])].iloc[:,1]
                                    # for eye_start_index, start_eye_time in start_eye_all.items():
                                    #     if data_eye_annotation.iloc[eye_start_index+2, 4] == 'DRAW_LIST' and \
                                    #        data_eye_annotation.iloc[eye_start_index+2, 5] == 'graphics/' + trial_messages[2] + '.vcl':
                                    #         start_eye = pd.Series([start_eye_time], index=[eye_start_index])
                                    #         start_eye_index = eye_start_index
                                    #
                                    # end_eye_all     = data_eye_annotation.loc[data_eye_annotation.iloc[:,2] == trial_messages[1]].iloc[:,1]
                                    # for eye_end_index, end_eye_time in end_eye_all.items():
                                    #     if start_eye_index:
                                    #         if eye_end_index > start_eye_index: # the first after the start index
                                    #             end_eye = pd.Series([end_eye_time], index=[eye_end_index])
                                    #             end_eye_index = eye_end_index
                                    #
                                    start_audio = data_eye_annotation.loc[ (data_eye_annotation.iloc[:,4] == 'ARECSTART') & (data_eye_annotation.iloc[:,6] == trial_messages[2]+'.wav')].iloc[:,1]
                                    end_audio   = data_eye_annotation.loc[ (data_eye_annotation.iloc[:,4] == 'ARECSTOP' ) & (data_eye_annotation.iloc[:,6] == trial_messages[2]+'.wav')].iloc[:,1]
                                    #
                                    # target_df = None
                                    # if trial != 'ReadRainbow1' and trial != 'ReadRainbow2' and trial != 'cookieThief':
                                    #     if start_eye_index and end_eye_index:
                                    #         trial_df = data_eye_annotation.iloc[start_eye_index:end_eye_index+1, :]
                                    #         target_df = trial_df.loc[trial_df.iloc[:, 4] == 'TARGET_POS', :]
                                    #         target_df.reset_index(drop=True, inplace=True)
                                    #####

                                    # if len(start_eye) == 1 and len(end_eye) == 1:
                                    #     continue
                                    start_eye = data_eye_annotation.loc[data_eye_annotation.iloc[:, 2] == trial_messages[0]].iloc[:, 1]
                                    start_eye_index = data_eye_annotation.loc[data_eye_annotation.iloc[:, 2] == trial_messages[0]].iloc[:, 1].index
                                    end_eye = data_eye_annotation.loc[data_eye_annotation.iloc[:, 2] == trial_messages[1]].iloc[:, 1]
                                    end_eye_index = data_eye_annotation.loc[data_eye_annotation.iloc[:, 2] == trial_messages[1]].iloc[:, 1].index

                                    # graphics = data_eye_annotation.loc[(data_eye_annotation.iloc[:, 4] == 'DRAW_LIST') & (
                                    #         data_eye_annotation.iloc[:, 5] == 'graphics/' + trial_messages[2] + '.vcl')].iloc[:, 1]

                                    target_df = None
                                    if trial != 'ReadRainbow1' and trial != 'ReadRainbow2' and trial != 'cookieThief' and len(
                                            start_eye_index) >= 1 and len(end_eye_index) >= 1:
                                        trial_df = data_eye_annotation.iloc[start_eye_index[0]:end_eye_index[0] + 1, :]
                                        target_df = trial_df.loc[trial_df.iloc[:, 4] == 'TARGET_POS', :]
                                        target_df.reset_index(drop=True, inplace=True)
                                    # use the first trial if found multiple
                                    if len(start_eye) > 1 and len(end_eye) > 1:
                                        # if len(graphics) == 0: # not found
                                        #     continue
                                        start_eye_copy = start_eye.copy()
                                        start_eye_copy.reset_index(drop=True, inplace=True)
                                        end_eye_copy = end_eye.copy()
                                        end_eye_copy.reset_index(drop=True, inplace=True)
                                        start_eye = pd.Series(start_eye_copy.iloc[0])
                                        end_eye = pd.Series(end_eye_copy.iloc[0])


                                    #  check the length of the trial
                                    # if len(start_eye) == 1 and len(end_eye) == 1:
                                    #     if float(end_eye.iloc[end_eye_index]) - float(start_eye.iloc[start_eye_index]) < 20000:
                                    #         start_eye = start_eye.drop([start_eye_index])
                                    #         end_eye = end_eye.drop([end_eye_index])

                                    # For all the identified trial start/stop indexes (there may be multiple runs of a single trial. Often there is only one.)
                                    for index_trial, (timestamp_start, timestamp_end) in enumerate( zip( start_eye, end_eye)):
                                        # if start_eye.index > 10000:
                                        #     continue
                                        # if float(timestamp_end) - float(timestamp_start) < 20000:
                                        #     continue
                                        # Save a description to document observations with
                                        description_trial = subject + '_' + trial + '-'  + '-' + str(index_trial) #+ session
                                        status_subject[trial] = [True]

                                        # Extract the raw data
                                        df_trial_raw = data_eye_samples[(data_eye_samples['timestamp'] >= float(timestamp_start)) & (data_eye_samples['timestamp'] <= float(timestamp_end))].copy()
                                        target_x_raw = np.zeros(int(timestamp_end) - int(timestamp_start) + 1)
                                        target_x_raw[:] = np.nan
                                        target_y_raw = np.zeros(int(timestamp_end) - int(timestamp_start) + 1)
                                        target_y_raw[:] = np.nan
                                        if target_df is not None:
                                            i = 0
                                            while i < len(target_df)-1:
                                                target_start = int(target_df.iloc[i, 1]) - int(timestamp_start)
                                                target_end = int(target_df.iloc[i + 1, 1]) - int(timestamp_start)
                                                target_x_raw[target_start:target_end] = int(re.search(r'\d+', str(target_df.iloc[i, 6])).group())
                                                target_y_raw[target_start:target_end] = int(re.search(r'\d+', str(target_df.iloc[i, 7])).group())
                                                i += 1
                                            target_x_raw[int(target_df.iloc[i, 1])-int(timestamp_start):] = int(
                                                re.search(r'\d+', str(target_df.iloc[i, 6])).group())
                                            target_y_raw[int(target_df.iloc[i, 1])-int(timestamp_start):] = int(
                                                re.search(r'\d+', str(target_df.iloc[i, 7])).group())

                                            if len(df_trial_raw) > len(target_x_raw):
                                                target_x_raw_fill = np.zeros(len(df_trial_raw))
                                                target_x_raw_fill[:] = np.nan
                                                target_x_raw_fill[:len(target_x_raw)] = target_x_raw
                                                target_y_raw_fill = np.zeros(len(df_trial_raw))
                                                target_y_raw_fill[:] = np.nan
                                                target_y_raw_fill[:len(target_y_raw)] = target_y_raw
                                                df_trial_raw['target_x'] = target_x_raw_fill
                                                df_trial_raw['target_y'] = target_y_raw_fill
                                            elif len(df_trial_raw) < len(target_x_raw):
                                                df_trial_raw['target_x'] = target_x_raw[:len(df_trial_raw)]
                                                df_trial_raw['target_y'] = target_x_raw[:len(df_trial_raw)]
                                            else:
                                                df_trial_raw['target_x'] = target_x_raw
                                                df_trial_raw['target_y'] = target_y_raw

                                        df_trial = df_trial_raw.copy()
                                        timestamp_startAudio = start_audio[ (start_audio > timestamp_start) & (start_audio < timestamp_end)].values
                                        timestamp_endAudio   = end_audio[   (end_audio   > timestamp_start) & (end_audio   < timestamp_end)].values

                                        data_wordBegin = None
                                        data_audio     = None
                                        fs_audio       = None
                                        # Prepare the trial eye data
                                        eye_lowError = eye_lowValError
                                        if subject == 'NLS_089_ses1':
                                            eye_lowError = 'RIGHT'
                                        if eye_lowError == 'LEFT':
                                            eye_lowError = 'left'
                                            df_trial.rename(columns={ 'pos_x_left':'pos_x',  'pos_y_left':'pos_y',  'vel_x_left':'vel_x',  'vel_y_left':'vel_y', 'pupil_left':'pupil'}, inplace=True)

                                        elif eye_lowError == 'RIGHT':
                                            eye_lowError = 'right'
                                            df_trial.rename(columns={'pos_x_right':'pos_x', 'pos_y_right':'pos_y', 'vel_x_right':'vel_x', 'vel_y_right':'vel_y', 'pupil_right':'pupil'}, inplace=True)
                                        else:
                                            print('\t\tno validation...using right')
                                            eye_lowError = 'right'
                                            df_trial.rename(columns={'pos_x_right':'pos_x', 'pos_y_right':'pos_y', 'vel_x_right':'vel_x', 'vel_y_right':'vel_y', 'pupil_right':'pupil'}, inplace=True)
                                        
                                        # Prepare the audio data
                                        if data_wordBegin is not None:
                                            try:
                                                if len(timestamp_startAudio) == 0:
                                                    print('NO AUDIO TIMESTAMPS FOUND FOR ', description_trial)
                                                else:
                                                    data_wordBegin.loc[:,'Time'] = (data_wordBegin.loc[:,'Time'] * 1000) + float(timestamp_startAudio[0])
                                                    data_wordBegin['token_first'] = data_wordBegin.apply(lambda x: x['Token'].lower().replace('▁','')[0], axis=1)
                                                    status_subject[trial+'_wordAlign'] = [True]
                                            except:
                                                print('  Word Token data not found')
                                                data_wordBegin = None
                                                data_audio     = None
                                                fs_audio       = None

                                        # Get any notes relevant to this trial
                                        if notes_file is not None:
                                            notes_trial = notes_file.loc[ notes_file['trial'] == trial].copy()  if notes_file is not None else None
                                            # Convert trial-timestamp (starts at zero for each trial) notes to global raw-data defined timestamps (defined by eyelink raw data)
                                            timestamp_trialStart = df_trial.iloc[0]['timestamp']
                                            for timestamp_update in ['timestamp_start', 'timestamp_end']:
                                                row_update = notes_trial[timestamp_update] < 1000
                                                notes_trial.loc[row_update, timestamp_update] = notes_trial.loc[row_update, timestamp_update] * 1000 + timestamp_trialStart
                                        else:
                                            notes_trial = None
                                        
                                        #############################################
                                        ##        BEGIN DATA ANALYSIS STEPS        ##
                                        #############################################
                                        try:
                                            '''
                                                Analyze: Eye Movement
                                            '''
                                            # Measure General Eye Movement
                                            df_trial, info_saccade, info_fixation, info_blink, info_pupil, info_smooth = analyze.get_eyeMovement(df_trial, analysis_constants, 'timestamp', 'pos_x',  'pos_y',  'vel_x', 'vel_y', col_pupil='pupil', notes=notes_trial, smooth=smooth)
                                            # df_trial, [info_saccade, info_fixation, info_blink] = annotation.remove_eyeMovement( notes_trial, df_trial, info_saccade, info_fixation, info_blink, 'timestamp', 'timestamp_start', 'timestamp_end', desc_suffix=eye_lowError)

                                            if smooth:
                                                feature_pursuit = analyze.get_pursuitMovement(df_trial, 'timestamp', 'pos_x',  'pos_y',  'vel_x', 'vel_y')
                                            '''
                                                Analyze: Saccade Movement
                                            '''
                                            # Measure General Eye Movement
                                            # df_trial, info_delay, delay_timestamps, info_deviation, _= analyze.get_saccadeMovement(df_trial, anti=anti)
                                            # df_trial, [info_saccade, info_fixation, info_blink] = annotation.remove_eyeMovement( notes_trial, df_trial, info_saccade, info_fixation, info_blink, 'timestamp', 'timestamp_start', 'timestamp_end', desc_suffix=eye_lowError)

                                            '''
                                                Analyze: Eye Gaze
                                            '''
                                            # Measure Eye gaze characterisics
                                            # df_trial, info_saccade, info_fixation, info_blink, info_gaze, cluster_fcn, cluster_desc = analyze.get_eyeGazeStimuli(df_trial, info_saccade, info_fixation, info_blink, trial, 'timestamp', 'pos_x',  'pos_y', 'pos_x', 'pos_y', 'duration', col_saccade='saccade', col_fixation='fixation',  col_blink='blink', trim_trial=False, save_trimPlot=False, path_trimPlot=os.path.join( path_subject,'cropTrial',description_trial+'.png'), notes=notes_trial)
                                            # df_trial, info_saccade, info_fixation, info_blink, info_gaze = annotation.update_eyeGazeStimuli( notes_trial, df_trial, info_saccade, info_fixation, info_blink, info_gaze, 'timestamp', 'timestamp_start', 'timestamp_end')
                                            
                                            info_gaze = None
                                            if add_annotation:
                                                print('ADDING ANNOTATION')
                                                notes_subject = annotation.add_annotation( notes_subject, filename, trial, df_trial, 'timestamp', 'pos_x', 'pos_y', 'saccade', 'blink', desc_suffix=eye_lowError)
                                                extract.update_subjectNotes(path_subject, notes_sheet, notes_subject)

                                            '''
                                                Data Check: Decide if data analysis is passable.
                                                            If not, try the other eye
                                            '''
                                            # Decide if we should try the other eye
                                            trial_length = (df_trial.iloc[-1]['timestamp'] - df_trial.iloc[0]['timestamp'])
                                            blink_length = info_blink['duration'].sum()
                                            missing_perc_threshold = 0.2
                                            if (blink_length > (trial_length*missing_perc_threshold)): #or (info_gaze is None):
                                                print('\t\tTrying other eye...')
                                                df_trial_t = df_trial_raw.copy()
                                                # Switch Eyes
                                                if eye_lowError == 'left':
                                                    eye_lowError = 'right'
                                                    df_trial_t.rename(columns={'pos_x_right':'pos_x', 'pos_y_right':'pos_y', 'vel_x_right':'vel_x', 'vel_y_right':'vel_y', 'pupil_right':'pupil'}, inplace=True)
                                                elif eye_lowError == 'right':
                                                    eye_lowError = 'left'
                                                    df_trial_t.rename(columns={ 'pos_x_left':'pos_x',  'pos_y_left':'pos_y',  'vel_x_left':'vel_x',  'vel_y_left':'vel_y', 'pupil_left':'pupil'}, inplace=True)
                                               
                                                # Measure General Eye Movement
                                                df_trial_t, info_saccade_t, info_fixation_t, info_blink_t, info_pupil_t, info_smooth_t = analyze.get_eyeMovement(df_trial_t, analysis_constants, 'timestamp', 'pos_x',  'pos_y',  'vel_x', 'vel_y', col_pupil='pupil', notes=notes_trial, smooth=smooth)
                                                if smooth:
                                                    feature_pursuit_t =  analyze.get_pursuitMovement(df_trial_t, 'timestamp', 'pos_x',  'pos_y',  'vel_x', 'vel_y')
                                                # df_trial_t, info_saccade_t, info_fixation_t, info_blink_t = annotation.remove_eyeMovement( notes_trial, df_trial_t, info_saccade_t, info_fixation_t, info_blink_t, 'timestamp', 'timestamp_start', 'timestamp_end', desc_suffix=eye_lowError)
                                                # df_trial_t, info_delay_t, delay_timestamps_t, info_deviation_t, _ = analyze.get_saccadeMovement(df_trial_t, anti=anti)
                                                # Measure Eye gaze characterisics
                                                # df_trial_t, info_saccade_t, info_fixation_t, info_blink_t, info_gaze_t, cluster_fcn_t, cluster_desc_t = analyze.get_eyeGazeStimuli(df_trial_t, info_saccade_t, info_fixation_t, info_blink_t, trial, 'timestamp', 'pos_x',  'pos_y', 'pos_x', 'pos_y', 'duration', col_saccade='saccade', col_fixation='fixation',  col_blink='blink', trim_trial=trim_trial, save_trimPlot=save_plots, path_trimPlot=os.path.join( path_subject,'trimTrial',description_trial+'.png'), notes=notes_trial)
                                                # df_trial_t, info_saccade, info_fixation, info_blink, info_gaze = annotation.update_eyeGazeStimuli( notes_trial, df_trial_t, info_saccade, info_fixation, info_blink, info_gaze, 'timestamp', 'timestamp_start', 'timestamp_end')

                                                info_gaze_t = None
                                                cluster_fcn_t = None
                                                cluster_desc_t = None
                                                if add_annotation:
                                                    print('ADDING OTHER EYE ANNOTATION')
                                                    notes_subject = annotation.add_annotation( notes_subject, filename, trial, df_trial_t, 'timestamp', 'pos_x', 'pos_y', 'saccade', 'blink', desc_suffix=eye_lowError)
                                                    extract.update_subjectNotes(path_subject, notes_sheet, notes_subject)

                                                # Check if this eye was better. If so, replace all of the data so far.
                                                if ((info_gaze is None) and (info_gaze_t is not None)) or (info_blink_t['duration'].sum() < (blink_length)):
                                                    df_trial      = df_trial_t
                                                    # info_delay = info_delay_t
                                                    # delay_timestamps = delay_timestamps_t
                                                    # info_deviation = info_deviation_t
                                                    info_saccade  = info_saccade_t
                                                    info_fixation = info_fixation_t
                                                    info_blink    = info_blink_t
                                                    info_pupil    = info_pupil_t
                                                    info_gaze     = info_gaze_t
                                                    info_smooth   = info_smooth_t
                                                    if smooth:
                                                        feature_pursuit = feature_pursuit_t
                                                    cluster_fcn   = cluster_fcn_t
                                                    cluster_desc  = cluster_desc_t
                                                else:
                                                    if info_gaze is None:
                                                        print('GAZE ERROR: Could not identify gaze from ', subject, ' - ', description_trial)
                                                    else:
                                                        print('MISSING ERROR: Could not find enough data from ', subject, ' - ', description_trial)
                                            
                                            '''
                                                Analyze: Combine audio characteristics with eye characteristics
                                            '''
                                            info_wordBegin = None
                                            # info_wordBegin  = analyze.get_wordCorrectSequenceInfo( data_wordBegin, trial, 'Time', 'Token', 'token_first')
                                            # if info_gaze is not None:
                                            #     info_wordBegin  = analyze.get_multimodalTiming( info_wordBegin, info_fixation, 'Time', 'word_index', 'gaze_index', 'timestamp_start', 'timestamp_end', 'focus', 'duration')
                                            # valid_wordBegin = info_wordBegin.loc[~info_wordBegin['word_index'].isnull(),:].copy() if (info_wordBegin is not None) else None
                                            #
                                            # if (info_wordBegin is not None) and (len(info_wordBegin[ info_wordBegin['word_index'] > 0]) == 0):
                                            #     print('ERROR: No correct tokens identified for ', description_trial)


                                        except Exception as e:
                                            print('ERROR processing ', description_trial)
                                            print(e)
                                            continue

                                        # Track Data Validity
                                        status_subject[trial+'_mvmt']      = [False] if info_saccade is None else [True]
                                        status_subject[trial+'_gaze']      = [False] if info_gaze is None else [True]
                                        status_subject[trial+'_wordBegin'] = [False] if info_wordBegin is None else [True]

                                        # Compile summary statistics for the trial.
                                        summary = {}
                                        list_df_outputs = []
                                        durationTrial = (df_trial.iloc[-1]['timestamp'] - df_trial.iloc[0]['timestamp']) / 1000
                                        for df_info, desc in zip( [info_saccade, info_fixation, info_blink, info_gaze, info_wordBegin], ['sac', 'fix', 'blk', 'gaz', 'wrd']):
                                            summary.update(analyze.get_summaryStats( df_info, durationTrial, prefix=desc))
                                            if df_info is not None:
                                                df_info['trial']         = trial
                                                df_info['session']       = session_count
                                                df_info['filename']      = filename
                                                df_info['subject']       = subject
                                                df_info['group']         = group
                                                # df_info['date_recorded'] = "{:04d}.{:02d}.{:02d}.{:02d}.{:02d}.{:02d}".format(time_eyeFile.year, time_eyeFile.month, time_eyeFile.day, time_eyeFile.hour, time_eyeFile.minute, time_eyeFile.second)
                                            list_df_outputs.append(df_info)
                                        if smooth:
                                            summary.update(analyze.get_summaryStats(info_smooth, durationTrial, prefix='smooth'))
                                            summary.update(feature_pursuit)
                                        [info_saccade, info_fixation, info_blink, info_gaze, info_wordBegin] = list_df_outputs
                                        
                                        summary.update( analyze.get_trialStats( df_trial, info_saccade, info_fixation, info_blink, info_gaze, 'timestamp', 'focus', 'gaze_line', 'gaze_line_start', 'gaze_line_end', 'gaze_word', 'gaze_word_start', 'gaze_word_end', 'timestamp_start', 'timestamp_end'))
                                        summary['trial']         = trial
                                        summary['session']       = session_count
                                        summary['subject']       = subject
                                        summary['group']         = group
                                        summary['validation_error'] = lowAvgErr

                                        info_pupil['trial'] = trial
                                        info_pupil['trial_index'] = str(index_trial)
                                        info_pupil['subject'] = subject
                                        info_pupil['group'] = group
                                        summary_pupil = pd.concat([summary_pupil, pd.DataFrame(info_pupil)], ignore_index=True)

                                        # summary['date_recorded'] = "{:04d}.{:02d}.{:02d}.{:02d}.{:02d}.{:02d}".format(time_eyeFile.year, time_eyeFile.month, time_eyeFile.day, time_eyeFile.hour, time_eyeFile.minute, time_eyeFile.second)
                                        # Add the subject output to the subject output dataframe
                                        summary_saccade  = summary_saccade.append(  info_saccade)
                                        summary_fixation = summary_fixation.append( info_fixation)
                                        summary_blink    = summary_blink.append(    info_blink)
                                        summary_subject  = summary_subject.append( pd.DataFrame(summary, index=[0]))
                                        if info_gaze is not None:
                                            summary_gaze     = summary_gaze.append(    info_gaze)
                                        if info_wordBegin is not None:
                                            summary_wordBegin = summary_wordBegin.append( info_wordBegin)

                                        # Generate the processing plot output, if necessary
                                        if show_plots or save_plots:
                                            try:
                                                folder_output = 'plot_eyeMovement'
                                                if not os.path.isdir(os.path.join( path_out_subject, folder_output)):
                                                    os.mkdir(os.path.join( path_out_subject, folder_output))
                                                path_output_plots = os.path.join( path_out_subject, folder_output,  description_trial+'.png')
                                                fplot.eyePos_prepPlot(      df_trial, 'timestamp', 'pos_x', 'pos_y')
                                                # fplot.plot_raw(             df_trial, 'timestamp', 'pos_x', 'pos_y', save_plots=save_plots, save_path=path_output_plots, target=False)
                                                fplot.plot_saccades(        df_trial, 'timestamp', 'pos_x', 'pos_y', 'vel', 'acel', 'saccade', 'fixation',         analysis_constants['threshold_fixVel'], analysis_constants['threshold_fixAcc'], save_plots=save_plots, save_path=path_output_plots)
                                                fplot.plot_saccades_blinks( df_trial, 'timestamp', 'pos_x', 'pos_y', 'vel', 'acel', 'saccade', 'blink', 'pursuit', analysis_constants['threshold_fixVel'], analysis_constants['threshold_fixAcc'], horizontal=True, delay_timestamps=None, save_plots=save_plots, save_path=path_output_plots, target=True, smooth=smooth)

                                                if info_gaze is not None:
                                                    folder_output = 'plot_eyeGaze'
                                                    if not os.path.isdir(os.path.join( path_out_subject, folder_output)):
                                                        os.mkdir(os.path.join( path_out_subject, folder_output))
                                                    path_output_plots = os.path.join( path_out_subject, folder_output,  description_trial+'.png')
                                                    fplot.plot_word(            df_trial, 'timestamp', 'pos_x', 'pos_y', 'fixation', 'focus', 'gaze_line', 'gaze_word', save_plots=save_plots, save_path=path_output_plots)
                                                    fplot.plot_fixationStimuli( cluster_fcn, info_fixation, 'pos_x', 'pos_y', 'gaze_word', 'gaze_line', cluster_descriptions=cluster_desc, annotate=False, save_plots=save_plots, save_path=path_output_plots)
                                                    fplot.plot_boundaryStimuli( cluster_fcn, cluster_descriptions=cluster_desc, annotate=True, save_plots=save_plots, save_path=path_output_plots)

                                                if info_wordBegin is not None:
                                                    folder_output = 'plot_wordBegin'
                                                    if not os.path.isdir(os.path.join( path_out_subject, folder_output)):
                                                        os.mkdir(os.path.join( path_out_subject, folder_output))
                                                    path_output_plots = os.path.join( path_out_subject, folder_output,  description_trial+'.png')
                                                    fplot.timestamp_prepPlot( info_fixation, ['timestamp_start', 'timestamp_end'], timestamp_zero=timestamp_start)
                                                    fplot.timestamp_prepPlot( info_blink,    ['timestamp_start', 'timestamp_end'], timestamp_zero=timestamp_start)
                                                    fplot.timestamp_prepPlot( info_saccade,  ['timestamp_start', 'timestamp_end'], timestamp_zero=timestamp_start)
                                                    fplot.timestamp_prepPlot( info_wordBegin,  'Time', timestamp_zero=timestamp_start)
                                                    # fplot.timestamp_prepPlot( valid_wordBegin, 'Time', timestamp_zero=timestamp_start)

                                                    if (data_audio is not None) and (fs_audio is not None):
                                                        time_start_audio = (float(timestamp_startAudio) - float(timestamp_start)) / 1000
                                                        time_audio = linspace(time_start_audio, time_start_audio+(data_audio.shape[0]/fs_audio), data_audio.shape[0])
                                                        fplot.plot_alignedGazeAudio ( info_saccade, info_fixation, info_blink, info_wordBegin, data_audio, time_audio, title=trial, save_plots=save_plots, save_path=path_output_plots)

                                                    fplot.plot_wordTokens_correct( info_wordBegin, 'Time', 'Token', 'word_index', save_plots=save_plots, save_path=path_output_plots)
                                                    fplot.plot_wordCorrect_value( info_saccade, info_fixation, info_blink, valid_wordBegin, 'Time', 'time_lookBeforeWord', col_color='token_first', save_plots=save_plots, save_path=path_output_plots)
                                                    fplot.plot_wordCorrect_value( info_saccade, info_fixation, info_blink, valid_wordBegin, 'Time', 'time_lookAfterWord',  col_color='token_first', save_plots=save_plots, save_path=path_output_plots)
                                                    fplot.plot_wordCorrect_value( info_saccade, info_fixation, info_blink, valid_wordBegin, 'Time', 'time_lookDuration',   col_color='token_first', save_plots=save_plots, save_path=path_output_plots)

                                            
                                            except Exception as e:
                                                print('ERROR plotting ', description_trial)
                                                print(e)
                                                # Close all figures to preserve memory
                                                for fig in plt.get_fignums():
                                                    plt.figure( fig)
                                                    plt.clf()
                                                    plt.close()
                                                continue

                                            if show_plots:
                                                plt.show()
                                            # Close all figures to preserve memory
                                            for fig in plt.get_fignums():
                                                plt.figure( fig)
                                                plt.clf()
                                                plt.close()

                                        if eye_lowError == 'left':
                                            df_trial.rename(columns={'pos_x':'pos_x_left',  'pos_y':'pos_y_left',  'vel_x':'vel_x_left',  'vel_y':'vel_y_left'},  inplace=True)
                                        elif eye_lowError == 'right':
                                            df_trial.rename(columns={'pos_x':'pos_x_right', 'pos_y':'pos_y_right', 'vel_x':'vel_x_right', 'vel_y':'vel_y_right'}, inplace=True)


                        # After extracting all trials from all present files, save the data to .hdf file
                        print('\n\tSaving Results...')

                        if save_processed:
                            warnings.filterwarnings('ignore',category=pd.io.pytables.PerformanceWarning)
                            # If existing extracted data exists, remove it
                            store = pd.HDFStore(path_output_processed)
                            if hdf_key_dir in store.keys():
                                store.remove(hdf_key_dir)
                            store.close()

                            summary_saccade.to_hdf(  path_output_processed, hdf_key_dir+'/saccade'+hdf_key_suffix,  mode='a')
                            summary_fixation.to_hdf( path_output_processed, hdf_key_dir+'/fixation'+hdf_key_suffix, mode='a')
                            summary_blink.to_hdf(    path_output_processed, hdf_key_dir+'/blink'+hdf_key_suffix,    mode='a')
                            summary_subject.to_hdf(  path_output_processed, hdf_key_dir+'/summary'+hdf_key_suffix,  mode='a')
                            if len(summary_gaze) > 0:
                                summary_gaze.to_hdf( path_output_processed, hdf_key_dir+'/gaze'+hdf_key_suffix,     mode='a')
                            if len(summary_wordBegin) > 0:
                                summary_wordBegin.to_hdf( path_output_processed, hdf_key_dir+'/wordBegin'+hdf_key_suffix, mode='a')
                            
                            if save_csv:
                                path_output_processed_csv = os.path.join(path_out_subject, 'output_csv')
                                if not(os.path.exists(path_output_processed_csv)) or not(os.path.isdir(path_output_processed_csv)):
                                    os.mkdir(path_output_processed_csv)
                                summary_saccade.to_csv(  os.path.join(path_output_processed_csv, subject+'_saccade_'  +hdf_key_dir+hdf_key_suffix+'.csv'))
                                summary_fixation.to_csv( os.path.join(path_output_processed_csv, subject+'_fixation_' +hdf_key_dir+hdf_key_suffix+'.csv'))
                                summary_blink.to_csv(    os.path.join(path_output_processed_csv, subject+'_blink_'    +hdf_key_dir+hdf_key_suffix+'.csv'))
                                summary_subject.to_csv(  os.path.join(path_output_processed_csv, subject+'_summary_'  +hdf_key_dir+hdf_key_suffix+'.csv'))
                                if len(summary_gaze) > 0:
                                    summary_gaze.to_csv( os.path.join(path_output_processed_csv, subject+'_gaze_'      +hdf_key_dir+hdf_key_suffix+'.csv'))
                                if len(summary_wordBegin) > 0:
                                    summary_wordBegin.to_csv( os.path.join(path_output_processed_csv, subject+'_wordBegin_'+hdf_key_dir+hdf_key_suffix+'.csv'))

                    # If path_output_processed already exists, simply import the existing data (do not re-process)
                    else:
                        print('Loading ', subject, '...')
                        summary_saccade  = pd.read_hdf( path_output_processed, hdf_key_dir+'/saccade'+hdf_key_suffix)
                        summary_fixation = pd.read_hdf( path_output_processed, hdf_key_dir+'/fixation'+hdf_key_suffix)
                        summary_blink    = pd.read_hdf( path_output_processed, hdf_key_dir+'/blink'+hdf_key_suffix)
                        summary_subject  = pd.read_hdf( path_output_processed, hdf_key_dir+'/summary'+hdf_key_suffix)
                        status_subject['edf']         = [True]
                        status_subject[trial+'_mvmt'] = [True]
                        try:
                            summary_gaze = pd.read_hdf( path_output_processed, hdf_key_dir+'/gaze'+hdf_key_suffix)
                            status_subject[trial+'_gaze'] = [True]
                        except:
                            print('WARNING: No gaze data found for ', subject)
                            summary_gaze = pd.DataFrame()
                        try:
                            summary_wordBegin = pd.read_hdf( path_output_processed, hdf_key_dir+'/wordBegin'+hdf_key_suffix)
                            status_subject[trial+'_wordBegin'] = [True]
                        except:
                            print('WARNING: No gaze data found for ', subject)
                            summary_wordBegin = pd.DataFrame()

                        if 'group' in summary_saccade.columns:
                            summary_saccade.loc[ :,'group'] = group
                            summary_fixation.loc[:,'group'] = group
                            summary_blink.loc[   :,'group'] = group
                            summary_subject.loc[ :,'group'] = group
                            if len(summary_gaze) > 0:
                                summary_gaze.loc[    :,'group'] = group
                            if len(summary_wordBegin) > 0:
                                summary_wordBegin.loc[    :,'group'] = group

                    # Append this subjects data to the overall summary array
                    summary_ALLsac = summary_ALLsac.append(summary_saccade)
                    summary_ALLfix = summary_ALLfix.append(summary_fixation)
                    summary_ALLblk = summary_ALLblk.append(summary_blink)
                    summary_ALLgaz = summary_ALLgaz.append(summary_gaze)
                    summary_ALLwrd = summary_ALLwrd.append(summary_wordBegin)
                    summary_ALLsub = summary_ALLsub.append(summary_subject)
                    summary_status = summary_status.append(pd.DataFrame({**{'subject':subject}, **status_subject}))

                    pupil_stats = {}
                    for col in summary_pupil.columns:
                        if col != 'subject' and col != 'group' and col != 'trial' and col != 'trial_index':
                            pupil_stats[col] = np.mean(np.array(summary_pupil[col]))
                        else:
                            pupil_stats[col] = summary_pupil.loc[0, col]
                    summary_ALLpupil = pd.concat([summary_ALLpupil, pd.DataFrame(pupil_stats, index=[0])], ignore_index=True)

    summary_ALLpupil.to_csv(os.path.join( path_output, 'pupil_summary.csv'))
    path_output_all = os.path.join( path_output, 'data_summary.hdf')
    # summary_ALLsac.to_hdf( path_output_all, hdf_key_dir+'/saccade'+hdf_key_suffix,   mode='a')
    # summary_ALLfix.to_hdf( path_output_all, hdf_key_dir+'/fixation'+hdf_key_suffix,  mode='a')
    # summary_ALLblk.to_hdf( path_output_all, hdf_key_dir+'/blink'+hdf_key_suffix,     mode='a')
    # summary_ALLgaz.to_hdf( path_output_all, hdf_key_dir+'/gaze'+hdf_key_suffix,      mode='a')
    # summary_ALLwrd.to_hdf( path_output_all, hdf_key_dir+'/wordBegin'+hdf_key_suffix, mode='a')
    # summary_ALLsub.to_hdf( path_output_all, hdf_key_dir+'/summary'+hdf_key_suffix,   mode='a')
    # summary_status.replace({True:1,False:0}).to_csv(os.path.join( path_output, 'data_status.csv'))


    def in_metricsToSkip(metric):
        return  ('left' in metric) or ('right' in metric) or ('perc-' in metric) or ('timestamp' in metric) or \
                ('gaze_line' in metric) or ('gaze_word' in metric) or ('gaze_index' in metric) or \
                (('pos' in metric) and (('x' in metric) or ('y' in metric))) or \
                ('line-' in metric.lower()) or ('word-' in metric.lower()) or ('mode' in metric)
    df_hwang = summary_ALLsub.loc[ :, [ not( in_metricsToSkip(c)) for c in summary_ALLsub.columns]]
    df_hwang.to_csv(os.path.join( path_output, 'data_summary.csv'))


if __name__ == '__main__':
    main()
    print('\n\nFin.')
