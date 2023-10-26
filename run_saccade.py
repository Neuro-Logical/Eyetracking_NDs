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
    import eyetracking.functions.summarize as summarize
except:
    import functions.extract as extract
    import functions.analyze as analyze
    import functions.annotation as annotation
    import functions.plot as fplot
    import functions.summarize as summarize


def main():

    # path_data = '/Users/remus/Documents/jhu/PD_project/all_data/NLS_06'
    # path_output = '/Users/remus/Documents/jhu/PD_project/all_data/NLS_06_test'
    #
    path_data = '/Users/remus/Documents/jhu/PD_project/all_data/archive_eyelink'
    path_output = '/Users/remus/Documents/jhu/PD_project/all_data/test'

    # path_data = '/Users/remus/Documents/jhu/PD_project/all_data/eyelink_test'
    # path_output = '/Users/remus/Documents/jhu/PD_project/all_data/eyelink_test_out'

    notes_sheet = 'tdm'
    hdf_key_dir = 'tdm'
    hdf_key_note = ''

    reextract = False
    reprocess = True
    trim_trial = False
    save_processed = True
    save_csv = True
    add_annotation = False
    save_plots = False
    show_plots = False

    trials = {  # 'stroop':            [    'Word_Color_long',   'Word_Color_long_END',                  'WordColor'],
        # 'stroop_onlyText':   ['Colors_preliminary1','Colors_preliminaryEnd1', 'Secuence_stroop_Previous_1'],
        # 'stroop_onlyColors': ['Colors_preliminary2','Colors_preliminaryEnd2', 'Secuence_stroop_Previous_2'],
        # 'cookieThief':       ['Exploration_Cookie', 'Exploration_CookieEnd',                 'CookieThief']
        # 'ReadRainbow1':    ['ReadRainbow1', 'ReadRainbowEnd1', 'NA'],
        # 'ReadRainbow2':    ['ReadRainbow2', 'ReadRainbowEnd2', 'NA'],
        # 'smoothPursuit1':      ['SmoothPur_1', 'SmoothPur_End1', 'NA'],
        # 'smoothPursuit2':      ['SmoothPur_2', 'SmoothPur_End2', 'NA'],
        # 'smoothPursuit3':      ['SmoothPur_3', 'SmoothPur_End3', 'NA'],
        # 'smoothPursuit4':      ['SmoothPur_4', 'SmoothPur_End4', 'NA'],
        # 'smoothPursuit5':      ['SmoothPur_5', 'SmoothPur_End5', 'NA'],
        # 'smoothPursuit6':      ['SmoothPur_6', 'SmoothPur_End6', 'NA'],
        # 'smoothPursuit7':      ['SmoothPur_7', 'SmoothPur_End7', 'NA']

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
        'Antisacca_Horiz2': ['Antisacc_Horiz_2', 'Antisacc_Horizend_2', 'VC_22'],
        'Antisacca_Horiz3': ['Antisacc_Horiz_3', 'Antisacc_Horizend_3', 'VC_23'],
        'Antisacca_Horiz4': ['Antisacc_Horiz_4', 'Antisacc_Horizend_4', 'VC_24'],
        'Antisacca_Horiz5': ['Antisacc_Horiz_5', 'Antisacc_Horizend_5', 'VC_25'],
        'Antisacca_Horiz6': ['Antisacc_Horiz_6', 'Antisacc_Horizend_6', 'VC_26'],
        'Antisacca_Horiz7': ['Antisacc_Horiz_7', 'Antisacc_Horizend_7', 'VC_27'],
        'Antisacca_Horiz8': ['Antisacc_Horiz_8', 'Antisacc_Horizend_8', 'VC_28'],
        'Antisacca_Horiz9': ['Antisacc_Horiz_9', 'Antisacc_Horizend_9', 'VC_29'],
        'Antisacca_Horiz10': ['Antisacc_Horiz_10', 'Antisacc_Horizend_10', 'VC_30'],
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

    anti = True
    smooth = False
    analysis_constants = {
                        'closest_sac':       5, # ms           default=20
                        'closest_blink':    60, # ms           default=50
                        'closest_pur':      50,
                        'threshold_fixDist':    40, # pixels,      default=20
                        'threshold_fixVel' :    30, # deg/sec,     default=25
                        'threshold_fixAcc' :  6000, # deg/sec/sec, default=3000
                        'threshold_purDist': 20,
                        'threshold_purVel': 5,
                        'threshold_purAcc': 200,
                        'gaze_tolerance_x' :   100, # pixels
                        'gaze_tolerance_y' :  None # pixels
                        }

    # Initialize final output dataframes
    status_template = {**{'edf': [False], 'group': [False]}, **{t: [False] for t in trials.keys()}, **{t + '_mvmt': [False] for t in trials.keys()},
                       **{t + '_gaze': [False] for t in trials.keys()}, **{t + '_wordAlign': [False] for t in trials.keys()},
                       **{t + '_wordBegin': [False] for t in trials.keys()}}
    # Create data extraction output folder
    path_output_data = os.path.join(path_output, 'data_processed')
    if not os.path.exists(path_output_data):
        os.mkdir(path_output_data)
    if len(hdf_key_note) > 0:
        hdf_key_suffix = '_' + hdf_key_note
    else:
        hdf_key_suffix = ''

    all_df = pd.DataFrame()
    not_found_df = []
    # Iterate through each experimental group
    for group in sorted(os.listdir(path_data), reverse=False):
        path_group = os.path.join(path_data, group)
        if os.path.isdir(path_group):
            print('\n\n----------\n', group)
            path_out_group = os.path.join(path_output_data, group)
            if not os.path.exists(path_out_group):
                os.mkdir(path_out_group)
            # Iterate through each subject in the experimental group
            for subject in sorted(os.listdir(path_group)):
                # if subject != 'PEC_011':
                #    continue
                if subject in ['NLS_089', 'PEC_014', 'AD_006', 'AD_010']:
                    continue
                path_subject = os.path.join(path_group, subject)
                if os.path.isdir(path_subject):
                    path_out_subject = os.path.join(path_out_group, subject)
                    if not os.path.exists(path_out_subject):
                        os.mkdir(path_out_subject)

                    # summary_subject = pd.DataFrame()
                    status_subject = deepcopy(status_template)
                    status_subject['group'] = group
                    summary_dict = {}

                    # Check if output was alrady extracted, or if it must be (re)processed
                    path_output_processed = os.path.join(path_out_subject, subject + '_info.hdf')
                    hdfstore_dir_exists = False
                    if os.path.isfile(path_output_processed):
                        store = pd.HDFStore(path_output_processed)
                        hdfstore_dir_exists = any([s.startswith('/' + hdf_key_dir + '/') for s in store.keys()])
                        store.close()
                    if reprocess or not (hdfstore_dir_exists):
                        # Extract subject notes
                        # notes_subject = extract.get_subjectNotes(path_subject, notes_sheet)
                        # Find all the edf files in the folder
                        session_count = 0
                        for filename in sorted(os.listdir(path_subject)):
                            delay_df = pd.DataFrame()
                            deviation_df = pd.DataFrame()
                            shoot_df = pd.DataFrame()
                            gain_df = pd.DataFrame()
                            gain_list = []
                            if filename.lower().endswith('.edf'):
                                session_count += 1
                                print('Subject:\t', '\t\t'.join([subject, group, filename]))
                                # print('Session: ', session)
                                status_subject['edf'] = [True]

                                # Extract the raw data from the edf file, if necessary
                                path_raw = os.path.join(path_subject, filename)
                                path_extract = os.path.splitext(path_raw)[0] + '.hdf5'
                                if not (os.path.exists(path_extract)) or reextract:
                                    path_intermediate = os.path.splitext(path_raw)[0] + '.asc'
                                    if not (os.path.exists(path_intermediate)) or reextract:
                                        extract.edf2asc(path_raw, path_intermediate)
                                    extract.asc2hdf(path_intermediate, path_extract)

                                # Extract data from available files
                                # notes_file = notes_subject.loc[notes_subject['filename'] == filename] if notes_subject is not None else None
                                data_eye_annotation = extract.hdf2df(path_extract, 'eyelink_annotations')
                                data_eye_samples = extract.hdf2df(path_extract, 'eyelink_samples')
                                # messages_all = data_eye_annotation.loc[ data_eye_annotation.iloc[:,0] == 'MSG']
                                # message_options = messages_all.iloc[:,2].unique()

                                # str_date = '-'.join( data_eye_annotation.loc[1, 3:].dropna())
                                # time_eyeFile = datetime.strptime(str_date, '%b-%d-%H:%M:%S-%Y')
                                # session_date = '{:04d}-{:02d}-{:02d}'.format(time_eyeFile.year, time_eyeFile.month, time_eyeFile.day)

                                # Reference Validation row:
                                #       0        1     2           3    4   5      6     7      8     9     10     11   12      13    14    15          16    17    18
                                # idx  MSG  6917588  !CAL  VALIDATION  HV9  LR   LEFT  POOR  ERROR  3.97  avg.  17.41  max  OFFSET  2.86  deg.  -2.4,120.1  pix.  None
                                try:
                                    msg_validation = data_eye_annotation[data_eye_annotation.loc[:, 3] == 'VALIDATION']
                                    index_lowAvgErr = msg_validation.loc[:, 9].astype(float).idxmin()
                                    eye_lowValError = msg_validation.loc[index_lowAvgErr, 6]
                                except:
                                    print('\t\tno validation...using right')
                                    eye_lowValError = 'NoVal'

                                # Iterate through all the trials of interest, defined in dictionary above
                                for trial, trial_messages in trials.items():
                                    print('\t', trial)
                                    sys.stdout.flush()

                                    # Find the trial starting and ending timestamp
                                    start_eye = data_eye_annotation.loc[data_eye_annotation.iloc[:, 2] == trial_messages[0]].iloc[:, 1]
                                    start_eye_index = data_eye_annotation.loc[data_eye_annotation.iloc[:, 2] == trial_messages[0]].iloc[:, 1].index
                                    end_eye = data_eye_annotation.loc[data_eye_annotation.iloc[:, 2] == trial_messages[1]].iloc[:, 1]
                                    end_eye_index = data_eye_annotation.loc[data_eye_annotation.iloc[:, 2] == trial_messages[1]].iloc[:, 1].index

                                    # graphics = data_eye_annotation.loc[(data_eye_annotation.iloc[:, 4] == 'DRAW_LIST') & (
                                    #         data_eye_annotation.iloc[:, 5] == 'graphics/' + trial_messages[2] + '.vcl')].iloc[:, 1]

                                    # if len(graphics) == 0:  # not found
                                    #     print('Not found graphics')
                                    #     not_found_df.append(f'{group}-{subject}-{trial}')
                                    #     continue

                                    target_df = None
                                    if trial != 'ReadRainbow1' and trial != 'ReadRainbow2' and trial != 'cookieThief' and len(
                                            start_eye_index) >= 1 and len(end_eye_index) >= 1:
                                        trial_df = data_eye_annotation.iloc[start_eye_index[0]:end_eye_index[0] + 1, :]
                                        target_df = trial_df.loc[trial_df.iloc[:, 4] == 'TARGET_POS', :]
                                        target_df.reset_index(drop=True, inplace=True)

                                    # if len(start_eye) == 1 and len(end_eye) == 1:
                                    #     continue
                                    # use the first trial if found multiple
                                    start_eye_copy = start_eye.copy()
                                    start_eye_copy.reset_index(drop=True, inplace=True)
                                    end_eye_copy = end_eye.copy()
                                    end_eye_copy.reset_index(drop=True, inplace=True)

                                    if len(start_eye) > 1:
                                        start_eye = pd.Series(start_eye_copy.iloc[0])
                                    if len(end_eye) > 1:
                                        end_eye = pd.Series(end_eye_copy.iloc[0])

                                    # For all the identified trial start/stop indexes (there may be multiple runs of a single trial. Often there is only one.)
                                    for index_trial, (timestamp_start, timestamp_end) in enumerate(zip(start_eye, end_eye)):
                                        if target_df is None:
                                            continue
                                        if start_eye.index > 20000:
                                            continue
                                        # Save a description to document observations with
                                        description_trial = subject + '_' + trial + '-' + '-' + str(index_trial)
                                        status_subject[trial] = [True]

                                        # Extract the raw eye tracker data and target data
                                        df_trial_raw = data_eye_samples[(data_eye_samples['timestamp'] >= float(timestamp_start)) & (
                                                    data_eye_samples['timestamp'] <= float(timestamp_end))].copy()
                                        target_x_raw = np.zeros(int(timestamp_end) - int(timestamp_start) + 1)
                                        target_x_raw[:] = np.nan
                                        target_y_raw = np.zeros(int(timestamp_end) - int(timestamp_start) + 1)
                                        target_y_raw[:] = np.nan
                                        if target_df is not None:
                                            i = 0
                                            while i < len(target_df) - 1:
                                                target_start = int(target_df.iloc[i, 1]) - int(timestamp_start)
                                                target_end = int(target_df.iloc[i + 1, 1]) - int(timestamp_start)
                                                target_x_raw[target_start:target_end] = int(re.search(r'\d+', str(target_df.iloc[i, 6])).group())
                                                target_y_raw[target_start:target_end] = int(re.search(r'\d+', str(target_df.iloc[i, 7])).group())
                                                i += 1
                                            target_x_raw[int(target_df.iloc[i, 1]) - int(timestamp_start):] = int(
                                                re.search(r'\d+', str(target_df.iloc[i, 6])).group())
                                            target_y_raw[int(target_df.iloc[i, 1]) - int(timestamp_start):] = int(
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
                                        # Prepare the trial eye data
                                        eye_lowError = eye_lowValError
                                        if eye_lowError == 'LEFT':
                                            eye_lowError = 'left'
                                            df_trial.rename(
                                                columns={'pos_x_left': 'pos_x', 'pos_y_left': 'pos_y', 'vel_x_left': 'vel_x', 'vel_y_left': 'vel_y'},
                                                inplace=True)
                                        elif eye_lowError == 'RIGHT':
                                            eye_lowError = 'right'
                                            df_trial.rename(columns={'pos_x_right': 'pos_x', 'pos_y_right': 'pos_y', 'vel_x_right': 'vel_x',
                                                                     'vel_y_right': 'vel_y'}, inplace=True)
                                        else:
                                            print('\t\tno validation...using right')
                                            eye_lowError = 'right'
                                            df_trial.rename(columns={'pos_x_right': 'pos_x', 'pos_y_right': 'pos_y', 'vel_x_right': 'vel_x',
                                                                     'vel_y_right': 'vel_y'}, inplace=True)

                                        '''
                                                                                        Analyze: Saccade Movement
                                                                                    '''
                                        # Measure General Eye Movement
                                        df_trial, info_saccade, info_fixation, info_blink, info_pupil, _ = analyze.get_eyeMovement(df_trial, analysis_constants,
                                                                                                                    'timestamp', 'pos_x', 'pos_y',
                                                                                                                    'vel_x', 'vel_y',
                                                                                                                    notes=None, smooth=smooth)

                                        df_trial, info_delay, delay_timestamps, info_deviation, info_shoot, info_gain, gain = analyze.get_saccadeMovement(df_trial, anti=anti)

                                        '''
                                                                                        Data Check: Decide if data analysis is passable.
                                                                                                    If not, try the other eye
                                                                                    '''
                                        # Decide if we should try the other eye
                                        trial_length = (df_trial.iloc[-1]['timestamp'] - df_trial.iloc[0]['timestamp'])
                                        blink_length = info_blink['duration'].sum()
                                        missing_perc_threshold = 0.2
                                        if (blink_length > (trial_length * missing_perc_threshold)):  # or (info_gaze is None):
                                            print('\t\tTrying other eye...')
                                            df_trial_t = df_trial_raw.copy()
                                            # Switch Eyes
                                            if eye_lowError == 'left':
                                                eye_lowError = 'right'
                                                df_trial_t.rename(
                                                    columns={'pos_x_right': 'pos_x', 'pos_y_right': 'pos_y', 'vel_x_right': 'vel_x', 'vel_y_right': 'vel_y'},
                                                    inplace=True)
                                            elif eye_lowError == 'right':
                                                eye_lowError = 'left'
                                                df_trial_t.rename(
                                                    columns={'pos_x_left': 'pos_x', 'pos_y_left': 'pos_y', 'vel_x_left': 'vel_x', 'vel_y_left': 'vel_y'},
                                                    inplace=True)

                                            # Measure General Eye Movement
                                            df_trial_t, info_saccade_t, info_fixation_t, info_blink_t, info_pupil_t, _ = analyze.get_eyeMovement(df_trial_t,
                                                                                                                                analysis_constants,
                                                                                                                                'timestamp', 'pos_x',
                                                                                                                                'pos_y', 'vel_x', 'vel_y',
                                                                                                                                notes=None, smooth=smooth)
                                            # df_trial_t, info_saccade_t, info_fixation_t, info_blink_t = annotation.remove_eyeMovement( notes_trial, df_trial_t, info_saccade_t, info_fixation_t, info_blink_t, 'timestamp', 'timestamp_start', 'timestamp_end', desc_suffix=eye_lowError)
                                            df_trial_t, info_delay_t, delay_timestamps_t, info_deviation_t, info_shoot_t, info_gain_t, gain_t = analyze.get_saccadeMovement(df_trial_t, anti=anti)

                                            if  (info_blink_t['duration'].sum() < (blink_length)):
                                                df_trial = df_trial_t
                                                info_delay = info_delay_t
                                                delay_timestamps = delay_timestamps_t
                                                info_deviation = info_deviation_t
                                                info_shoot = info_shoot_t
                                                info_gain = info_gain_t
                                                gain = gain_t
                                            else:
                                                print('MISSING ERROR: Could not find enough data from ', subject, ' - ', description_trial)
                                        if info_delay is not None:
                                            delay_df = pd.concat([delay_df, info_delay], ignore_index=True)
                                        if info_deviation is not None:
                                            deviation_df = pd.concat([deviation_df, info_deviation], ignore_index=True)
                                        if info_shoot is not None:
                                            shoot_df = pd.concat([shoot_df, info_shoot], ignore_index=True)
                                        if info_gain is not None:
                                            gain_df = pd.concat([gain_df, info_gain], ignore_index=True)
                                        if len(gain) > 0:
                                            gain_list.extend(gain)

                                        if show_plots or save_plots:
                                            # if group == 'AD':
                                            try:
                                                folder_output = 'plot_eyeMovement'
                                                if not os.path.isdir(os.path.join( path_out_subject, folder_output)):
                                                    os.mkdir(os.path.join( path_out_subject, folder_output))
                                                path_output_plots = os.path.join( path_out_subject, folder_output,  description_trial+'.png')
                                                fplot.eyePos_prepPlot(      df_trial, 'timestamp', 'pos_x', 'pos_y')
                                                # fplot.plot_raw(             df_trial, 'timestamp', 'pos_x', 'pos_y', save_plots=save_plots, save_path=path_output_plots, target=True)
                                                # fplot.plot_saccades(        df_trial, 'timestamp', 'pos_x', 'pos_y', 'vel', 'acel', 'saccade',          analysis_constants['threshold_fixVel'], analysis_constants['threshold_fixAcc'], save_plots=save_plots, save_path=path_output_plots)
                                                fplot.plot_saccades_blinks( df_trial, 'timestamp', 'pos_x', 'pos_y', 'vel', 'acel', 'saccade', 'blink', 'pursuit', analysis_constants['threshold_fixVel'], analysis_constants['threshold_fixAcc'], horizontal=True, delay_timestamps=delay_timestamps, save_plots=save_plots, save_path=path_output_plots, target=True, smooth=smooth)
                                            except Exception as e:
                                                print('ERROR plotting ', description_trial)
                                                print(e)
                                                # Close all figures to preserve memory
                                                for fig in plt.get_fignums():
                                                    plt.figure( fig)
                                                    plt.clf()
                                                    plt.close()
                                                continue

                                summary_dict['subject'] = [subject]
                                summary_dict['session'] = [session_count]
                                summary_dict['group'] = [group]
                                summary_dict['trial'] = [path_output.split('/')[-1]]

                                # Calculate by trials with basic statistics
                                if len(shoot_df) > 0:
                                    for column in shoot_df:
                                        summary_dict['mean_' + str(column)] =  np.mean(np.array(shoot_df[column]))
                                        summary_dict['max_' + str(column)] = np.max(np.array(shoot_df[column]))
                                        summary_dict['median_' + str(column)] = np.median(np.array(shoot_df[column]))
                                        summary_dict['std_' + str(column)] = np.std(np.array(shoot_df[column]))
                                        if str(column) == 'wrong_direction':
                                            summary_dict['sum_' + str(column)] = np.sum(np.array(shoot_df[column]))
                                        if str(column) == 'no_reaction':
                                            summary_dict['sum_' + str(column)] = np.sum(np.array(shoot_df[column]))

                                if len(delay_df) > 0:
                                    for column in delay_df:
                                        summary_dict['mean_' + str(column)] = np.mean(np.array(delay_df[column]))
                                        summary_dict['max_' + str(column)] = np.max(np.array(delay_df[column]))
                                        summary_dict['median_' + str(column)] = np.median(np.array(delay_df[column]))
                                        summary_dict['std_' + str(column)] = np.std(np.array(delay_df[column]))

                                if len(deviation_df) > 0:
                                    for column in deviation_df:
                                        summary_dict['mean_' + str(column)] = np.mean(np.array(deviation_df[column]))
                                        summary_dict['max_' + str(column)] = np.max(np.array(deviation_df[column]))
                                        summary_dict['median_' + str(column)] = np.median(np.array(deviation_df[column]))
                                        summary_dict['std_' + str(column)] = np.std(np.array(deviation_df[column]))

                                if len(gain_df) > 0 and not anti:
                                    summary_dict['Hypometria Percentage [%]'] = np.sum(np.array(gain_df['num_hypo']))/np.sum(np.array(gain_df['num_saccade']))
                                    summary_dict['Hypermetria Percentage [%]'] = np.sum(np.array(gain_df['num_hyper'])) / np.sum(np.array(gain_df['num_saccade']))
                                    num_correct = np.sum(np.array(gain_df['num_saccade'])) - np.sum(np.array(gain_df['num_hypo'])) - np.sum(np.array(gain_df['num_hyper']))
                                    summary_dict['Accuracy [%]'] = num_correct / np.sum(np.array(gain_df['num_saccade']))

                                    summary_dict['Hypometria Gain [%] (\u03BC)'] = np.mean(np.array(gain_df['hypo_mean'])) if np.sum(np.array(gain_df['num_hypo'])) > 0 else np.nan
                                    summary_dict['Hypermetria Gain [%] (\u03BC)'] = np.mean(np.array(gain_df['hyper_mean'])) if np.sum(np.array(gain_df['num_hyper'])) > 0 else np.nan
                                    summary_dict['Gain [%] (min)'] = np.min(np.array(gain_list))
                                    summary_dict['Gain [%] (max)'] = np.max(np.array(gain_list))
                                    summary_dict['Gain [%] (\u03C3)'] = np.std(np.array(gain_list))


                                summary_subject = pd.DataFrame(summary_dict)
                                all_df = pd.concat([all_df, summary_subject], ignore_index=True)

    not_found_df = pd.DataFrame(not_found_df)
    # not_found_df.to_csv(path_output+'/not_found_df.csv', index=False)

    # boxplot_order = ['PD', 'CTL', 'PDM', 'AD']
    # for col in all_df.columns:
    #     if col != 'subject' and col != 'group' and col != 'trial' and col != 'session':
    #         summarize.get_boxplots(all_df, col, order=boxplot_order, path_output=path_output, save_outputs=True,
    #                                show_plots=False)

    all_df.to_csv(path_output+'/saccade_data_summary.csv', index=False)


if __name__ == '__main__':
    main()
    print('\n\nFin.')