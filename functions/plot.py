from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

try:
    import eyetracking.functions.process as fcn
except:
    try:
        import functions.process as fcn
    except:
        import process as fcn

global_plot_dpi = 100


def eyePos_prepPlot(df_trial, col_timestamp, col_pos_x, col_pos_y):
    # Before plotting, convert time to seconds and start at 0 sec
    df_trial[col_timestamp] = (df_trial[col_timestamp] - df_trial.iloc[0][col_timestamp]) / 1e3
    # Remove outlier values
    df_trial[col_pos_x]  = fcn.remove_outliers( df_trial[col_pos_x].to_numpy(), 0, 1920)
    df_trial[col_pos_y]  = fcn.remove_outliers( df_trial[col_pos_y].to_numpy(), 0, 1080)


def timestamp_prepPlot(df_trial, col_timestamp, timestamp_zero=None, multiplier=0.001):
    # Before plotting, convert time to seconds and start at 0 sec
    if timestamp_zero is None:
        if isinstance( col_timestamp, list):
            timestamp_zero = df_trial.iloc[0][col_timestamp[0]]
        else:
            timestamp_zero = df_trial.iloc[0][col_timestamp]

    df_trial.loc[:,col_timestamp] = (df_trial.loc[:,col_timestamp] - float(timestamp_zero)) * multiplier


def plot_raw(df_trial, col_timestamp, col_pos_x, col_pos_y, save_plots=False, save_path=None, target=False):

    _, ax0 = plt.subplots()
    plt.plot( df_trial[col_timestamp], df_trial[col_pos_x], 'b', label="gaze_X")
    plt.plot( df_trial[col_timestamp], df_trial[col_pos_y], 'g', label="gaze_Y")
    if target:
        plt.plot(df_trial[col_timestamp],  df_trial['target_x'], 'c--', label="target_X")
        plt.plot(df_trial[col_timestamp], df_trial['target_y'], 'r--', label="target_Y")
    plt.title( 'X/Y Location vs Time')
    plt.ylabel('Pixels')
    plt.xlabel('Time')
    plt.tick_params(left=False, right=False, top=False)
    plt.legend()

    if save_plots:
        filename_parts = os.path.splitext(save_path)
        filename = filename_parts[0] + '_raw-locaitonVtime' + filename_parts[1]
        plt.savefig(filename, dpi=global_plot_dpi)


    plt.figure()
    ax0 = plt.subplot(1,1,1)
    plt.ylim((df_trial[col_pos_y].max(), df_trial[col_pos_y].min())) # Invert Y axis
    plt.plot( df_trial[col_pos_x], df_trial[col_pos_y], 'b', label="pos_X/Y")
    if target:
        plt.plot(df_trial['target_x'], df_trial['target_y'], 'g--', label="target_X/Y")
    plt.title('X/Y Location')
    plt.legend()
    plt.tick_params(left=False, right=False, top=False)

    if save_plots:
        filename_parts = os.path.splitext(save_path)
        filename = filename_parts[0] + '_raw-locaiton' + filename_parts[1]
        plt.savefig(filename, dpi=global_plot_dpi)


def plot_trim(df_data, trial_start, trial_end, col_timestamp, col_pos_x, col_pos_y, save_plots=False, save_path=None):
    trial_end = trial_end - 1 # B/c end is exclusive, but need existing values to plot

    plt.figure()
    ax0 = plt.subplot(2,1,1)
    plt.plot( df_data[col_timestamp], df_data[col_pos_x], 'b')
    plt.plot( df_data[col_timestamp], df_data[col_pos_y], 'g')
    plt.axvspan( df_data.loc[trial_start, col_timestamp], df_data.loc[trial_end, col_timestamp], color='g', alpha=0.3)
    plt.title('Trial Trim start/stop identification')
    plt.xlabel('Time (sec)')
    plt.ylabel('Eye Position (X/Y)')
    ax0.set_xticklabels([])

    zoom_margin = 300
    ax1 = plt.subplot(2,2,3)
    plt.plot( df_data.loc[:trial_start+zoom_margin, col_timestamp], df_data.loc[:trial_start+zoom_margin, col_pos_x], 'b')
    plt.plot( df_data.loc[:trial_start+zoom_margin, col_timestamp], df_data.loc[:trial_start+zoom_margin, col_pos_y], 'g')
    plt.plot( df_data.loc[             trial_start, col_timestamp], df_data.loc[             trial_start, col_pos_x], 'ro')
    plt.plot( df_data.loc[             trial_start, col_timestamp], df_data.loc[             trial_start, col_pos_y], 'ro')
    ax1.set_xticklabels([])

    ax2 = plt.subplot(2,2,4)
    plt.plot( df_data.loc[trial_end-zoom_margin:, col_timestamp], df_data.loc[trial_end-zoom_margin:, col_pos_x], 'b')
    plt.plot( df_data.loc[trial_end-zoom_margin:, col_timestamp], df_data.loc[trial_end-zoom_margin:, col_pos_y], 'g')
    plt.plot( df_data.loc[             trial_end, col_timestamp], df_data.loc[             trial_end, col_pos_x], 'ro')
    plt.plot( df_data.loc[             trial_end, col_timestamp], df_data.loc[             trial_end, col_pos_y], 'ro')
    ax2.set_xticklabels([])

    if save_plots:
        filename_parts = os.path.splitext(save_path)
        filename = filename_parts[0] + '_trim' + filename_parts[1]
        plt.savefig(filename, dpi=global_plot_dpi)


def plot_saccades( df_trial, col_timestamp, col_pos_x_left, col_pos_y_left, col_vel_left, col_acel_left, col_saccade_left, col_fix_left, threshold_fixVel, threshold_fixAcc, save_plots=False, save_path=None):
    startStop_left  = fcn.get_indexStartStop( df_trial[col_saccade_left])
    startStop_left_fix  = fcn.get_indexStartStop( df_trial[col_fix_left])

    linewidth = 0.4

    plt.figure()
    plt.ylim((df_trial[col_pos_y_left].max(), df_trial[col_pos_y_left].min())) # Invert Y axis
    # plt.plot(df_trial[col_pos_x_right], df_trial[col_pos_y_left], 'g', label=col_pos_x_left)
    # data = Image.open('/Users/remus/Desktop/Cookie-Thef.png')
    # print(data.size)
    # data = data.resize((1920, 1080), Image.ANTIALIAS)
    # print(data.size)
    plt.plot(df_trial.loc[startStop_left[0][0]:startStop_left[0][1], col_pos_x_left], df_trial.loc[startStop_left[0][0]:startStop_left[0][1], col_pos_y_left], 'r', linewidth=2, alpha=0.7,
             label='saccade')
    for start_sac, end_sac in startStop_left[1:]:
        plt.plot(df_trial.loc[start_sac:end_sac, col_pos_x_left], df_trial.loc[start_sac:end_sac, col_pos_y_left], 'r', linewidth=2, alpha=0.7)
        plt.plot(df_trial.loc[start_sac, col_pos_x_left],  df_trial.loc[start_sac, col_pos_y_left],  'ro')
        plt.plot(df_trial.loc[  end_sac, col_pos_x_left],  df_trial.loc[  end_sac, col_pos_y_left],  'ro')

    plt.plot(df_trial.loc[startStop_left_fix[0][0]:startStop_left_fix[0][1], col_pos_x_left], df_trial.loc[startStop_left_fix[0][0]:startStop_left_fix[0][1], col_pos_y_left], 'b', linewidth=2, alpha=0.7,
             label='fixation')
    for start_fix, end_fix in startStop_left_fix[1:]:
        plt.plot(df_trial.loc[start_fix:end_fix, col_pos_x_left], df_trial.loc[start_fix:end_fix, col_pos_y_left], 'b', linewidth=2, alpha=0.7)
    # plt.imshow(data)
    plt.title('Saccades')
    plt.grid(alpha=0.5)
    plt.ylabel('Eye position in Y (pixels)')
    plt.xlabel('Eye position in X (pixels)')
    plt.legend()
    plt.tight_layout()

    if save_plots:
        filename_parts = os.path.splitext(save_path)
        filename = filename_parts[0] + '_sac' + filename_parts[1]
        plt.savefig(filename, dpi=global_plot_dpi)


    plt.figure(figsize=(16, 9))
    plt.plot(df_trial[col_timestamp], df_trial[col_pos_x_left], 'b', linewidth=linewidth, label=col_pos_x_left)
    plt.plot(df_trial[col_timestamp], df_trial[col_pos_y_left], 'c', linewidth=linewidth, label=col_pos_y_left)
    plt.plot(df_trial.loc[startStop_left[0][0]:startStop_left[0][1], col_timestamp], df_trial.loc[startStop_left[0][0]:startStop_left[0][1],col_pos_x_left], 'r', label='saccade', linewidth=linewidth*2, alpha=0.7)
    for start_sac, end_sac in startStop_left[1:]:
        plt.plot(df_trial.loc[start_sac:end_sac, col_timestamp], df_trial.loc[start_sac:end_sac, col_pos_x_left], 'r', linewidth=linewidth*2, alpha=0.7)
    plt.ylabel('Eye position in X (pixels)')
    plt.xlabel('Time (sec)')
    # plt.legend()
    plt.tight_layout()
    
    if save_plots:
        filename_parts = os.path.splitext(save_path)
        filename = filename_parts[0] + '_sacVtime' + filename_parts[1]
        plt.savefig(filename, dpi=global_plot_dpi*4)


    plt.figure(figsize=(16, 9))
    ax0 = plt.subplot(2,1,1)
    plt.plot(df_trial[col_timestamp], df_trial[col_pos_x_left], 'b', linewidth=linewidth, label=col_pos_x_left)
    for start_sac, end_sac in startStop_left:
        plt.plot(df_trial.loc[start_sac:end_sac, col_timestamp], df_trial.loc[start_sac:end_sac, col_pos_x_left], 'r', linewidth=linewidth*2, alpha=0.7)
    plt.title('Ssaccades vs Time')
    plt.ylabel('gaze_X (pixels))')

    ax2 = plt.subplot(2,1,2, sharex=ax0)
    plt.plot(df_trial[col_timestamp], df_trial[col_acel_left], 'c', linewidth=linewidth, label='Acceleration')
    plt.plot(df_trial[col_timestamp], [threshold_fixAcc]*len(df_trial[col_timestamp]), 'k', linewidth=0.5)
    plt.plot(df_trial[col_timestamp], [threshold_fixAcc*-1]*len(df_trial[col_timestamp]), 'k', linewidth=0.5)
    plt.ylim((-4*threshold_fixAcc,threshold_fixAcc*4))
    plt.title('Saccade Thresholds')
    plt.ylabel('Acceleration (deg/s/s)')
    ax2.tick_params('y', colors='c')
    ax3 = ax2.twinx()
    plt.plot(df_trial[col_timestamp], df_trial[col_vel_left], 'b', linewidth=linewidth, label='velocity')
    plt.plot(df_trial[col_timestamp], [threshold_fixVel]*len(df_trial[col_timestamp]), 'k', linewidth=0.5)
    plt.plot(df_trial[col_timestamp], [threshold_fixVel*-1]*len(df_trial[col_timestamp]), 'k', linewidth=0.5)
    plt.ylim((-4*threshold_fixVel,threshold_fixVel*4))
    plt.ylabel('Velocity (deg/s)')
    ax3.tick_params('y', colors='b')
    ax0.set_xticklabels([])
    ax3.set_xticklabels([])
    if save_plots:
        filename_parts = os.path.splitext(save_path)
        filename = filename_parts[0] + '_sacVtime-thresh' + filename_parts[1]
        plt.savefig(filename, dpi=global_plot_dpi*4)


def plot_saccades_blinks(df_trial, col_timestamp, col_pos_x_left, col_pos_y_left, col_vel_left, col_acel_left, col_saccade_left, col_blink_left, col_pursuit_left, threshold_fixVel, threshold_fixAcc, horizontal=True, delay_timestamps=None, save_plots=False, save_path=None, target=False, smooth=False):
    startStop_saccade_left  = fcn.get_indexStartStop( df_trial[col_saccade_left])
    startStop_blink_left    = fcn.get_indexStartStop( df_trial[col_blink_left])
    plt.figure()
    if target:
        plt.plot(df_trial[col_timestamp], df_trial['target_x'], 'c--', label="target_X")
        plt.plot(df_trial[col_timestamp], df_trial['target_y'], 'm--', label="target_Y")

    if horizontal:
        plt.plot(df_trial[col_timestamp], df_trial[col_pos_x_left], 'b', label=col_pos_x_left)
        plt.plot(df_trial[col_timestamp], df_trial[col_pos_y_left], 'g', label=col_pos_y_left)
    else:
        plt.plot(df_trial[col_timestamp], df_trial[col_pos_x_left], 'b', label=col_pos_x_left)
        plt.plot(df_trial[col_timestamp], df_trial[col_pos_y_left], 'g', label=col_pos_y_left)

    # if smooth:
    #     startStop_pursuit_left = fcn.get_indexStartStop(df_trial[col_pursuit_left])
    #     if len(startStop_pursuit_left) > 0:
    #         plt.plot(df_trial.loc[startStop_pursuit_left[0][0]:startStop_pursuit_left[0][1], col_timestamp],
    #                  df_trial.loc[startStop_pursuit_left[0][0]:startStop_pursuit_left[0][1], col_pos_x_left], 'k', label='pursuit', linewidth=3,
    #                  alpha=0.7)
    #         plt.plot(df_trial.loc[startStop_pursuit_left[0][0]:startStop_pursuit_left[0][1], col_timestamp],
    #                  df_trial.loc[startStop_pursuit_left[0][0]:startStop_pursuit_left[0][1], col_pos_y_left], 'k', linewidth=3,
    #                  alpha=0.7)
    #         for start_sac, end_sac in startStop_pursuit_left[1:]:
    #             plt.plot(df_trial.loc[start_sac:end_sac, col_timestamp], df_trial.loc[start_sac:end_sac, col_pos_x_left], 'k', linewidth=3, alpha=0.7)
    #             plt.plot(df_trial.loc[start_sac:end_sac, col_timestamp], df_trial.loc[start_sac:end_sac, col_pos_y_left], 'k', linewidth=3, alpha=0.7)

    if len(startStop_saccade_left) > 0:
        plt.plot(df_trial.loc[startStop_saccade_left[0][0]:startStop_saccade_left[0][1], col_timestamp],
                 df_trial.loc[startStop_saccade_left[0][0]:startStop_saccade_left[0][1], col_pos_x_left], 'r', label='saccade', linewidth=3,
                 alpha=0.7)
        plt.plot(df_trial.loc[startStop_saccade_left[0][0]:startStop_saccade_left[0][1], col_timestamp],
                 df_trial.loc[startStop_saccade_left[0][0]:startStop_saccade_left[0][1], col_pos_y_left], 'r', linewidth=3,
                 alpha=0.7)
        for start_sac, end_sac in startStop_saccade_left[1:]:
            plt.plot(df_trial.loc[start_sac:end_sac, col_timestamp], df_trial.loc[start_sac:end_sac, col_pos_x_left], 'r', linewidth=3, alpha=0.7)
            plt.plot(df_trial.loc[start_sac:end_sac, col_timestamp], df_trial.loc[start_sac:end_sac, col_pos_y_left], 'r', linewidth=3, alpha=0.7)

    if delay_timestamps:
        plt.plot([df_trial.loc[delay_timestamps[0][0], col_timestamp], df_trial.loc[delay_timestamps[0][1], col_timestamp]], [0, 0], 'k',
                 linewidth=5, label='latency')
        for i in range(1, len(delay_timestamps)):
            plt.plot([df_trial.loc[delay_timestamps[i][0], col_timestamp], df_trial.loc[delay_timestamps[i][1], col_timestamp]], [0, 0], 'k',
                    linewidth=5)
    if len(startStop_blink_left) > 0:
        plt.plot([df_trial.loc[startStop_blink_left[0][0], col_timestamp], df_trial.loc[startStop_blink_left[0][1], col_timestamp]], [0,0], 'y', linewidth=5, label='blink')
        for start_blink, end_blink in startStop_blink_left[1:]:
            plt.plot([df_trial.loc[start_blink, col_timestamp], df_trial.loc[end_blink, col_timestamp]], [0,0], 'y', linewidth=5)

    plt.legend()
    plt.title('Eye movement')
    plt.ylabel('Gaze position (pixels)')
    plt.xlabel('Time (sec)')
    # plt.legend()
    plt.tight_layout()

    if save_plots:
        filename_parts = os.path.splitext(save_path)
        filename = filename_parts[0] + '_sacBlinkVtime-Left' + filename_parts[1]
        plt.savefig(filename, dpi=global_plot_dpi)
    plt.clf()
    plt.close()


def plot_word( df_trial, col_timestamp, col_pos_x, col_pos_y, col_fixation, col_focus, col_gaze_line, col_gaze_word, save_plots=False, save_path=None):
    plt.figure(figsize=(16,9))
    ax0 = plt.subplot(2,1,1)
    plt.plot( df_trial[col_timestamp], df_trial[col_pos_x], 'b', label='gaze-X')
    plt.plot( df_trial[col_timestamp], df_trial[col_pos_y], 'g', label='gaze-Y')
    plt.title('Stimuli (word) Fixation Identification')
    plt.xlabel('Time (sec)')
    plt.ylabel('Right Eye Position')
    plt.legend()

    ax1 = plt.subplot(2,1,2, sharex=ax0)
    # index_plot = df_trial.index
    index_plot = df_trial[col_fixation] & (df_trial[col_gaze_line] % 1 == 0) & (df_trial[col_gaze_word] % 1 == 0)
    plt.plot( df_trial.loc[index_plot, col_timestamp], df_trial.loc[index_plot, col_gaze_line]+0.02, color='c', marker='o', linestyle='None', label='gazeline')
    plt.plot( df_trial.loc[index_plot, col_timestamp], df_trial.loc[index_plot, col_gaze_word]-0.02, color='m', marker='o', linestyle='None', label='gazeword')
    time_focus = df_trial.loc[ df_trial[col_focus], col_timestamp]
    plt.plot( [ time_focus.iloc[0], time_focus.iloc[-1] ], [-0.25,-0.25], 'b', linewidth=2, label='focus')
    plt.ylim((-0.3,5.3))
    plt.xlabel('Time (sec)')
    plt.ylabel('Line/Word Index')
    plt.legend()

    if save_plots:
        filename_parts = os.path.splitext(save_path)
        filename = filename_parts[0] + '_wordFix' + filename_parts[1]
        plt.savefig(filename, dpi=global_plot_dpi)


def plot_wordTokens_correct( data_wordBegin, col_time, col_token, col_wordIndex, save_plots=False, save_path=None):
    plt.figure()
    for _, row in data_wordBegin.iterrows():
        if np.isnan( row[col_wordIndex]):
            plt.plot(row[col_time], -1, 'ro')
            plt.annotate(row[col_token], [row[col_time], -1])
        else:
            plt.plot(row[col_time], row[col_wordIndex], 'go')
            plt.annotate(row[col_token], [row[col_time], row[col_wordIndex]])

    plt.title('Correct Word on Audio')
    plt.ylabel('Word Index')
    plt.xlabel('Time (sec)')

    if save_plots:
        filename_parts = os.path.splitext(save_path)
        filename = filename_parts[0] + '_wordBegin-correct' + filename_parts[1]
        plt.savefig(filename, dpi=global_plot_dpi)


def plot_alignedGazeAudio( data_sac, data_fix, data_blk, data_wordBegin, data_audio, time_audio, title='', save_plots=False, save_path=None):
    plt.figure(figsize=(16,9))
    ax0 = plt.subplot(2,1,1)
    _plot_gazeWord( data_fix, data_blk, data_sac, df_event=data_wordBegin, df_event_colTime='Time', df_event_colEvent='word_word')
    plt.title(title+' EYE GAZE')
    plt.ylabel('Word Index')

    ax2 = plt.subplot(2,1,2,sharex=ax0)
    plt.plot(time_audio,data_audio)
    plt.plot(data_wordBegin['Time'], data_wordBegin['word_word']-0.05, 'y^')
    plt.title(title+' RAW AUDIO')
    plt.xlabel('Trial Time (sec)')

    if save_plots:
        filename_parts = os.path.splitext(save_path)
        filename = filename_parts[0] + '_wordBegin-audio' + filename_parts[1]
        plt.savefig(filename, dpi=global_plot_dpi)


def plot_wordCorrect_value( data_sac, data_fix, data_blk, data_wordBegin, col_time, col_value, col_color=None, save_plots=False, save_path=None):
    plt.figure(figsize=(16,9))
    ax0 = plt.subplot(2,1,1)
    _plot_gazeWord( data_fix, data_blk, data_sac, df_event=data_wordBegin, df_event_colTime=col_time, df_event_colEvent='word_word')
    plt.title('Eye Gaze')
    plt.ylabel('Word Index')
    plt.ylim((-0.25,3.25))

    ax1 = plt.subplot(2,1,2,sharex=ax0)
    if col_color is not None:
        for _, row in data_wordBegin.iterrows():
            plt.plot( [row[col_time],row[col_time]], [0,row[col_value]], 'o-', color=row[col_color])
    else:
        for _, row in data_wordBegin.iterrows():
            plt.plot( [row[col_time],row[col_time]], [0,row[col_value]], 'o-')

    plt.axhline(y=0, color='k', alpha=0.3)
    plt.title(col_value)
    plt.ylabel(col_value+' (sec)')
    plt.xlabel('Trial Time (sec)')

    if save_plots:
        filename_parts = os.path.splitext(save_path)
        filename = filename_parts[0] + '_wordBegin-' + col_value + filename_parts[1]
        plt.savefig(filename, dpi=global_plot_dpi)


def plot_fixationStimuli( cluster_fcn, df_fixation, col_fix_pos_x, col_fix_pos_y, col_gaze_word, col_gaze_line, cluster_descriptions=None, annotate=False, save_plots=False, save_path=None):

    # pos_stimuli = cluster_fcn.cluster_centers_
    pos_stimuli = cluster_fcn._fit_X
    num_words   = df_fixation[col_gaze_word].max() +1 # add 1 b/c zero indexed
    
    plt.figure()
    plt.plot(pos_stimuli[:,0], pos_stimuli[:,1], 'r^')
    if (cluster_descriptions is not None) and annotate:
        for i, desc in enumerate( cluster_descriptions):
            plt.annotate(desc, pos_stimuli[i,:])
    
    color_line   = [ ['b', 'g'], ['orange', 'm'], ['brown', 'gray'] ]
    for line in sorted( df_fixation[col_gaze_line].unique()):
        line_fixations = df_fixation[ df_fixation[col_gaze_line] == line]
        for word in sorted( line_fixations[col_gaze_word].unique()):
            word_fixations      = line_fixations[ line_fixations[col_gaze_word] == word]
            word_cluster_center = pos_stimuli[line*num_words+word,:]
            plot_color = color_line[line%3][word%2]

            plt.plot(word_fixations[col_fix_pos_x], word_fixations[col_fix_pos_y], color=plot_color, marker='o', linestyle='None')
            for fix_x, fix_y in zip(word_fixations[col_fix_pos_x], word_fixations[col_fix_pos_y]):
                plt.plot([word_cluster_center[0],fix_x], [word_cluster_center[1],fix_y], color=plot_color, linestyle=':', alpha=0.5)
    
    plt.xlim((0,1920))
    plt.ylim((0,1080))
    plt.tight_layout()

    if save_plots:
        filename_parts = os.path.splitext(save_path)
        filename = filename_parts[0] + '_fixStimuli' + filename_parts[1]
        plt.savefig(filename, dpi=global_plot_dpi)



def plot_boundaryStimuli( cluster_fcn, cluster_descriptions=None, annotate=True, save_plots=False, save_path=None):

    # Plot the decision boundary. For that, we will assign a color to each
    resolution = 1
    xx, yy = np.meshgrid(np.arange(0, 1920, resolution, dtype=np.float32), np.arange(0, 1080, resolution, dtype=np.float32))
    zz = np.c_[xx.ravel(), yy.ravel()]
    zz = cluster_fcn.predict(zz.astype(np.float64))
    zz = zz.reshape(xx.shape)

    plt.figure()
    # if cluster_descriptions is None:
    if True:
        plt.contourf(xx, yy, zz, cmap=plt.get_cmap('prism'), levels=24*2, alpha=0.2)
    else:
        from matplotlib.colors import ListedColormap
        cmap = []
        for desc in cluster_descriptions:
            cmap.append(desc[0])
        plt.contourf(xx, yy, zz, cmap=ListedColormap(cmap), levels=len(cmap), alpha=0.2)
    # plt.colorbar()

    # pos_stimuli = cluster_fcn.cluster_centers_
    pos_stimuli = cluster_fcn._fit_X
    plt.plot(pos_stimuli[:,0], pos_stimuli[:,1], 'r^')
    if (cluster_descriptions is not None) and annotate:
        for i, desc in enumerate( cluster_descriptions):
            plt.annotate(','.join(desc), pos_stimuli[i,:])

    plt.xlim((0,1920))
    plt.ylim((0,1080))
    plt.tight_layout()

    if save_plots:
        filename_parts = os.path.splitext(save_path)
        filename = filename_parts[0] + '_fixBoundary' + filename_parts[1]
        plt.savefig(filename, dpi=global_plot_dpi)


def _plot_gazeWord( data_fix, data_blk, data_sac, df_event=None, df_event_colTime=None, df_event_colEvent=None):
    for _, row in data_fix.iterrows():
        plt.plot( [row['timestamp_start'], row['timestamp_end']], [row['gaze_word'], row['gaze_word']], 'm', linewidth=3)
    for _, row in data_blk.iterrows():
        plt.plot( [row['timestamp_start'], row['timestamp_end']], [row['gaze_word_start'], row['gaze_word_end']], 'g', linewidth=2, alpha=0.5)
    for _, row in data_sac.iterrows():
        plt.plot( [row['timestamp_start'], row['timestamp_end']], [row['gaze_word_start'], row['gaze_word_end']], 'r', linewidth=1, alpha=0.3)
    if df_event is not None:
        plt.plot(df_event[df_event_colTime], df_event[df_event_colEvent]-0.05, 'y^')
    plt.ylim((-0.25,3.25))