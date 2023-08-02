from sklearn.neighbors import KNeighborsClassifier as skl_knn
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

try:
    import eyetracking.functions.annotation as annotation
except:
    try:
        import functions.annotation as annotation
    except:
        import annotation as annotation



def get_blinks( df_data, column_position, column_velocity, column_acceleration, closest_blink=50, threshold_fixVel=30, threshold_fixAcc=4000):
    '''Identifies blink events, and identifies each individual start/stop index
    
    Args:
        df_data (pandas dataframe):         Pandas dataframe containing all of the relevant raw data
        column_position (str or list):      Column header(s) designating the position of eye position data. Nan values in this column will be used initially to locate blinks.
        column_velocity (str):              Column header designating the locaiton of eye radial velocity data in degrees/second.
        column_acceleration (str):          column header designating the position of eye radial acceleration data in degrees/second^2.
        closest_blink (float, optional):    Smallest separation in time between blinks in data points. Any blinks less than this far apart will be merged into one event.
        threshold_fixVel (float, optional): Maximum velocity threshold allowed during a fixation.
        threshold_fixAcc (float, optional): Maximum acceleration threshold allowed during a fixation.
        
    Returns:
        dataframe with blinks removed and replaced with interpolated values
    '''
    # Get initial start/stop values based on NaN values in the position column
    blink_start, blink_end = get_indexStartStop( np.any( df_data[column_position].isna(), axis=1), get_pairs=False)

    # Merge any blinks that are too close
    blink_tooClose = []
    for i in range( len( blink_start)-1):
        # If the end index and the next start index are too close, save that index to delete later
        if (blink_start[i+1] - blink_end[i]) < closest_blink:
            blink_tooClose.append(i)
    # Delete the identified indicies
    blink_end   = blink_end.delete(blink_tooClose)
    blink_start = blink_start.delete([i+1 for i in blink_tooClose])

    # Create the final blink output array, initialized to all false
    blink_return = pd.Series(False, index=df_data.index)
    for start, end in zip(blink_start, blink_end):
        # Extend the start index backwards until the velocity and acceleration can be classified as belonging to the previous fixation
        while (np.abs(df_data.loc[start, column_velocity]) > threshold_fixVel) or (np.abs(df_data.loc[start, column_acceleration]) > threshold_fixAcc) or any(df_data.loc[start, [column_velocity, column_acceleration]].isna()):
            start -= 1
            if start <= df_data.index[0]:
                start = df_data.index[0]
                break
        # Extend the end index forwards until the velocity and acceleration can be classified as belonging to the next fixation
        while (np.abs(df_data.loc[  end, column_velocity]) > threshold_fixVel) or (np.abs(df_data.loc[  end, column_acceleration]) > threshold_fixAcc) or any(df_data.loc[  end, [column_velocity, column_acceleration]].isna()):
            end += 1
            if end >= df_data.index[-1]:
                end = df_data.index[-1]
                break

        # All values between the last and next fixation should be classified as a blink
        blink_return.loc[start:end] = True
    
    return blink_return



def get_saccades( df_data, column_positionX, column_positionY, column_velocity, column_acceleration, column_blink, closet_sac=20,threshold_fixDist=20, threshold_fixVel=25, threshold_fixAcc=3000):
    '''Identifies blink events, and identifies each individual start/stop index
    
    Args:
        df_data (pandas dataframe):          Pandas dataframe containing all of the relevant raw data
        column_positionX (str):              Column header designating the X-axis position (pixels) of eye position data.
        column_positionY (str):              Column header designating the Y-axis position (pixels) of eye position data.
        column_velocity (str):               Column header designating the locaiton of eye radial velocity data in degrees/second.
        column_acceleration (str):           column header designating the position of eye radial acceleration data in degrees/second^2.
        column_blink (str):                  column header designating the occurance of blink events.
        threshold_fixDist (float, optional): Maximum distance change threshold (pixels) allowed during a fixation.
        threshold_fixVel (float, optional):  Maximum velocity threshold (deg/sec) allowed during a fixation.
        threshold_fixAcc (float, optional):  Maximum acceleration threshold (deg/sec/sec) allowed during a fixation.
        
    Returns:
        dataframe with blinks removed and replaced with interpolated values
    '''
    # Get initial start/stop values based on tbe velocity threshold
    sac_start, sac_end = get_indexStartStop(df_data[column_velocity].abs() > threshold_fixVel, get_pairs=False)

    # remove any blinks that are too close
    saccade_tooClose = []
    for i in range(len(sac_start)):
        # If the end index and the next start index are too close, save that index to delete later
        if (sac_end[i] - sac_start[i]) < closet_sac:
            saccade_tooClose.append(i)
    # Delete the identified indicies
    sac_end = sac_end.delete(saccade_tooClose)
    sac_start = sac_start.delete(saccade_tooClose)

    # Create the final blink output array, initialized to all false
    sac_return = pd.Series(False, index=df_data.index)
    for start, end in zip(sac_start, sac_end):
        # Assure that during this velocity spike, acceleration rise above the threshold and there was not a blink
        if any(df_data.loc[start:end, column_acceleration].abs() > threshold_fixAcc) and not( any(df_data.loc[start:end,column_blink])):

            # Find the specific start/stop values
            # Extend the start index backwards until the velocity and acceleration can be classified as belonging to the previous fixation
            while (np.abs(df_data.loc[start,column_velocity]) > threshold_fixVel) or (np.abs(df_data.loc[start,column_acceleration]) > threshold_fixAcc) or any(df_data.loc[start, [column_velocity,column_acceleration]].isna()):
                start -= 1
                if start <= df_data.index[0]:
                    start = df_data.index[0]
                    break
            # Extend the end index forwards until the velocity and acceleration can be classified as belonging to the next fixation
            while (np.abs(df_data.loc[end,column_velocity]) > threshold_fixVel) or (np.abs(df_data.loc[  end,column_acceleration]) > threshold_fixAcc) or any(df_data.loc[  end, [column_velocity,column_acceleration]].isna()):
                end += 1
                if end >= df_data.index[-1]:
                    end = df_data.index[-1]
                    break

            # Make sure that the new bounds also don't contain any blinks and that this saccade did indeed travel some distance to a new target
            if not( any(df_data.loc[start:end,column_blink])) and (np.sqrt(np.square(df_data.loc[start,column_positionX] - df_data.loc[end,column_positionX]) + np.square(df_data.loc[start,column_positionY] - df_data.loc[end,column_positionY])) > threshold_fixDist):
                # Save the specific start/stop values
                sac_return.loc[start:end] = True

    return sac_return


def get_pursuit(df_data, column_positionX, column_positionY, column_velocity, column_acceleration, column_blink, column_saccade, closet_pur=10, threshold_purDist=5, threshold_avgpurDist=1, threshold_purAcc=1000,
                threshold_purVel=10, threshold_fixVel=25, threshold_fixAcc=6000):
    '''Identifies smooth pursuit events, and identifies each individual start/stop index

    Args:
        df_data (pandas dataframe):          Pandas dataframe containing all of the relevant raw data
        column_positionX (str):              Column header designating the X-axis position (pixels) of eye position data.
        column_positionY (str):              Column header designating the Y-axis position (pixels) of eye position data.
        column_velocity (str):               Column header designating the locaiton of eye radial velocity data in degrees/second.
        column_acceleration (str):           column header designating the position of eye radial acceleration data in degrees/second^2.
        column_blink (str):                  column header designating the occurance of blink events.
        threshold_fixDist (float, optional): Maximum distance change threshold (pixels) allowed during a fixation.
        threshold_fixVel (float, optional):  Maximum velocity threshold (deg/sec) allowed during a fixation.
        threshold_fixAcc (float, optional):  Maximum acceleration threshold (deg/sec/sec) allowed during a fixation.

    Returns:
        dataframe with blinks removed and replaced with interpolated values
    '''
    # Get initial start/stop values based on tbe velocity threshold
    data_bool_up = df_data[column_velocity].abs() > threshold_purVel
    data_bool_down = df_data[column_velocity].abs() < threshold_fixVel
    data_bool = (data_bool_up & data_bool_down)
    pur_start, pur_end = get_indexStartStop(data_bool, get_pairs=False)

    # Merge any blink that are too close
    pur_tooClose = []
    for i in range(len(pur_start) - 1):
        # If the end index and the next start index are too close, save that index to delete later
        if (pur_start[i + 1] - pur_end[i]) < closet_pur:
            pur_tooClose.append(i)
    # Delete the identified indicies
    pur_end = pur_end.delete(pur_tooClose)
    pur_start = pur_start.delete([i + 1 for i in pur_tooClose])

    # Create the final pursuit output array, initialized to all false
    pur_return = pd.Series(False, index=df_data.index)
    for start, end in zip(pur_start, pur_end):
        # Find the specific start/stop values
        # if any(df_data.loc[start:end, column_acceleration].abs() > threshold_purAcc) and any(df_data.loc[start:end, column_acceleration].abs() < threshold_fixAcc):
            # Extend the start index backwards until below the threshold
            # condition = (np.abs(df_data.loc[start, column_acceleration]) > threshold_purAcc) and (np.abs(df_data.loc[start, column_acceleration]) < threshold_fixAcc)
            #
            # while condition:
            #     start -= 1
            #     if start <= df_data.index[0]:
            #         start = df_data.index[0]
            #         break
            # # Extend the end index forwards until the velocity and acceleration can be classified as belonging to the next fixation
            # while condition:
            #     end += 1
            #     if end >= df_data.index[-1]:
            #         end = df_data.index[-1]
            #         break
            # Make sure that the new bounds also don't contain any blinks and that this saccade did indeed travel some distance to a new target

        # avg_moving_distance_x = np.abs(df_data.loc[start:end, column_positionX].diff())
        # avg_moving_distance_x = avg_moving_distance_x[1:]
        # avg_moving_distance_y = np.abs(df_data.loc[start:end, column_positionY].diff())
        # avg_moving_distance_y = avg_moving_distance_y[1:]
        # avg_moving_distance = np.mean(avg_moving_distance_x + avg_moving_distance_y)
        # print(avg_moving_distance)
        travel_distance = np.sqrt(np.square(df_data.loc[start, column_positionX] - df_data.loc[end, column_positionX]) + np.square(
                        df_data.loc[start, column_positionY] - df_data.loc[end, column_positionY]))
        if travel_distance > threshold_purDist: #and avg_moving_distance > threshold_avgpurDist:
            # Save the specific start/stop values
            pur_return.loc[start:end] = True

    return pur_return


def get_stimuliFixation(df_fixation, name_stimuli, col_fix_time_start, col_fix_time_end, col_fix_pos_x, col_fix_pos_y, col_fix_duration, col_gaze='gaze', col_gaze_line='gazeLine', col_gaze_word='gazeWord', col_gaze_text='gazeText', col_gaze_color='gazeColor', col_focus='focus', notes=None):
    '''Adds columns to dataframes which identify the stimui the subject is looking at. Currently adds the columns below
        * gaze
        * gazeLine
        * gazeWord
        * gazeColor
        * gazeText
    
    Args:
        df_fixation (pandas dataframe):     Pandas dataframe containing data about individual fixations
        name_stimuli (str):                 The name of the test to retrieve correlating stimuli for.
        col_fix_time_start:                 column header designating the fixation start timestamp.
        col_fix_time_end:                   column header designating the fixation end timestamp.
        col_fix_pos_x (str):                Column header designating the X-axis position (pixels) of eye position data.
        col_fix_pos_y (str):                Column header designating the Y-axis position (pixels) of eye position data.
        col_fix_duration (str):             Column header designating the presence of blink events.
        col_gaze (str, optional):           Column header designating the gaze stimuli.
        col_gaze_line (str, optional):      Column header designating the gaze line index.
        col_gaze_word (str, optional):      Column header designating the gaze word index.
        col_gaze_text (str, optional):      Column header designating the gaze stimuli text content.
        col_gaze_color (str, optional):     Column header designating the gaze stimuli color.
        col_focus (str, optional):          Column header designating whether the subject was focused on the task.
        notes (pandas dataframe, optional): Dataframe containing annotations about the data
    
    Returns:
        dataframe with blinks removed and replaced with interpolated values
    '''
    # Initialize Columns
    df_fixation[col_gaze_line ] = -1
    df_fixation[col_gaze_word ] = -1
    df_fixation[col_gaze_text ] = '.'
    df_fixation[col_gaze_color] = '.'
    df_fixation[col_gaze      ] = '.'

    # Define expected characteristics
    clusters_initial_x, clusters_initial_y = get_clusterLocations(name_stimuli)
    if clusters_initial_x is None:
        print('\t\tNo stimuli for this test')
        return df_fixation, None, (None, None)
    num_words = len(clusters_initial_x)
    num_lines = len(clusters_initial_y)
    
    # Trim the fixations to use, if indicated in notes
    df_fixation_use, df_fixation_skip = annotation.separate_stimuliProgress( notes, df_fixation, col_fix_time_start, col_fix_time_end)
    if len(df_fixation_use) == 0:
        print('\t\tNo usable fixations found')
        return df_fixation, None, (num_words, num_lines)

    # Find the newline thresholds
    # pos_fix_x = df_fixation_use[col_fix_pos_x].to_numpy().reshape(-1,1)
    pos_fix_max_x = df_fixation_use[col_fix_pos_x].nlargest( num_lines).iloc[-1] # .max() # np.max(pos_fix_x)
    pos_fix_min_x = df_fixation_use[col_fix_pos_x].nsmallest(num_lines).iloc[-1] # .min() # np.min(pos_fix_x)
    threshold_endline   = (2/3 * (pos_fix_max_x - pos_fix_min_x)) + pos_fix_min_x
    threshold_startline = (1/3 * (pos_fix_max_x - pos_fix_min_x)) + pos_fix_min_x
    maxTime_newline     = 2000 # milliseconds

    # Sort the values in time order
    df_fixation_use = df_fixation_use.sort_values(col_fix_time_start)

    # Define how to find newlines
    def _get_lineReset(df_fixation_use, maxTime_newline):
        df_fixation_use['line_reset'] = False
        for i, fixation_current in df_fixation_use.iterrows():
            if fixation_current[col_fix_pos_x] > threshold_endline:
                for j, fixation_compare in df_fixation_use.loc[i+1:].iterrows():
                    if fixation_compare[col_fix_pos_x] > threshold_endline:
                        break
                    if (fixation_compare[col_fix_time_start] - fixation_current[col_fix_time_end]) > maxTime_newline:
                        break
                    if fixation_compare[col_fix_pos_x] < threshold_startline:
                        # Success, we have found a line reset.
                        df_fixation_use.loc[i:j, 'line_reset'] = True
                        # import matplotlib.pyplot as plt
                        # plt.plot(df_fixation_use['line_reset'])
                        # plt.show()
                        break
        return df_fixation_use

    # Find the first iteration of newlines, and get start/stop index
    df_fixation_use = _get_lineReset(df_fixation_use, maxTime_newline)
    # After finding startStop index, We need to expand each startStop index forward and backwards to include the ends
    # Adding [False] above already increased the index by one to get the end, we just need to pull the start back
    identified_lines = get_indexStartStop( pd.concat(( pd.DataFrame([False]), ~df_fixation_use['line_reset'], pd.DataFrame([False]))).reset_index().loc[:,0])
    if len(identified_lines) == 0:
        identified_lines = [df_fixation_use.index[0], df_fixation_use.index[-1]]
    identified_lines = [ [start-2,end] for start, end in identified_lines]
    identified_lines[ 0][ 0] = 0
    identified_lines[-1][-1] = identified_lines[-1][-1] - 1

    num_iters = 0
    while (len(identified_lines) < num_lines) and (num_iters < 100):
        # Not all lines were distinguishable.
        maxTime_newline *= 1.1
        df_fixation_use = _get_lineReset(df_fixation_use, maxTime_newline)
        identified_lines = get_indexStartStop( pd.concat(( pd.DataFrame([False]), ~df_fixation_use['line_reset'], pd.DataFrame([False]))).reset_index().loc[:,0])
        if len(identified_lines) == 0:
            identified_lines = [df_fixation_use.index[0], df_fixation_use.index[-1]]
        identified_lines = [ [start-2,end] for start, end in identified_lines]
        identified_lines[ 0][ 0] = 0
        identified_lines[-1][-1] = identified_lines[-1][-1] - 1
        num_iters += 1

    num_iters = 0
    while (len(identified_lines) > num_lines) and (num_iters < 100):
        # More than the expected number of newlines were found. Combine Lines unitl only num_lines remaining
        # Combine the shortest line duration with its neighbor with the closest average y position
        min_duration     = df_fixation_use.iloc[-1][col_fix_time_end] - df_fixation_use.iloc[0][col_fix_time_start]
        index_to_combine = 0
        for i in range(len(identified_lines)):
            start_current, stop_current = identified_lines[i]
            stop_current = min( stop_current, len(df_fixation_use)-1)
            duration = df_fixation_use.iloc[stop_current][col_fix_time_end] - df_fixation_use.iloc[start_current][col_fix_time_end]
            
            if duration < min_duration:
                min_duration = duration

                if i > 0:
                    start_before,  stop_before  = identified_lines[i-1]
                    pos_y_before  = df_fixation_use.iloc[start_before :stop_before ].loc[:,col_fix_pos_y].mean()
                else:
                    pos_y_before  = 99999
                if i < len(identified_lines)-1:
                    start_after,   stop_after   = identified_lines[i+1]
                    pos_y_after   = df_fixation_use.iloc[start_after  :stop_after  ].loc[:,col_fix_pos_y].mean()
                else:
                    pos_y_after  = 99999

                pos_y_current = df_fixation_use.iloc[start_current:stop_current].loc[:,col_fix_pos_y].mean()
                if abs(pos_y_after - pos_y_current) < abs(pos_y_before - pos_y_current) :
                    index_to_combine = i
                else:
                    index_to_combine = i - 1

        # Combine lines
        identified_lines[index_to_combine][-1] = identified_lines[index_to_combine+1][-1]
        identified_lines.pop(index_to_combine+1)
        num_iters += 1


    # import matplotlib.pyplot as plt
    # plt.figure()
    # plot_initial_x = [x_loc for y_loc in clusters_initial_y for x_loc in clusters_initial_x]
    # plot_initial_y = [y_loc for y_loc in clusters_initial_y for x_loc in clusters_initial_x]
    # plt.plot(plot_initial_x, plot_initial_y, 'c.', markersize=12, alpha=0.2, label='"True" loc')
    # plt.plot(df_fixation_use[col_fix_pos_x], df_fixation_use[col_fix_pos_y], 'b.', markersize=8, label='fixation')
    # plt.xlim((0,1920))
    # plt.ylim((1080, 0)) # Invert Y axis
    # plt.legend()
    # # plt.show()

    # plt.figure()
    # plt.plot(df_fixation_use[col_fix_time_start], df_fixation_use[col_fix_pos_x], 'b.', markersize=8, label='fixation')
    # plt.plot(df_fixation_use.iloc[identified_lines[0][1]-1:identified_lines[1][0]+1][col_fix_time_start], df_fixation_use.iloc[identified_lines[0][1]-1:identified_lines[1][0]+1][col_fix_pos_x], 'g', label='newline')
    # for i in range(1, len(identified_lines)-1):
    #     plt.plot(df_fixation_use.iloc[identified_lines[i][1]-1:identified_lines[i+1][0]+1][col_fix_time_start], df_fixation_use.iloc[identified_lines[i][1]-1:identified_lines[i+1][0]+1][col_fix_pos_x], 'g')
    
    # plt.figure()
    # plot_initial_x = [x_loc for y_loc in clusters_initial_y for x_loc in clusters_initial_x]
    # plot_initial_y = [y_loc for y_loc in clusters_initial_y for x_loc in clusters_initial_x]
    # plt.plot(plot_initial_x, plot_initial_y, 'c.', markersize=12, alpha=0.2, label='"True" loc')
    # plt.plot(df_fixation_use[col_fix_pos_x], df_fixation_use[col_fix_pos_y], 'b.', markersize=8, label='fixation')
    # plt.plot(df_fixation_use.iloc[identified_lines[0][1]-1:identified_lines[1][0]+1][col_fix_pos_x], df_fixation_use.iloc[identified_lines[0][1]-1:identified_lines[1][0]+1][col_fix_pos_y], 'g', label='newline')
    # for i in range(1, len(identified_lines)-1):
    #     plt.plot(df_fixation_use.iloc[identified_lines[i][1]-1:identified_lines[i+1][0]+1][col_fix_pos_x], df_fixation_use.iloc[identified_lines[i][1]-1:identified_lines[i+1][0]+1][col_fix_pos_y], 'g')
    # plt.xlim((0,1920))
    # plt.ylim((1080, 0)) # Invert Y axis
    # plt.legend()
    # plt.show()
    
    if len(identified_lines) != num_lines:
        print('\t\tIncorrect number of lines identified')
        return df_fixation, None, (num_words, num_lines)

    # For each isolated line, find num_words word within it
    pos_word = []
    prevLine_index_stop = 0
    for gaze_line, (index_start, index_stop) in enumerate(identified_lines):
        df_fixation_line = df_fixation_use.iloc[index_start:index_stop]

        # Find the inital guess for word locations
        kmeans_weight_y = df_fixation_line[col_fix_duration].to_numpy()
        pos_fix_x       = df_fixation_line[col_fix_pos_x].to_numpy().reshape(-1,1)
        pos_fix_y       = df_fixation_line[col_fix_pos_y].to_numpy().reshape(-1,1)
        separation_x    = (pos_fix_x.max() - pos_fix_x.min()) / num_words
        pos_first_x     = pos_fix_x.min() + (separation_x / 2)
        clusters_initial_x = np.array([pos_first_x + l * separation_x for l in range(num_words)]).reshape(-1,1)
        clusters_initial_y = np.repeat(pos_fix_y.mean(), clusters_initial_x.shape[0], axis=0).reshape(-1,1)

        # Use your initial guess to run a K-Means clustering to find the best word locations in this line..
        clusters_labels  = list(range(num_words))
        clusters_initial = np.hstack([clusters_initial_x, clusters_initial_y])
        pos_fix          = np.hstack([         pos_fix_x,          pos_fix_y])
        if pos_fix.shape[0] < num_words:
            print('\t\tIncorrect number of words identified')
            return df_fixation, None, (num_words, num_lines)
        try:
            cluster_stimuli = KMeans(n_clusters=num_words, init=clusters_initial, n_init=1).fit(pos_fix, clusters_labels, sample_weight=kmeans_weight_y)
        except Exception as e:
            print('\tERROR Finding KMeans Clusters')
            print(e)
            return df_fixation, None, (num_words, num_lines)
        pos_word.append( cluster_stimuli.cluster_centers_)

        # Label each stimuli for this line with a word index. Include stimuli from after the previous line
        # Some stimuli may exist "in between lines". These should not be used for clustering, but still need to be labeled.
        fixations_update = df_fixation_use.iloc[prevLine_index_stop:index_stop].index
        try:
            df_fixation_use.loc[fixations_update,col_gaze_word] = cluster_stimuli.predict(df_fixation_use.iloc[prevLine_index_stop:index_stop].loc[:,(col_fix_pos_x, col_fix_pos_y)].astype(np.float64).values).reshape(-1,1)
        except:
            df_fixation_use.loc[fixations_update,col_gaze_word] = cluster_stimuli.predict(df_fixation_use.iloc[prevLine_index_stop:index_stop].loc[:,(col_fix_pos_x, col_fix_pos_y)].astype(np.float32).values).reshape(-1,1)
        df_fixation_use.loc[fixations_update,col_gaze_line] = gaze_line

        prevLine_index_stop = index_stop

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(df_fixation_line[col_fix_pos_x], df_fixation_line[col_fix_pos_y], 'b.', label='Fixation')
        # # plt.plot(df_fixation_line.loc['timestamp_start']/1000, df_fixation_line.loc[col_fix_pos_y], 'b.', label='Fixation')
        # plt.title('Fixations in Line ' + str(gaze_line))
        # plt.xlabel('pixels')
        # plt.ylabel('pixels')
        # plt.xlim((0,1920))
        # plt.ylim((1080, 0)) # Invert Y axis
        # plt.legend()

        # plt.figure()
        # # plot_initial_x = [x_loc for y_loc in clusters_initial_y for x_loc in clusters_initial_x]
        # # plot_initial_y = [y_loc for y_loc in clusters_initial_y for x_loc in clusters_initial_x]
        # plt.plot(plot_initial_x, plot_initial_y, 'c.', markersize=12, alpha=0.2, label='"True" loc')
        # plt.plot(pos_fix_x, pos_fix_y, 'b.', label='Fixation')
        # plt.plot(pos_word[-1][:,0], pos_word[-1][:,1], 'r^', label='Centroid')
        # plt.title('Fixations in Line ' + str(gaze_line))
        # plt.xlabel('pixels')
        # plt.ylabel('pixels')
        # plt.xlim((0,1920))
        # plt.ylim((1080, 0)) # Invert Y axis
        # plt.legend()
        # plt.show()

    # Now that we found initial positions for each word on each line, run one final clustering run to fine-tune positions.
    # Note labels and position axis are switched the row index (word) and x index (screen) are different
    cluster_labels   = list(range(num_lines*num_words))
    clusters_stimuli = np.array([pos_word[l][w] for l in range(num_lines) for w in range(num_words)])
    cluster_final    = skl_knn(n_neighbors=1).fit(clusters_stimuli, cluster_labels)

    # Identify where the subject is "reading"
    # Starts when they focus on the first word and ends when the look away from the last word
    df_fixation_use[col_focus] = False
    index_focus_start = df_fixation_use[ (df_fixation_use[col_gaze_line] ==           0) & (df_fixation_use[col_gaze_word] ==           0)].index[0]
    index_focus_end   = df_fixation_use[ (df_fixation_use[col_gaze_line] == num_lines-1) & (df_fixation_use[col_gaze_word] == num_words-1)].index[-1]
    df_fixation_use.loc[index_focus_start:index_focus_end+1, col_focus] = True

    # Update any ignored stimuli
    if len(df_fixation_skip) > 0:
        missing_labels = cluster_final.predict( df_fixation_skip.loc[:,(col_fix_pos_x, col_fix_pos_y)].astype(np.float64).values)
        df_fixation_skip.loc[:,col_gaze_word] = np.remainder(    missing_labels, num_words)
        df_fixation_skip.loc[:,col_gaze_line] = np.floor_divide( missing_labels, num_lines)
        df_fixation_skip[col_focus]    = False
        df_fixation_skip['line_reset'] = False
        df_fixation = pd.concat([df_fixation_skip, df_fixation_use], sort=False)
    else:
        df_fixation = df_fixation_use

    return df_fixation, cluster_final, (num_words, num_lines)



def clean_wordTokens(df_wordTokens, col_token, col_repeat, col_time, col_delay, col_inVocab, repeated_minTime=100):
    '''This function removes tokens that are repeated less then `repeated_minTime` milliseconds apart. 
    These occurences are usually an artifact of the token extraction framework, and are not real

    Args:
        df_wordTokens (pandas dataframe):   Dataframe containing raw information about word tokens
        col_token (str):                    Column header deisgnating the word token.
        col_repeat (str)                    Column header designating whether a word token is a repeat of the previous token.
        col_time (str):                     Column header designating the word start timestamp.
        col_delay (str):                    Column header designating what to call the column containing the word delay
        col_inVocab (str):                  Column header designating whether the word token is within the expected vocabulary.
        repeated_minTime (float, optional): Minimum expected delay time between tokens (milliseconds)

    Returns:
        data_wordCorrect dataframe containing word tokens without repeats.
    '''
    # Calculate the delay between tokens
    df_wordTokens[col_delay] = df_wordTokens[col_time].diff()

    # Mark repeated tokens
    df_wordTokens[col_repeat] = False
    i_prev = df_wordTokens.index[0]
    for i in df_wordTokens.index[1:]:
        if df_wordTokens.loc[i_prev, col_token] == df_wordTokens.loc[i, col_token]:
            df_wordTokens.loc[i,col_repeat] = True
        i_prev = i

    # Clean impossibly fast repeated tokens - likely a transcription error
    data_wordCorrect = df_wordTokens.copy().drop(df_wordTokens[(df_wordTokens[col_delay] < repeated_minTime) & df_wordTokens[col_repeat]].index)
    # Clean out of vocabulary words
    data_wordCorrect = data_wordCorrect.drop(data_wordCorrect[ ~data_wordCorrect[col_inVocab]].index)

    return data_wordCorrect



def get_wordCorrectSequence( data_wordCorrect, stimuli, col_token, col_wordIndex):
    '''Determine whether the word tokens are correct and occur in the expected sequence.
    
    Args:
        data_wordCorrect (pandas dataframe): Dataframe containing cleaned and verified word tokens.
        stimuli (iterable):                  Contains the expected correct stimuli sequence
        col_token (iterable):                Contains the expected detected token sequence
        col_wordIndex (str):                 Column header designating where to place the index of correct words. Incorrect words will be left as Nan. 

    Returns:
        data_wordCorrect with correct tokens identified. Incorrecte tokens will have nan values in `col_wordIndex`
    '''
    
    # Identify correctly spoken words, by interating forward through the tokens
    data_wordCorrect['correct_forward'] = np.nan
    index_correct = data_wordCorrect.columns.get_loc('correct_forward')
    index_word = 0
    for i_truth, truth in enumerate(stimuli):
        while index_word < len(data_wordCorrect):
            tokenFirst = data_wordCorrect.iloc[index_word][col_token]
            if tokenFirst.startswith( truth[0]):
                data_wordCorrect.iloc[index_word,index_correct] = i_truth
                break
            if truth[1].startswith(tokenFirst):
                data_wordCorrect.iloc[index_word,index_correct] = -1 # Negative one indicates they said word not color
            index_word += 1
        index_word += 1

    # Identify correctly spoken words, by interating backward through the tokens
    data_wordCorrect['correct_backward'] = np.nan
    index_correct = data_wordCorrect.columns.get_loc('correct_backward')
    index_word = len(data_wordCorrect) - 1
    for i_truth, truth in reversed(list(enumerate(stimuli))):
        while index_word >= 0:
            tokenFirst = data_wordCorrect.iloc[index_word][col_token]
            if tokenFirst.startswith(truth[0]):
                data_wordCorrect.iloc[index_word,index_correct] = i_truth
                break
            if truth[1].startswith(tokenFirst):
                data_wordCorrect.iloc[index_word,index_correct] = -1
                break
            index_word -= 1
        index_word -= 1

    # Tokens which match the forward and backwards progression are confidently correct.
    index_matching = data_wordCorrect[ data_wordCorrect['correct_forward'] == data_wordCorrect['correct_backward']].index
    data_wordCorrect[col_wordIndex] = np.nan
    data_wordCorrect.loc[index_matching,col_wordIndex] = data_wordCorrect.loc[index_matching,'correct_forward']

    return data_wordCorrect



def get_stimuliFixation_desc( df_info, name_stimuli, num_words, num_lines, col_gaze_line, col_gaze_word, col_gaze_text, col_gaze_color, col_gaze, col_gaze_index):
    '''Using exiting gaze_line and gaze_word values, adds descriptions of "text", "color", and a combined unique identifier, to each line of df_info.

    Args:
        df_info (pandas dataframe): Contains raw data for a given trial
        name_stimuli (str):         Name of the target stimuli to use when identifying fixations. String must be present in :py:func:`~eyetracking.functions.process.get_stimuliDescriptions`
        num_words (int):            Number of words in a line
        num_lines (int):            Number of lines in the trial
        col_gaze_line (str):        Column header in where the gaze line index can be found.
        col_gaze_word (str):        Column header in where the gaze word index can be found.
        col_gaze_text (str):        Desired column header in where the gaze text content should be placed.
        col_gaze_color (str):       Desired column header in where the gaze text color should be placed.
        col_gaze (str):             Desired column header in where the gaze identifier should be placed.
        col_gaze_index (str):       Column header in where the gaze word index can be found.

    Returns:
        df_info (passed dataframe with col_gaze_text, col_gaze_color, and col_gaze added), stimuli (list, stimuli descriptions returned for convenience)
    '''
    # Add the rest of the labels
    stimuli = get_stimuliDescriptions(name_stimuli)
    if len(stimuli) > 0:
        labels  = df_info[col_gaze_line]*num_words + df_info[col_gaze_word]
        df_info.loc[:,col_gaze_color] = stimuli[labels,0]
        df_info.loc[:,col_gaze_text ] = stimuli[labels,1]
        df_info.loc[:,col_gaze      ] = df_info.apply(lambda r: ','.join((str(r[col_gaze_word]), str(r[col_gaze_line]), r[col_gaze_text], r[col_gaze_color])), axis=1)
        df_info.loc[:,col_gaze_index] = labels

    return df_info, stimuli



def add_eventStimuli(df_event, df_fixation, col_timestamp_start, col_timestamp_end, col_gaze, col_gaze_line, col_gaze_word, col_gaze_color, col_gaze_text, col_gaze_index, col_focus):
    '''Using a dataframe describing some event, update existing gaze_line, gaze_word, gaze_text, gaze_color, and gaze_identifier values to identify start and stop gaze information.
    Adds ``gaze_*_start`` and ``gaze_*_end`` columns to each event row, designating the starting and ending gaze stimuli, respectively.

    ``col_focus`` and ``gaze_same`` columns are also added to indicate whether the subject is focused on the task, and whether the event gaze stays the same.

    Args:
        df_event (pandas dataframe):    Contains processed data for an event of interest (saccades, fixations, blinks, etc)
        df_fixation (pandas dataframe): Contains corresponding fixation data from which gaze fixation locations can be extracted.
        col_timestamp_start (str):      Column header in where the timestamp for the event start can be found.
        col_timestamp_end (str):        Column header in where the timestamp for the event end can be found.
        col_gaze (str):                 Column header in where the gaze identifier can be found.
        col_gaze_line (str):            Column header in where the gaze line index can be found.
        col_gaze_word (str):            Column header in where the gaze word index can be found.
        col_gaze_color (str):           Column header in where the gaze text color can be found.
        col_gaze_text (str):            Column header in where the gaze text content can be found.
        col_gaze_index (str):           Column header in where the gaze word index can be found.
        col_focus (str):                Column header in where the focus status of the subject can be found.

    Returns:
        df_event (passed dataframe with col_gaze_line_start, col_gaze_word_start, col_gaze_text_start, col_gaze_color_start, col_gaze_start, col_gaze_line_end, col_gaze_word_end, col_gaze_text_end, col_gaze_color_end, col_gaze_end, col_focus, 'gaze_same' columns added
    '''
    col_gaze_line_start  = col_gaze_line + '_start'
    col_gaze_word_start  = col_gaze_word + '_start'
    col_gaze_text_start  = col_gaze_text + '_start'
    col_gaze_color_start = col_gaze_color+ '_start'
    col_gaze_start       = col_gaze      + '_start'
    col_gaze_index_start = col_gaze_index+ '_start'
    col_gaze_line_end    = col_gaze_line + '_end'
    col_gaze_word_end    = col_gaze_word + '_end'
    col_gaze_text_end    = col_gaze_text + '_end'
    col_gaze_color_end   = col_gaze_color+ '_end'
    col_gaze_end         = col_gaze      + '_end'
    col_gaze_index_end   = col_gaze_index+ '_end'
    
    df_event[col_gaze_line_start ] = -1
    df_event[col_gaze_word_start ] = -1
    df_event[col_gaze_text_start ] = '.'
    df_event[col_gaze_color_start] = '.'
    df_event[col_gaze_start      ] = '.'
    df_event[col_gaze_index_start] = -1
    df_event[col_gaze_line_end   ] = -1
    df_event[col_gaze_word_end   ] = -1
    df_event[col_gaze_text_end   ] = '.'
    df_event[col_gaze_color_end  ] = '.'
    df_event[col_gaze_end        ] = '.'
    df_event[col_gaze_index_end  ] = -1
    df_event[col_focus           ] = False
    df_event['gaze_same'         ] = False

    for index, row in df_event.iterrows():
        fixation_start = df_fixation[ row[col_timestamp_start] == df_fixation[col_timestamp_end  ]]
        if index != 0:
            if len(fixation_start) == 0:
                continue
            df_event.loc[index, col_gaze_start      ] = fixation_start[col_gaze      ].values
            df_event.loc[index, col_gaze_line_start ] = fixation_start[col_gaze_line ].values
            df_event.loc[index, col_gaze_word_start ] = fixation_start[col_gaze_word ].values
            df_event.loc[index, col_gaze_text_start ] = fixation_start[col_gaze_text ].values
            df_event.loc[index, col_gaze_color_start] = fixation_start[col_gaze_color].values
            df_event.loc[index, col_gaze_index_start] = fixation_start[col_gaze_index].values

        fixation_end   = df_fixation[ row[col_timestamp_end  ] == df_fixation[col_timestamp_start]]
        if index != df_event.shape[0]-1:
            if len(fixation_end) == 0:
                continue
            df_event.loc[index, col_gaze_end        ] = fixation_end[col_gaze      ].values
            df_event.loc[index, col_gaze_line_end   ] = fixation_end[col_gaze_line ].values
            df_event.loc[index, col_gaze_word_end   ] = fixation_end[col_gaze_word ].values
            df_event.loc[index, col_gaze_text_end   ] = fixation_end[col_gaze_text ].values
            df_event.loc[index, col_gaze_color_end  ] = fixation_end[col_gaze_color].values
            df_event.loc[index, col_gaze_index_end  ] = fixation_end[col_gaze_index].values

        if fixation_start[col_focus].values & fixation_end[col_focus].values:
            df_event.loc[index, col_focus] = True
        if fixation_start[col_gaze].values == fixation_end[col_gaze].values:
            df_event.loc[index, 'gaze_same'] = True

    return df_event



def add_trialStimuli( df_trial, list_df_event, list_df_event_transition, col_timestamp, col_timestamp_start, col_timestamp_end, col_gaze, col_gaze_line, col_gaze_word, col_gaze_text, col_gaze_color, col_gaze_index, col_focus):
    '''Using a dataframe describing raw data for a whole trial, update existing gaze_line, gaze_word, gaze_text, gaze_color, and gaze_identifier values to identify start and stop gaze information.
    Adds ``gaze_*_start`` and ``gaze_*_end`` columns to each event row, designating the starting and ending gaze stimuli, respectively.

    ``col_focus`` and ``transition`` columns are also added to indicate whether the subject is focused on the task, and whether the event gaze stays the same.

    Args:
        df_trial (pandas dataframe):        Contains processed data for an event of interest (saccades, fixations, blinks, etc)
        list_df_event (list of dataframes): list of dataframes static events to be annotated in the raw data.
        list_df_event_transition (list of dataframes): list of dataframes containing transition events to be annotated in the raw data.
        col_timestamp (str):                Column header in where the the sample timestamp can be found.
        col_timestamp_start (str):          Column header in where the timestamp for the event start can be found.
        col_timestamp_end (str):            Column header in where the timestamp for the event end can be found.
        col_gaze (str):                     Column header in where the gaze identifier can be found.
        col_gaze_line (str):                Column header in where the gaze line index can be found.
        col_gaze_word (str):                Column header in where the gaze word index can be found.
        col_gaze_text (str):                Column header in where the gaze text content can be found.
        col_gaze_color (str):               Column header in where the gaze text color can be found.
        col_gaze_index (str):               Column header in where the gaze word index can be found.
        col_focus (str):                    Column header in where the focus status of the subject can be found.

    Returns:
        df_trial (passed dataframe with col_gaze_line_start, col_gaze_word_start, col_gaze_text_start, col_gaze_color_start, col_gaze_start, col_gaze_line_end, col_gaze_word_end, col_gaze_text_end, col_gaze_color_end, col_gaze_end, col_focus, 'transition' columns added
    '''
    if not isinstance(list_df_event, list):
        list_df_event = [list_df_event]
    if not isinstance(list_df_event_transition, list):
        list_df_event_transition = [list_df_event_transition]

    col_gaze_line_start  = col_gaze_line + '_start'
    col_gaze_word_start  = col_gaze_word + '_start'
    col_gaze_text_start  = col_gaze_text + '_start'
    col_gaze_color_start = col_gaze_color+ '_start'
    col_gaze_start       = col_gaze      + '_start'
    col_gaze_index_start = col_gaze_index+ '_start'
    col_gaze_line_end    = col_gaze_line + '_end'
    col_gaze_word_end    = col_gaze_word + '_end'
    col_gaze_text_end    = col_gaze_text + '_end'
    col_gaze_color_end   = col_gaze_color+ '_end'
    col_gaze_end         = col_gaze      + '_end'
    col_gaze_index_end   = col_gaze_index+ '_end'

    df_trial[col_gaze_line ] = -1
    df_trial[col_gaze_word ] = -1
    df_trial[col_gaze_text ] = '.'
    df_trial[col_gaze_color] = '.'
    df_trial[col_gaze      ] = '.'
    df_trial[col_gaze_index] = -1
    df_trial[col_focus     ] = False
    df_trial['transition'  ] = False

    for df_data in list_df_event_transition:
        for _, row in df_data.iterrows():
            time_isolated = (df_trial[col_timestamp] > row[col_timestamp_start]) & (df_trial[col_timestamp] < row[col_timestamp_end])

            if row['gaze_same']:
                df_trial.loc[time_isolated,col_gaze      ] = row[col_gaze_start]
                df_trial.loc[time_isolated,col_gaze_line ] = row[col_gaze_line_start]
                df_trial.loc[time_isolated,col_gaze_word ] = row[col_gaze_word_start]
                df_trial.loc[time_isolated,col_gaze_text ] = row[col_gaze_text_start]
                df_trial.loc[time_isolated,col_gaze_color] = row[col_gaze_color_start]
                df_trial.loc[time_isolated,col_gaze_index] = row[col_gaze_index_start]
            else:
                df_trial.loc[time_isolated,col_gaze      ] =  row[col_gaze_start      ] + '+' + row[col_gaze_end      ]
                df_trial.loc[time_isolated,col_gaze_line ] = (row[col_gaze_line_start ]    +    row[col_gaze_line_end ]) / 2
                df_trial.loc[time_isolated,col_gaze_word ] = (row[col_gaze_word_start ]    +    row[col_gaze_word_end ]) / 2
                df_trial.loc[time_isolated,col_gaze_text ] =  row[col_gaze_text_start ] + '+' + row[col_gaze_text_end ]
                df_trial.loc[time_isolated,col_gaze_color] =  row[col_gaze_color_start] + '+' + row[col_gaze_color_end]
                df_trial.loc[time_isolated,col_gaze_index] = (row[col_gaze_index_start]    +    row[col_gaze_index_end]) / 2
                df_trial.loc[time_isolated,'transition']   = True
            
            if row[col_focus]:
                df_trial.loc[time_isolated,col_focus      ] = row[col_focus]

    for df_data in list_df_event:
        for _, row in df_data.iterrows():
            time_isolated = (df_trial[col_timestamp] >= row[col_timestamp_start]) & (df_trial[col_timestamp] <= row[col_timestamp_end])

            df_trial.loc[time_isolated,col_gaze      ] = row[col_gaze]
            df_trial.loc[time_isolated,col_gaze_line ] = row[col_gaze_line]
            df_trial.loc[time_isolated,col_gaze_word ] = row[col_gaze_word]
            df_trial.loc[time_isolated,col_gaze_text ] = row[col_gaze_text]
            df_trial.loc[time_isolated,col_gaze_color] = row[col_gaze_color]
            df_trial.loc[time_isolated,col_gaze_index] = row[col_gaze_index]
    
    return df_trial
            


def add_stimuliFromCluster(df_data, name_stimuli, cluster_predict, words_per_line, col_pos_x, col_pos_y, col_gaze, col_gaze_line, col_gaze_word, col_gaze_color, col_gaze_text, col_exclude=None):
    '''Uses a sklearn prediciton function to predict gaze stimuli and annotate that prediction in the dataframe.

    Args:
        df_data (pandas dataframe):          Dataframe containing raw fixation data
        name_stimuli (str):                  Name of the target stimuli to use when identifying fixations. String must be present in :py:func:`~eyetracking.functions.process.get_stimuliDescriptions`
        cluster_predict (sklearn predictor): Object with `.predict` method used to predict the stimuli group using x and y values.
        words_per_line (int):                Number of expected words per line.
        col_pos_x (str):                     Column header in where the gaze x position can be found.
        col_pos_y (str):                     Column header in where the gaze x position can be found.
        col_gaze (str):                      Column header in where the gaze identifier can be found.
        col_gaze_line (str):                 Column header in where the gaze line index can be found.
        col_gaze_word (str):                 Column header in where the gaze word index can be found.
        col_gaze_color (str):                Column header in where the gaze text color can be found.
        col_gaze_text (str):                 Column header in where the gaze text content can be found.
        col_exclude (str, optional):         Column header which designates dataframe rows to exclude from stimuli assignment.

    Returns:

    '''
    df_data = df_data.copy()
    if len(df_data) == 0:
        df_data[col_gaze_word ] = None
        df_data[col_gaze_line ] = None
        df_data[col_gaze_text ] = None
        df_data[col_gaze_color] = None
        df_data[col_gaze      ] = None
        return df_data, get_stimuliDescriptions(name_stimuli)

    rows_ignore_x = np.isnan(df_data.loc[ : , col_pos_x]) 
    rows_ignore_y = np.isnan(df_data.loc[ : , col_pos_x])
    df_data.loc[rows_ignore_x, col_pos_x] = 0
    df_data.loc[rows_ignore_y, col_pos_y] = 0

    df_data.rename(columns={col_pos_x:'cluster-kmeans_predict_x', col_pos_y:'cluster-kmeans_predict_y'}, inplace=True)
    labels = cluster_predict.predict(df_data.loc[ : , ('cluster-kmeans_predict_x', 'cluster-kmeans_predict_y')].astype(np.float64).values)
    df_data.rename(columns={'cluster-kmeans_predict_x':col_pos_x, 'cluster-kmeans_predict_y':col_pos_y}, inplace=True)

    stimuli = get_stimuliDescriptions(name_stimuli)
    df_data[col_gaze_word ] = labels % words_per_line
    df_data[col_gaze_line ] = labels // words_per_line
    df_data[col_gaze_color] = stimuli[labels,0]
    df_data[col_gaze_text ] = stimuli[labels,1]
    df_data[col_gaze      ] = df_data.apply(lambda r: ','.join((str(r[col_gaze_word]), str(r[col_gaze_line]), r[col_gaze_text], r[col_gaze_color])), axis=1)

    df_data.loc[ (rows_ignore_x | rows_ignore_y), [col_gaze     , col_gaze_color, col_gaze_text]] = '.'
    df_data.loc[ (rows_ignore_x | rows_ignore_y), [col_gaze_line,  col_gaze_word               ]] =  -1
    if col_exclude is not None:
        df_data.loc[ np.any( df_data.loc[:,col_exclude], axis=1), [col_gaze     , col_gaze_color, col_gaze_text]] = '.'
        df_data.loc[ np.any( df_data.loc[:,col_exclude], axis=1), [col_gaze_line,  col_gaze_word               ]] =  -1

    return df_data, stimuli



def trim_wordTrial( df_data, col_timestamp, col_timestamp_start, col_timestamp_end, column_line, column_word, events_avoid=[], df_summary_trim=[]):
    '''Trims the passed in data to START at the LAST instance the subject looks at the FIRST word, 
    and will also END at the FIRST instance the subject looks at the LAST word.
    
    Args:
        df_data (pandas dataframe):           Pandas dataframe containing all of the relevant raw data
        col_timestamp (str):                  Column header in where the the sample timestamp can be found.
        col_timestamp_start (str):            Column header in where the timestamp for the event start can be found.
        col_timestamp_end (str):              Column header in where the timestamp for the event end can be found.
        column_line (str):                    Column header designating the word index that the subject is looking at.
        column_word (str):                    Column header designating the line index that the subject is looking at.
        events_avoid (list of str, optional): List of boolean columns designating where to avoid trimming the data.
        df_summary_trim (list or pandas dataframe, optional): Pandas dataframe(s) containing summary/event information. Will trim stats to those occuring during trial.
        
    Returns:
        *trial_start* - index of trial start, *trial_end* - index of trial end, *df_data* - raw data which has been trimmed, *list_of_df_summary* - A list of the summary dataframes passed in, trimmed accordingly.
    '''
    # End the trial the last time they look at the last word
    trial_end = df_data[ (df_data[column_line] == df_data[column_line].unique().max()) & (df_data[column_word] == df_data[column_word].unique().max()) ].index
    if len(trial_end) > 0:
        # save the first time they look at the last word
        trial_end = trial_end[0]
    else:
        # otherwise, save the end of the trial
        trial_end = df_data.index[-1]

    # Don't end in the middle of a saccade
    while ( any( df_data.loc[trial_end, events_avoid])):
        trial_end += 1
        if trial_end >= df_data.index[-1]-2:
            break
    trial_end += 1
    # Start the trial the last time they look at the first word, before the end of the trial
    trial_startTEMP = df_data[ (df_data[column_line] == 0) & (df_data[column_word] == 0) ].index
    trial_start = trial_startTEMP[ trial_startTEMP < trial_end]
    if len(trial_start) > 0:
        # save the last time they look at the first word
        trial_start = trial_start[-1]
    else:
        # otherwise, save the beginning of the recording
        trial_start = df_data.index[0]
    # Don't start in the middle of a saccade or blink
    while ( any( df_data.loc[trial_start, events_avoid])):
        trial_start -= 1
        if trial_start <= df_data.index[0]+1:
            break
    trial_start -= 1
    # Trim the dataframe to just the desired indicies, and reindex the array
    df_data = df_data.loc[trial_start:trial_end, :]

    if not isinstance(df_summary_trim, list):
        df_summary_trim = [df_summary_trim]
    df_trimmed = [ df.loc[ (df[col_timestamp_start] > df_data.iloc[ 0][col_timestamp]) & df[col_timestamp_end] < df_data.iloc[-1][col_timestamp]] for df in df_summary_trim]

    return trial_start, trial_end, df_data, df_trimmed



def get_tokenVocabulary(name_trial):
    '''Get expected vocabulary to be included in a trial.

    Currently implemented trials:
    
        * stroop
        * colors_preliminary

    Args:
        name_trial (str): Name of the trial to extract descriptions for.

    Returns:
        stimuli (np_array containing stimuli descriptions)
    '''
    stimuli = []

    if 'stroop' in name_trial.strip().lower() or 'colors_preliminary' in name_trial.strip().lower():
        stimuli.append(['▁RE'])
        stimuli.append(['▁B'])
        stimuli.append(['▁G'])

    return np.array(stimuli)



def get_stimuliDescriptions(name_trial):
    '''Get descriptor information of a trial's stimuli.

    Currently implemented trials:
    
        * stroop
        * colors_preliminary

    Args:
        name_trial (str): Name of the trial to extract descriptions for.

    Returns:
        stimuli (np_array containing stimuli descriptions)
    '''
    stimuli = []

    trial = name_trial.strip().lower()
    if (trial == 'stroop') or (trial == 'stroop_onlycolors') or (trial == 'colors_preliminary2'):
        stimuli.append(['r','b']) # line 0 word 0
        stimuli.append(['b','r']) # line 0 word 1
        stimuli.append(['r','g']) # line 0 word 2
        stimuli.append(['g','b']) # line 0 word 3
        stimuli.append(['b','g']) # line 1 word 0
        stimuli.append(['g','b']) # line 1 word 1
        stimuli.append(['b','r']) # line 1 word 2
        stimuli.append(['b','g']) # line 1 word 3
        stimuli.append(['g','r']) # line 2 word 0
        stimuli.append(['r','g']) # line 2 word 1
        stimuli.append(['g','b']) # line 2 word 2
        stimuli.append(['r','b']) # line 2 word 3
        stimuli.append(['r','b']) # line 3 word 0
        stimuli.append(['b','r']) # line 3 word 1
        stimuli.append(['b','g']) # line 3 word 2
        stimuli.append(['g','r']) # line 3 word 3
        stimuli.append(['b','r']) # line 4 word 0
        stimuli.append(['r','g']) # line 4 word 1
        stimuli.append(['g','r']) # line 4 word 2
        stimuli.append(['r','g']) # line 4 word 3
        stimuli.append(['r','g']) # line 5 word 0
        stimuli.append(['g','b']) # line 5 word 1
        stimuli.append(['r','b']) # line 5 word 2
        stimuli.append(['b','r']) # line 5 word 3

    elif (trial == 'stroop_onlytext') or (trial == 'colors_preliminary1'):
        stimuli.append(['b','k']) # line 0 word 0
        stimuli.append(['g','k']) # line 0 word 1
        stimuli.append(['b','k']) # line 0 word 2
        stimuli.append(['g','k']) # line 0 word 3
        stimuli.append(['r','k']) # line 1 word 0
        stimuli.append(['b','k']) # line 1 word 1
        stimuli.append(['g','k']) # line 1 word 2
        stimuli.append(['r','k']) # line 1 word 3
        stimuli.append(['g','k']) # line 2 word 0
        stimuli.append(['r','k']) # line 2 word 1
        stimuli.append(['r','k']) # line 2 word 2
        stimuli.append(['b','k']) # line 2 word 3
        stimuli.append(['r','k']) # line 3 word 0
        stimuli.append(['b','k']) # line 3 word 1
        stimuli.append(['g','k']) # line 3 word 2
        stimuli.append(['r','k']) # line 3 word 3
        stimuli.append(['g','k']) # line 4 word 0
        stimuli.append(['g','k']) # line 4 word 1
        stimuli.append(['r','k']) # line 4 word 2
        stimuli.append(['g','k']) # line 4 word 3
        stimuli.append(['b','k']) # line 5 word 0
        stimuli.append(['r','k']) # line 5 word 1
        stimuli.append(['b','k']) # line 5 word 2
        stimuli.append(['b','k']) # line 5 word 3

    return np.array(stimuli)



def get_clusterLocations(name_trial):
    '''Get true cluster locations (pixels) for a trial of interest

    Currently implemented trials:
    
        * stroop
        * colors_preliminary
    
    Args:
        name_trial (str): Name of the trial to extract descriptions for.

    Returns:
        stimuli_x (np_array containing stimuli description locations in x), stimuli_y (np_array containing stimuli locations in y)
    '''
    stimuli_x = []
    stimuli_y = []

    if 'stroop' in name_trial.strip().lower() or 'colors_preliminary' in name_trial.strip().lower():
        # Initial locations of word columns
        stimuli_x.append(  355)
        stimuli_x.append(  813)
        stimuli_x.append( 1265)
        stimuli_x.append( 1720)
        # Initial locations of word rows
        stimuli_y.append(  270)
        stimuli_y.append(  409)
        stimuli_y.append(  569)
        stimuli_y.append(  729)
        stimuli_y.append(  890)
        stimuli_y.append( 1051)
    
        return np.array(stimuli_x).reshape(-1,1), np.array(stimuli_y).reshape(-1,1)
    
    return None, None



# def check_ARresidual(df, columns=["pos_x", "pos_y", "vel_x", "vel_y"], data_label=None, threshold=100, save_path='./', filename='residual'):
#     res_points = 0
#     res_value = 0

#     found_good = True
#     found_okay = True
#     for col in columns:
#         if data_label is None:
#             data_suffix = ''
#         elif not data_label.startswith('_'):
#             data_suffix = '_' + data_label
#         else:
#             data_suffix = data_label
#         col_name = col + data_suffix

#         data  = np.array(df.loc[:,col_name].fillna(method='ffill').fillna(method='bfill'))
#         if np.isnan(data).all():
#             continue
#         abs_change = np.abs(np.diff(data))
#         # calculate sum of values above or below the threshold
#         res_points += (abs_change > threshold).sum() 
#         # calculate number of points above or below the threshold
#         res_value += np.mean(abs_change)
#         # Track validity
#         if res_points > 0:
#             found_good = False
#         if res_points > threshold:
#             found_okay = False

#         import matplotlib.pyplot as plt
#         plt.figure()
#         plt.axhline(y=threshold, color='r', linestyle='-')
#         plt.axhline(y=-threshold, color='r', linestyle='-')
#         plt.title("abs change for " + str(col_name))
#         plt.ylabel("Residual")
#         plt.xlabel("Time (samples)")
#         plt.plot(abs_change)
#         plt.savefig(os.path.join( save_path, filename + "_abs-change-for_"+str(col_name)))
#         plt.clf()
#         plt.close()
#         gc.collect()

#     # print('eye: ', df.columns)
#     # print('found_good: ', found_good)
#     # print('found_okay: ', found_okay)
#     # print('res_value:   ', res_value)
#     # print('res_points:  ', res_points)
#     return found_good, found_okay, res_value, res_points



def remove_outliers( data, threshold_min, threshold_max):
    '''Removes outliers from the passed data array that are below or above the indicated min or max values, respectively.

    Removed values are replaced with the previous valid value.

    Args:
        data (numpy array or pandas series): Array (single row) to remove outliers from
        threshold_min (float):               Minimum acceptable value. Values below this will be removed.
        threshold_max (float):               Maximum acceptable value. Values above this will be removed.
    
    Returns:
        Numpy array matching the input shape, with values outside the valid range removed.
    '''
    # Silence warnings that occur when logical or encounters NaN values
    with np.errstate(invalid='ignore'):
        index = np.logical_or( (threshold_max < data), (data < threshold_min)).nonzero()[0]

    for i in index:
        data[i] = np.nan
    
    return data



def get_indexStartStop( data_boolean, get_pairs=True):
    '''Finds the start and stop indicies of TRUE values in a boolean array.
    Usually used to find the range of an event from a True/False dataframe column

    Args:
        data_boolean (pandas series): Array of boolean values to find the start/stop indicies.
        get_pairs (bool, optional):   Whether to return pairs of values in a list, or two separate list of start/stop indicies

    Returns:
        ``get_pairs = False`` -> List of Tuples where each list element is a tuple contining `[index_start, index_end]` values;
        ``get_pairs = True``  -> Two arrays containing the index of the event start and the index of the event end.
    '''

    # while data_boolean.iloc[start] == True:
    #     start += 1
    #     if start >= end:
    #         break
    # while data_boolean.iloc[end] == True:
    #     end   -= 1
    #     if end <= start:
    #         break
    # insert a False at the beginning to prevent missing the first start
    data_boolean = pd.concat([pd.Series([False]), data_boolean])
    # insert a False at the beginning to prevent missing the last stop
    # data_boolean.insert(max(data_boolean.shape), False)
    data_boolean = pd.concat([data_boolean, pd.Series([False], index=[data_boolean.index[-1]])])
    # start = 0
    # end = max(data_boolean.shape) - 1
    # data = data_boolean.iloc[start:end+1]
    startStop = data_boolean.astype(np.int8).diff()
    start     = startStop[ startStop ==  1].index
    end       = startStop[ startStop == -1].index

    if get_pairs:
        return list( zip( start, end))
    else:
        return start, end



if __name__ == '__main__':

    import analyze as analyze
    import extract as extract
    import process as process
    import plot    as fplot
    import sys
    import os

    import time
    start_time = time.time()

    path_data = '/Users/trevor/Dropbox/Mac (2)/Documents/datasets/eyelink'

    # Variables
    subject = 'NLS_6'
    trials         = {'stroop': ['Word_Color_long', 'Word_Color_long_END']}
    trim_trial     = True

    analysis_constants = { 'closest_blink' :       50, # ms
                           'threshold_fixDist' :   35, # pixels,      default=20
                           'threshold_fixVel' :    30, # deg/sec,     default=25
                           'threshold_fixAcc' :  4000, # deg/sec/sec, default=3000
                           'gaze_tolerance_x' :   100, # pixels
                           'gaze_tolerance_y' :  None # pixels
                         }
    # Read in the raw data
    target_filename = subject + '.hdf5'
    path_file       = ''
    for root, dirs, files in os.walk(path_data):
        for filename in files:
            if filename == target_filename:
                path_file = os.path.join(root, filename)
                break
        if len(path_file) > 0:
            break
    if not len(path_file) > 0:
        print( target_filename, ' not found.')
        sys.exit()

    data_eye_annotation = extract.hdf2df( path_file, 'eyelink_annotations')
    data_eye_samples    = extract.hdf2df( path_file, 'eyelink_samples')

    for trial, trial_messages in trials.items():
        # Extract the timestamps of interest
        start = data_eye_annotation.loc[ data_eye_annotation.iloc[:,2] == trial_messages[0]].iloc[:,1]
        end   = data_eye_annotation.loc[ data_eye_annotation.iloc[:,2] == trial_messages[1]].iloc[:,1]
        # For all the identified trial start/stop indexes (there may be multiple runs of a single trial. Often there is only one.)
        for index_trial, (row_start, row_end) in enumerate( zip( start, end)):
            # Save a description to document observations with
            description_trial = subject + '_' + trial + '-' + str(index_trial)

            df_trial = data_eye_samples[ (data_eye_samples['timestamp'] >= float(start)) & (data_eye_samples['timestamp'] <= float(end))]
            
            df_trial, info_saccade_L, info_fixation_L, info_blink_L = analyze.get_eyeMovement(df_trial, analysis_constants, 'timestamp', 'pos_x_left',  'pos_y_left',  'vel_x_left', 'vel_y_left',  data_label='left')
            # df_trial, info_saccade_R, info_fixation_R, info_blink_R = analyze.get_eyeMovement(df_trial, analysis_constants, 'timestamp', 'pos_x_right', 'pos_y_right', 'vel_x_right','vel_y_right', data_label='right')
            
            # cluster_fcn, (num_words, num_lines) = get_stimuliClusters_old(info_fixation_L, trial, 'pos_x', 'pos_y', 'duration', col_gaze_line='gaze_line_left', col_gaze_word='gaze_word_left')
            # cluster_fcn, (num_words, num_lines) = get_stimuliClusters_rowFirst(info_fixation_L, trial, 'timestamp_start', 'timestamp_end', 'pos_x', 'pos_y', 'duration', col_gaze_line='gaze_line_left', col_gaze_word='gaze_word_left')
            # cluster_fcn, (num_words, num_lines) = get_stimuliClusters(info_fixation_R, trial, 'timestamp_start', 'timestamp_end', 'pos_x', 'pos_y', 'duration', col_gaze_line='gaze_line_right', col_gaze_word='gaze_word_right')
            info_fixation_L, cluster_fcn, (num_words, num_lines) = get_stimuliFixation(info_fixation_L, trial, 'timestamp_start', 'timestamp_end', 'pos_x', 'pos_y', 'duration', col_gaze='gaze_left', col_gaze_line='gaze_line_left', col_gaze_word='gaze_word_left', col_gaze_text='gaze_text_left', col_gaze_color='gaze_color_left')

            # df_trial, cluster_desc = add_stimuli(df_trial, trial, cluster_fcn, num_words,       'pos_x_left',       'pos_y_left',  'gaze_left',  'gaze_line_left',  'gaze_word_left',  'gaze_text_left',  'gaze_color_left', col_exclude=('saccade_left', 'blink_left'))
            # df_trial, cluster_desc = add_stimuli(df_trial, trial, cluster_fcn, num_words,      'pos_x_right',      'pos_y_right', 'gaze_right', 'gaze_line_right', 'gaze_word_right', 'gaze_text_right', 'gaze_color_right', col_exclude=('saccade_right','blink_right'))

            # trial_start, trial_end, df_trial, [info_saccade_L, info_fixation_L, info_blink_L] = trim_wordTrial( df_trial, 'gaze_line_left', 'gaze_word_left', events_avoid=['saccade_left', 'blink_left'], df_summary_trim=[info_saccade_L, info_fixation_L, info_blink_L], col_timestamp='timestamp', col_timestamp_start='timestamp_start', col_timestamp_end='timestamp_end')
            
            # info_saccade_L, _  = add_stimuli(info_saccade_L,  trial, cluster_fcn, num_words, 'pos_start_x', 'pos_start_y', 'gaze_start', 'gaze_line_start', 'gaze_word_start', 'gaze_text_start', 'gaze_color_start')
            # info_saccade_L, _  = add_stimuli(info_saccade_L,  trial, cluster_fcn, num_words,   'pos_end_x',   'pos_end_y',   'gaze_end',   'gaze_line_end',   'gaze_word_end',   'gaze_text_end',   'gaze_color_end')
            # info_fixation_L, _ = add_stimuli(info_fixation_L, trial, cluster_fcn, num_words,       'pos_x',       'pos_y',       'gaze',       'gaze_line',       'gaze_word',       'gaze_text',       'gaze_color')
            # info_blink_L, _    = add_stimuli(info_blink_L,    trial, cluster_fcn, num_words, 'pos_start_x', 'pos_start_y', 'gaze_start', 'gaze_line_start', 'gaze_word_start', 'gaze_text_start', 'gaze_color_start')
            # info_blink_L, _    = add_stimuli(info_blink_L,    trial, cluster_fcn, num_words,   'pos_end_x',   'pos_end_y',   'gaze_end',   'gaze_line_end',   'gaze_word_end',   'gaze_text_end',   'gaze_color_end')
            
            info_saccade_L = add_eventStimuli(info_saccade_L, info_fixation_L, 'timestamp_start', 'timestamp_end', 'gaze_left', 'gaze_line_left', 'gaze_word_left', 'gaze_color_left', 'gaze_text_left', col_exclude=None)
            info_blink_L   = add_eventStimuli(info_blink_L, info_fixation_L, 'timestamp_start', 'timestamp_end', 'gaze_left', 'gaze_line_left', 'gaze_word_left', 'gaze_color_left', 'gaze_text_left', col_exclude=None)
            df_trial       = add_trialStimuli(df_trial, info_fixation_L, [info_saccade_L, info_blink_L], 'timestamp', 'timestamp_start', 'timestamp_end', 'gaze_left', 'gaze_line_left', 'gaze_word_left', 'gaze_text_left', 'gaze_color_left')
            
            info_gaze = []
            for stimuli in sorted( info_fixation_L['gaze_left'].unique()):
                # Skip the first and last lines
                if (stimuli[2] == '0') or (stimuli[2] == str(num_lines-1)):
                    continue
                info_gaze.append( analyze.get_gazeStats(stimuli, info_saccade_L, info_fixation_L, info_blink_L, 'pos_x', 'pos_y', 'gaze_left', 'gaze_left_start', 'gaze_left_end', 'duration', 'velocity_avg', 'velocity_max'))
            info_gaze = pd.DataFrame(info_gaze, columns=analyze.get_gazeStats_key())

            # cluster_desc = [','.join(s) for s in get_stimuliDescriptions('stroop')]
            import matplotlib.pyplot as plt
            fplot.plot_word(df_trial, 'timestamp', 'pos_x_left', 'pos_y_left', 'gaze_line_left', 'gaze_word_left', save_plots=False, save_path=None)
            fplot.plot_fixationStimuli(cluster_fcn, info_fixation_L, 'pos_x', 'pos_y', 'gaze_word', 'gaze_line', cluster_descriptions=cluster_desc, save_plots=False, save_path=None)
            fplot.plot_boundaryStimuli(cluster_fcn, cluster_descriptions=cluster_desc, save_plots=False, save_path=None)
            plt.show()

            # df_trial        = add_stimuli(df_trial,        trial, cluster_fcn, num_words, num_lines),      'pos_x_right',      'pos_y_right', 'gaze_right', 'gaze_line_right', 'gaze_word_right', 'gaze_text_right', 'gaze_color_right', col_exclude=('saccade_right', 'blink_right'))
            # df_trial        = trim_wordTrial( df_trial, 'gaze_line_right', 'gaze_word_right', events_avoid=['saccade_right', 'blink_right'], df_summary_trim=[info_saccade_R, info_fixation_R, info_blink_R], col_timestamp='timestamp', col_timestamp_start='timestamp_start', col_timestamp_end='timestamp_end')
            # info_saccade_R  = add_stimuli(info_saccade_R,  trial, cluster_fcn, num_words, 'pos_start_x', 'pos_start_y', 'gaze_start', 'gaze_line_start', 'gaze_word_start', 'gaze_text_start', 'gaze_color_start')
            # info_saccade_R  = add_stimuli(info_saccade_R,  trial, cluster_fcn, num_words,   'pos_end_x',   'pos_end_y',   'gaze_end',   'gaze_line_end',   'gaze_word_end',   'gaze_text_end',   'gaze_color_end')
            # info_fixation_R = add_stimuli(info_fixation_R, trial, cluster_fcn, num_words,       'pos_x',       'pos_y',       'gaze',       'gaze_line',       'gaze_word',       'gaze_text',       'gaze_color')
            # info_blink_R    = add_stimuli(info_blink_R,    trial, cluster_fcn, num_words, 'pos_start_x', 'pos_start_y', 'gaze_start', 'gaze_line_start', 'gaze_word_start', 'gaze_text_start', 'gaze_color_start')
            # info_blink_R    = add_stimuli(info_blink_R,    trial, cluster_fcn, num_words,   'pos_end_x',   'pos_end_y',   'gaze_end',   'gaze_line_end',   'gaze_word_end',   'gaze_text_end',   'gaze_color_end')
            # info_gaze_R = analyze.get_gazeStats(info_saccade_R, info_fixation_R, info_blink_R, 'pos_x', 'pos_y', 'gaze', 'gaze_start', 'gaze_end', 'duration', 'velocity_avg', 'velocity_max')
            
            
            print('\ninfo_saccade')
            print(info_saccade_L)
            # print(info_saccade_R)
            print('\ninfo_fixation')
            print(info_fixation_L)
            # print(info_fixation_R)
            print('\ninfo_blink')
            print(info_blink_L)
            # print(info_blink_R)
            print('\ninfo_gaze')
            print(pd.DataFrame(info_gaze_L, columns=analyze.get_gazeStats_key()).sort_values('stimuli'))
            # print(info_gaze_R)
    
    print('TOTAL TIME: ', time.time() - start_time)


# def get_screenLocations(testName):
#     stimuli = {}

#     if 'stroop' in testName.strip().lower() or 'colors_preliminary' in testName.strip().lower():
#         stimuli['0,0,r,b'] = ( 297, 411, 249, 290)
#         stimuli['0,1,b,r'] = ( 754, 841, 249, 290)
#         stimuli['0,2,r,g'] = (1206,1358, 249, 290)
#         stimuli['0,3,g,b'] = (1661,1775, 249, 290)
#         stimuli['1,0,b,g'] = ( 297, 448, 388, 430)
#         stimuli['1,1,g,b'] = ( 754, 866, 388, 430)
#         stimuli['1,2,b,r'] = (1206,1296, 388, 430)
#         stimuli['1,3,b,g'] = (1661,1812, 388, 430)
#         stimuli['2,0,g,r'] = ( 297, 388, 548, 590)
#         stimuli['2,1,r,g'] = ( 754, 905, 548, 590)
#         stimuli['2,2,g,b'] = (1206,1323, 548, 590)
#         stimuli['2,3,r,b'] = (1661,1775, 548, 590)
#         stimuli['3,0,r,b'] = ( 297, 388, 708, 750)
#         stimuli['3,1,b,r'] = ( 754, 843, 708, 750)
#         stimuli['3,2,b,g'] = (1206,1360, 708, 750)
#         stimuli['3,3,g,r'] = (1661,1750, 708, 750)
#         stimuli['4,0,b,r'] = ( 297, 388, 869, 910)
#         stimuli['4,1,r,g'] = ( 754, 907, 869, 910)
#         stimuli['4,2,g,r'] = (1206,1289, 869, 910)
#         stimuli['4,3,r,g'] = (1661,1814, 869, 910)
#         stimuli['5,0,r,g'] = ( 297, 450,1030,1072)
#         stimuli['5,1,g,b'] = ( 754, 868,1030,1072)
#         stimuli['5,2,r,b'] = (1206,1321,1030,1072)
#         stimuli['5,3,b,r'] = (1661,1750,1030,1072)

#     return stimuli

# stimuli['0,0,red,blue']   = [ 167, 231, 140, 163]
# stimuli['0,1,blue,red']   = [ 424, 473, 140, 163]
# stimuli['0,2,red,green']  = [ 678, 764, 140, 163]
# stimuli['0,3,green,blue'] = [ 934, 998, 140, 164]
# stimuli['1,0,blue,green'] = [ 166, 252, 218, 241]
# stimuli['1,1,green,blue'] = [ 424, 487, 218, 242]
# stimuli['1,2,blue,red']   = [ 678, 729, 218, 242]
# stimuli['1,3,blue,green'] = [ 934,1019, 218, 241]
# stimuli['2,0,green,red']  = [ 167, 218, 308, 331]
# stimuli['2,1,red,green']  = [ 423, 509, 308, 331]
# stimuli['2,2,green,blue'] = [ 679, 744, 308, 332]
# stimuli['2,3,red,blue']   = [ 934, 998, 308, 332]
# stimuli['3,0,red,blue']   = [ 167, 218, 308, 331]
# stimuli['3,1,blue,red']   = [ 424, 474, 398, 421]
# stimuli['3,2,blue,green'] = [ 678, 765, 398, 421]
# stimuli['3,3,green,red']  = [ 934, 984, 398, 422]
# stimuli['4,0,blue,red']   = [ 167, 218, 489, 512]
# stimuli['4,1,red,green']  = [ 424, 510, 489, 512]
# stimuli['4,2,green,red']  = [ 678, 725, 489, 512]
# stimuli['4,3,red,green']  = [ 934,1020, 489, 512]
# stimuli['5,0,red,green']  = [ 167, 253, 580, 603]
# stimuli['5,1,green,blue'] = [ 424, 488, 579, 603]
# stimuli['5,2,red,blue']   = [ 678, 743, 579, 603]
# stimuli['5,3,blue,red']   = [ 934, 984, 579, 602]
# Measurements based on screen size 1186/667
