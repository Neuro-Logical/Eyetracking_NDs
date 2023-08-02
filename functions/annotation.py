from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import pandas as pd

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
    start = 0
    end   = max(data_boolean.shape) - 1
    while data_boolean.iloc[start] == True:
        start += 1
        if start >= end:
            break
    while data_boolean.iloc[end  ] == True:
        end   -= 1
        if end <= start:
            break
    data = data_boolean.iloc[start:end+1]
    startStop = data.astype(np.int8).diff()
    start     = startStop[ startStop ==  1].index
    end       = startStop[ startStop == -1].index

    if get_pairs:
        return list( zip( start, end))
    else:
        return start, end





def remove_eyeMovement( notes, df_rawData, df_saccade, df_fixation, df_blink, col_timestamp, col_timestamp_start, col_timestamp_end, desc_suffix=''):
    '''Removes identified eye movement based on time ranges flagged in notes file as ignore.

    Args:
        notes (pandas dataframe):       Dataframe containing annotated notes about the raw data
        df_rawData (pandas dataframe):  Dataframe containing the raw data.
        df_saccade (pandas dataframe):  Dataframe containing information about individual saccades
        df_fixation (pandas dataframe): Dataframe containing information about individual fixations
        df_blink (pandas dataframe):    Dataframe containing information about individual blinks
        col_timestamp (str):            Identifier for the column containing timestamp information
        col_timestamp_start (str):      Identifier for the column containting timestamps of an event start (saccade, fixation, and blinks have a starting and ending timestamp)
        col_timestamp_end (str):        Identifier for the column containting timestamps of an event end (saccade, fixation, and blinks have a starting and ending timestamp)
        desc_suffix (str, optional):   Suffix used to describe the data source. Typically used to designate "right" or "left" eye.
    
    Returns:
        df_rawData, df_saccade, df_fixation, and df_blink with the annotated eye movement characteristics removed.
    '''
    list_df_outputs = []

    if notes is not None:
        for df_info, desc_data in zip( [df_saccade, df_fixation, df_blink], ['saccade'+desc_suffix, 'fixation'+desc_suffix, 'blink'+desc_suffix]):
            notes_use = notes.loc[ notes['data'] == desc_data]

            if 'ignore_before' in notes_use['update'].values:
                for _, row in notes_use[ notes_use['update'] == 'ignore_before'].iterrows():
                    df_info = df_info.loc[ df_info[col_timestamp_start] >= row['timestamp_start']]
                    df_rawData.loc[ df_rawData[col_timestamp] < row['timestamp_start'], desc_data ] = False

            if 'ignore_after' in notes_use['update'].values:
                for _, row in notes_use[ notes_use['update'] == 'ignore_after'].iterrows():
                    df_info = df_info.loc[ df_info[col_timestamp_end] <= row['timestamp_end']]
                    df_rawData.loc[ df_rawData[col_timestamp] > row['timestamp_end'], desc_data ] = False

            if 'ignore_between' in notes_use['update'].values:
                for _, row in notes_use[ notes_use['update'] == 'ignore_between'].iterrows():
                    ts_start = row['timestamp_start']
                    ts_end   = row['timestamp_end']
                    if ts_start > ts_end:
                        ts_start = row['timestamp_end']
                        ts_end   = row['timestamp_start']
                    for i in df_info.index:
                        if (df_info.loc[i, col_timestamp_start] < ts_start) and (df_info.loc[i, col_timestamp_end] > ts_start):
                            df_info.loc[i,col_timestamp_end] = ts_start
                            pass
                        if (df_info.loc[i, col_timestamp_start] < ts_end)   and (df_info.loc[i, col_timestamp_end] > ts_end):
                            df_info.loc[i,col_timestamp_start] = ts_end
                            pass
                    df_info = df_info.loc[ (df_info[col_timestamp_end] <= ts_start) | (df_info[col_timestamp_start] >= ts_end)].copy()
                    df_rawData.loc[ (df_rawData[col_timestamp] >= ts_start) & (df_rawData[col_timestamp] <= ts_end), desc_data ] = False

            list_df_outputs.append(df_info)

    else:
        list_df_outputs = [df_saccade, df_fixation, df_blink]

    return df_rawData, list_df_outputs[0], list_df_outputs[1], list_df_outputs[2]



def separate_stimuliProgress( notes, df_fixation, col_fix_time_start, col_fix_time_end):
    '''Separates all fixations into two dataframes based on whether they occured during a subjects "focused" task period, 
    or whether they occured before or after this identified period.
    
    Args:
        notes (pandas dataframe):       Dataframe containing annotated notes about the raw data.
        df_fixation (pandas dataframe): Dataframe containing information about individual fixations.
        col_fix_time_start (str):       Identifier for the column in `df_fixation` containing the fixation starting timestamp.
        col_fix_time_end (str):         Identifier for the column in `df_fixation` containing the fixation ending timestamp.

    Returns:
        *df_fixations_keep* designating fixations during the focus time, and *df_fixations_skip* containing fixations outside of the focus time.
    '''
    if notes is None:
        return df_fixation, df_fixation.drop(df_fixation.index, inplace=False)

    df_fixation_use        = df_fixation.copy()
    df_fixation_use['use'] = True
    notes_use       = notes.loc[ notes['data'] == 'stimuli_progress']

    if 'ignore_before' in notes_use['update'].values:
        for _, row in notes_use[ notes_use['update'] == 'ignore_before'].iterrows():
            df_fixation_use.loc[(df_fixation_use[col_fix_time_end]   < row['timestamp_start']), 'use'] = False
    if 'ignore_after' in notes_use['update'].values:
        for _, row in notes_use[ notes_use['update'] == 'ignore_after'].iterrows():
            df_fixation_use.loc[(df_fixation_use[col_fix_time_start] > row['timestamp_end']),   'use'] = False
    if 'use_between' in notes_use['update'].values:
        for _, row in notes_use[ notes_use['update'] == 'use_between'].iterrows():
            ts_start = row['timestamp_start']
            ts_end   = row['timestamp_end']
            if ts_start > ts_end:
                ts_start = row['timestamp_end']
                ts_end   = row['timestamp_start']
            df_fixation_use.loc[(df_fixation_use[col_fix_time_end]   < ts_start), 'use'] = False
            df_fixation_use.loc[(df_fixation_use[col_fix_time_start] > ts_end),   'use'] = False

    df_fixation_keep = df_fixation_use.loc[  df_fixation_use['use']].drop(columns='use', inplace=False)
    df_fixation_skip = df_fixation_use.loc[ ~df_fixation_use['use']].drop(columns='use', inplace=False)
    return df_fixation_keep, df_fixation_skip



def corect_gazeStimuli( notes, df_rawData, col_timestamp_start, col_timestamp_end, col_gaze_line, col_gaze_word):
    '''Uses annotations to apply a correction to gaze stimuli designation.
    
    Args:
        notes (pandas dataframe):      Dataframe containing annotated notes about the raw data
        df_rawData (pandas dataframe): Dataframe containing gaze raw data to apply the correction to.
        col_timestamp_start (str):     Identifier for the column in `df_rawData` containing the gaze starting timestamp.
        col_timestamp_end (str):       Identifier for the column in `df_rawData` containing the gaze ending timestamp.
        col_gaze_line (str):           Identifier for the column in `df_rawData` containing the gaze line index.
        col_gaze_word (str):           Identifier for the column in `df_rawData` containing the gaze word index.
    '''
    if notes is not None:
        notes_use = notes.loc[ notes['data'] == 'stimuliFixation']

        if 'set_gaze_line' in notes_use['update'].values:
            for _, row in notes_use[ notes_use['update'] == 'set_gaze_line'].iterrows():
                rows_update = (df_rawData[col_timestamp_start] > row['timestamp_start']) & (df_rawData[col_timestamp_end] < row['timestamp_end']) & (df_rawData[col_gaze_line] == row['value'])
                df_rawData.loc[ rows_update, col_gaze_line] = row['value_new']

        if 'set_gaze_word' in notes_use['update'].values:
            for _, row in notes_use[ notes_use['update'] == 'set_gaze_word'].iterrows():
                rows_update = (df_rawData[col_timestamp_start] > row['timestamp_start']) & (df_rawData[col_timestamp_end] < row['timestamp_end']) & (df_rawData[col_gaze_word] == row['value'])
                df_rawData.loc[ rows_update, col_gaze_word] = row['value_new']

    return df_rawData



def add_annotation( df_notes, df_rawData, col_timestamp, col_pos_x, col_pos_y, col_saccade, col_blink, filename, trial, desc_suffix=''):
    '''Generates a GUI allowing easy point-and-click annotation of raw data.

    Args:
        df_notes (pandas dataframe):   Dataframe containing annotated notes about the raw data
        df_rawData (pandas dataframe): Dataframe containing raw data to annotate
        col_timestamp (str):           Identifier for the column containing timestamp information
        col_pos_x (str):               Identifier for the column containing eye position in the X direction
        col_pos_y (str):               Identifier for the column containing eye position in the Y direction
        col_saccade (str):             Identifier for the column containing information about the presence of saccades
        col_blink (str):               Identifier for the column containing information about the presence of blinks
        filename (str):                Filename where the raw data came from. This is saved into the notes dataframe for later data organization.
        trial (str):                   Trial where the raw data came from. This is saved into the notes dataframe for later data organization.
        desc_suffix (str, optional):   Suffix used to describe the data source. Typically used to designate "right" or "left" eye.
    '''
    try:
        import eyetracking.functions.process as process
    except:
        try:
            import functions.process as process
        except:
            import process as process
    
    while True:
        # Create a figure, leaving room at the bottom for buttons
        fig, ax = plt.subplots(figsize=(12,8))
        plt.subplots_adjust(bottom=0.2)
        plt.title('Indicate Stimuli Progress Time')

        # Plot the data you would like to annotate
        # Raw data
        plt.plot(df_rawData[col_timestamp], df_rawData[col_pos_x], 'b', linewidth=0.4, label=col_pos_x)
        plt.plot(df_rawData[col_timestamp], df_rawData[col_pos_y], 'c', linewidth=0.4, label=col_pos_x)
        # Highlight saccades in RED
        startStop_saccade  = process.get_indexStartStop( df_rawData[col_saccade])
        plt.plot(df_rawData.loc[startStop_saccade[0][0]:startStop_saccade[0][1], col_timestamp], df_rawData.loc[startStop_saccade[0][0]:startStop_saccade[0][1],col_pos_x], 'r', label='saccade', linewidth=3, alpha=0.7)
        for start_sac, end_sac in startStop_saccade[1:]:
            plt.plot(df_rawData.loc[start_sac:end_sac, col_timestamp], df_rawData.loc[start_sac:end_sac, col_pos_x], 'r', linewidth=1, alpha=0.7)
        # Highlight blinks in Blue
        startStop_blink    = process.get_indexStartStop( df_rawData[col_blink])
        if len(startStop_blink) > 0:
            plt.plot([df_rawData.loc[startStop_blink[0][0], col_timestamp], df_rawData.loc[startStop_blink[0][1], col_timestamp]], [0,0], 'g', linewidth=5, label='blink')
            for start_blink, end_blink in startStop_blink[1:]:
                plt.plot([df_rawData.loc[start_blink, col_timestamp], df_rawData.loc[end_blink, col_timestamp]], [0,0], 'g', linewidth=3)

        # Create an axis for each annotation button
        ax_trimBefore    = plt.axes([0.1,  0.05, 0.1, 0.075])
        ax_trimAfter   = plt.axes([0.2,  0.05, 0.1, 0.075])
        ax_mvtIgnore    = plt.axes([0.35, 0.05, 0.1, 0.075])
        ax_useBetween   = plt.axes([0.6,  0.05, 0.1, 0.075])
        ax_ignoreBefore = plt.axes([0.7,  0.05, 0.1, 0.075])
        ax_ignoreAfter  = plt.axes([0.8,  0.05, 0.1, 0.075])
        # Initialize each annotation button
        b_trimBefore   = Button(ax_trimBefore,  'TRIM BEF')
        b_trimAfter    = Button(ax_trimAfter,   'TRIM AFT')
        b_mvtIgnore    = Button(ax_mvtIgnore,    'IGN BTW')
        b_useBetween   = Button(ax_useBetween,   'USE BTW')
        b_ignoreBefore = Button(ax_ignoreBefore, 'IGN BEF')
        b_ignoreAfter  = Button(ax_ignoreAfter,  'IGN AFT')


        # Create a class to track the most recent click locations from the GUI
        class Annotator:
            def __init__(self, axes):
                self.axes     = axes
                self.notes    = pd.DataFrame(columns=['filename', 'trial', 'data', 'update', 'timestamp_start', 'timestamp_end', 'value', 'value_new'])
                self.press = (0,0)
                self.release = (0,0)
            
            # This function is called every click press
            def on_press(self, event):
                if event.inaxes is not self.axes:
                    return
                self.press = (event.xdata, event.ydata)
                print('\t-->Pressed: ', self.press)
            
            # This function is called every click release
            def on_release(self, event):
                if event.inaxes is not self.axes:
                    return
                self.release = (event.xdata, event.ydata)
                print('\t-->Released: ', self.release)

            # Create one function to respond to each button

            def addNote_mvtIgnore(self, event):
                print('\t-->Saccade Fixation Blink Ignore Between ', self.press[0], ' - ', self.release[0])
                # Save the press and release locations in a new notes row designating SACCADE IGNORE_BETWEEN
                df_row_sac = pd.DataFrame({'data': ['saccade_'+desc_suffix],  'update': ['ignore_between'], 'timestamp_start': [self.press[0]], 'timestamp_end': [self.release[0]]})
                # Save the press and release locations in a new notes row designating FIXATION IGNORE_BETWEEN
                df_row_fix = pd.DataFrame({'data': ['fixation_'+desc_suffix], 'update': ['ignore_between'], 'timestamp_start': [self.press[0]], 'timestamp_end': [self.release[0]]})
                # Save the press and release locations in a new notes row designating BLINK IGNORE_BETWEEN
                df_row_blk = pd.DataFrame({'data': ['blink_'+desc_suffix],    'update': ['ignore_between'], 'timestamp_start': [self.press[0]], 'timestamp_end': [self.release[0]]})
                # Update the notes dataframe
                self.notes = pd.concat([self.notes, df_row_sac, df_row_fix, df_row_blk])
                return

            
            def addNote_trimBefore(self, event):
                print('\t-->Trim Before ', self.press[0])
                # Save the press location in a new notes row designating TRIM_TRIAL IGNORE_BEFORE
                df_row = pd.DataFrame({'data': ['trim_trial'],  'update': ['trim_before'], 'timestamp_start': [self.press[0]]})
                # Update the notes dataframe
                self.notes = pd.concat([self.notes, df_row])
                return
            
            def addNote_trimAfter(self, event):
                print('\t-->Trim After ', self.press[0])
                # Save the press location in a new notes row designating TRIM_TRIAL IGNORE_AFTER
                df_row = pd.DataFrame({'data': ['trim_trial'],  'update': ['trim_after'], 'timestamp_start': [self.press[0]]})
                # Update the notes dataframe
                self.notes = pd.concat([self.notes, df_row])
                return

            def addNote_useBetween(self, event):
                print('\t-->Use Between ', self.press[0], ' - ', self.release[0])
                # Save the press and release locations in a new notes row designating STIMULI_PROGRESS USE_BETWEEN
                df_row = pd.DataFrame({'data': ['stimuli_progress'], 'update': ['use_between'], 'timestamp_start': [self.press[0]], 'timestamp_end': [self.release[0]]})
                # Update the notes dataframe
                self.notes = pd.concat([self.notes, df_row])
                return

            def addNote_ignoreBefore(self, event):
                print('\t-->Ignore Before ', self.press[0])
                # Save the press location in a new notes row designating STIMULI_PROGRESS IGNORE_BEFORE
                df_row = pd.DataFrame({'data': ['stimuli_progress'], 'update': ['ignore_before'], 'timestamp_start': [self.press[0]]})
                # Update the notes dataframe
                self.notes = pd.concat([self.notes, df_row])
                return

            def addNote_ignoreAfter(self, event):
                print('\t-->Ignore After ', self.press[0])
                # Save the press location in a new notes row designating STIMULI_PROGRESS IGNORE_AFTER
                df_row = pd.DataFrame({'data': ['stimuli_progress'], 'update': ['ignore_after'], 'timestamp_end': [self.press[0]]})
                # Update the notes dataframe
                self.notes = pd.concat([self.notes, df_row])
                return

        # Initialize the class to track mouse and button presses
        annotate = Annotator(ax)

        # Link each button to the associated class function
        b_trimBefore.on_clicked(   annotate.addNote_trimBefore)
        b_trimAfter.on_clicked(    annotate.addNote_trimAfter)
        b_mvtIgnore.on_clicked(    annotate.addNote_mvtIgnore)
        b_useBetween.on_clicked(   annotate.addNote_useBetween)
        b_ignoreBefore.on_clicked( annotate.addNote_ignoreBefore)
        b_ignoreAfter.on_clicked(  annotate.addNote_ignoreAfter)
        # Link the press/release actions to the associated class function
        fig.canvas.mpl_connect('button_press_event',   annotate.on_press)
        fig.canvas.mpl_connect('button_release_event', annotate.on_release)

        # Show the plot window, which will launch the interactive GUI. Close the window to exit.
        print('Close plot window when finished...')
        plt.show()

        # Once all notes are recorded, add columns to the generated notes dataframe to track metadata
        annotate.notes['filename'] = filename
        annotate.notes['trial']    = trial

        # Confirm with the user the notes look correct.
        # If not, the loop will give the user another chance to generate the correct annotations
        print(annotate.notes)
        user_confirm = input('\n\nAccept the above annotations (y/n)? ')
        if user_confirm.lower().startswith('y'):
            break

    # Add the newly generated notes to any existing notes
    if df_notes is not None:
        df_notes = pd.concat([df_notes, annotate.notes])
    else:
        df_notes = annotate.notes

    # Return the notes.
    # You may want to save/update the file here if it is not saved/updated outside of this function
    return df_notes
