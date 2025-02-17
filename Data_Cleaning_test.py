# Import necessary libraries at the top
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np


#-------------------------------------------------------------

def import_and_combine_data(folder_path):
    """
    This function reads CSV files from the specified folder, adds 'round' and 'device' columns
    based on the file names, and returns a single combined DataFrame with numeric 'round' and 'device' columns.

    Parameters:
    - folder_path: The path to the folder containing the CSV files.

    Returns:
    - A single DataFrame containing all the data from the CSV files, with numeric 'round' and 'device' columns.
    """
    # Get a list of all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

    # List to hold all DataFrames
    all_dataframes = []

    # Loop through the list of CSV files and read them into DataFrames
    for file in csv_files:
        # Extract the filename (without the directory path) and remove the extension
        filename = os.path.basename(file).replace('.csv', '')

        # Extract 'round' and 'device' from the filename (e.g., 'features_r4-d01.csv')
        round_key = filename.replace('features_', '').split('-')[0]  # Extracts 'r4'
        device_key = filename.split('-')[1]  # Extracts 'd01', 'd02', etc.

        try:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file)

            # If the DataFrame is empty, skip it
        
            if df.empty:
                print(f"Warning: {file} is empty and will be skipped.")
                continue

            # Convert 'round' from 'r4' to 4, 'device' from 'd01' to 1
            round_number = int(round_key[1:])  # Extract the numeric part after 'r'
            device_number = int(device_key[1:])  # Extract the numeric part after 'd'

            # Add the 'round' and 'device' columns to the DataFrame
            df['round'] = round_number
            df['device'] = device_number

            # Append the modified DataFrame to the list
            all_dataframes.append(df)

        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

    # Check if any valid DataFrames were added
    if not all_dataframes:
        raise ValueError("No valid DataFrames to concatenate.")

    # Concatenate all DataFrames into a single DataFrame
    final_df = pd.concat(all_dataframes, ignore_index=True)

    return final_df

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def get_Cycle_thresholds(df, devices, rounds, min_temp_threshold=40):
    """
    Computes cycles and minimum temperatures for each device and round.

    Parameters:
    - df: DataFrame containing the data
    - devices: List of device identifiers or single device to process
    - rounds: List of round numbers or single round to process
    - min_temp_threshold: Minimum temperature threshold

    Returns:
    - Dictionary with device IDs mapping to tuple of (cycles at minimums, minimum temperatures)
    """
    round_data = {}

    for round_number in rounds:
        device_data = {}
        
        for device in devices:
            exceed_threshold_df = df[(df['device'] == device) & 
                                   (df['round'] == round_number) & 
                                   (df['min_temperature'] <= min_temp_threshold)]
            
            if exceed_threshold_df.empty:
                device_data[device] = ([], [])
                continue
            
            exceed_threshold_df = exceed_threshold_df.reset_index(drop=True)
            exceed_threshold_df['segment'] = (exceed_threshold_df['cycle'].diff() > 1).cumsum()
            
            segments = exceed_threshold_df.groupby('segment')
            min_temp_indices = segments['min_temperature'].idxmin()
            
            cycles = exceed_threshold_df.loc[min_temp_indices, 'cycle'].astype(float).tolist()
            temps = exceed_threshold_df.loc[min_temp_indices, 'min_temperature'].astype(float).tolist()
            
            device_data[device] = (cycles, temps)
        
        round_data[round_number] = device_data
    
    return round_data[rounds[0]] if len(rounds) == 1 else round_data


#-----------------------------------------------------------------------------------------------------------------------------------------------------

def make_segments(df, devices, rounds, min_temp_threshold=40, cut_at_CL_one=True, cycle_threshold=500):
    """
    Segments device data based on cycles.
    
    Parameters:
    - df: DataFrame containing the data
    - devices: List of device identifiers or single device
    - rounds: List of round numbers or single round
    - min_temp_threshold: Minimum temperature threshold
    - cut_at_CL_one: Whether to cut data at consumed life = 1
    - cycle_threshold: Minimum cycle number to include
    
    Returns:
    - Nested dictionary: round -> device -> list of DataFrames (segments)
    """
    relevant_data = df[df['round'].isin(rounds) & df['device'].isin(devices)]
    initial_rows = len(relevant_data)

    thresholds_data = get_Cycle_thresholds(relevant_data, devices, rounds, min_temp_threshold)
    
    if isinstance(devices, int):
        devices = [devices]
    if isinstance(rounds, int):
        rounds = [rounds]
        
    segmented_data = {}
    rows_removed_due_to_CL = 0
    rows_removed_due_to_cycle_threshold = 0

    for round_number in rounds:
        device_segments = {}
        
        round_thresholds = thresholds_data[round_number] if isinstance(thresholds_data, dict) else thresholds_data
        
        for device in devices:
            cycles, temps = round_thresholds.get(device, ([], []))
            
            device_data = relevant_data[(relevant_data['device'] == device) & 
                                      (relevant_data['round'] == round_number)]
            
            if cut_at_CL_one:
                before_cl_cut = len(device_data)
                device_data = device_data[device_data['consumed_life'] <= 1]
                rows_removed_due_to_CL += (before_cl_cut - len(device_data))
            
            before_cycle_cut = len(device_data)
            device_data = device_data[device_data['cycle'] >= cycle_threshold]
            rows_removed_due_to_cycle_threshold += (before_cycle_cut - len(device_data))
            
            segments = []
            if not cycles:
                segments.append(device_data)
            else:
                # First segment
                first_segment = device_data[device_data['cycle'] < cycles[0]]
                if not first_segment.empty:
                    segments.append(first_segment)
                
                # Middle segments
                for i in range(len(cycles) - 1):
                    segment = device_data[
                        (device_data['cycle'] >= cycles[i]) & 
                        (device_data['cycle'] < cycles[i + 1])
                    ]
                    if not segment.empty:
                        segments.append(segment)
                
                # Last segment
                last_segment = device_data[device_data['cycle'] >= cycles[-1]]
                if not last_segment.empty:
                    segments.append(last_segment)
            
            device_segments[device] = segments
        
        segmented_data[round_number] = device_segments
    
    remaining_rows = sum(len(seg) for round_data in segmented_data.values()
                         for device_segments in round_data.values() for seg in device_segments)
    
    removed_rows = initial_rows - remaining_rows
    print('segment_devices_data:')
    print(f"Total rows removed: {removed_rows:.2f} out of {initial_rows:.2f} ({100 * removed_rows / initial_rows:.2f}%)")
    print(f"Rows removed due to consumed_life > 1: {rows_removed_due_to_CL:.2f}")
    print(f"Rows removed due to cycle < {cycle_threshold:.2f}: {rows_removed_due_to_cycle_threshold:.2f}")
    print('----------------------------------------------------------------------------')

    return segmented_data[rounds[0]] if len(rounds) == 1 else segmented_data

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def exclude_transition_cycles(
    segmented_data, 
    transition_margin=10, 
    filter_columns=None,
    verbose=True
):
    """
    Marks data points near segment transitions as NaN based on a margin.
    Skips marking rows at the start of the first segment and the end of the last segment.
    
    Parameters:
    ----------- 
    segmented_data : dict
        Segmented data by round and device.
        Structure: {round_number: {device: [DataFrames (segments)]}}.
    transition_margin : int, optional
        Number of cycles to mark as NaN at the start and end of each segment to avoid transition regions.
    filter_columns : list, optional
        List of columns to mark as NaN based on the cycle. If None, all columns are marked as NaN (default behavior).
    verbose : bool, optional
        If True, prints the global summary. Default is True.
        
    Returns:
    --------
    filtered_segmented_data : dict
        The same structure as segmented_data, but with values near transition margins replaced with NaN,
        applied only to specified columns.
    """
    # Initialize the output dictionary with the same structure as the input
    filtered_segmented_data = {}
    total_rows_marked_nan = 0
    total_nans_marked = 0
    total_original_rows = 0

    for round_number, devices in segmented_data.items():
        for device, segments in devices.items():
            filtered_segments = []
            for i, segment in enumerate(segments):
                # Work on a copy of the DataFrame to avoid modifying the original data
                filtered_segment = segment.copy()
                original_row_count = len(segment)
                total_original_rows += original_row_count

                # Determine the range of cycles to mark as NaN
                min_cycle = filtered_segment['cycle'].min()
                max_cycle = filtered_segment['cycle'].max()

                # Adjust transition margin for boundary segments
                if i == 0:  # First segment, skip margin at the start
                    adjusted_min_cycle = min_cycle
                else:
                    adjusted_min_cycle = min_cycle + transition_margin

                if i == len(segments) - 1:  # Last segment, skip margin at the end
                    adjusted_max_cycle = max_cycle
                else:
                    adjusted_max_cycle = max_cycle - transition_margin

                rows_to_nan = ~filtered_segment['cycle'].between(adjusted_min_cycle, adjusted_max_cycle)

                # Apply NaN to the specified columns or all columns
                if filter_columns:
                    for col in filter_columns:
                        if filtered_segment[col].dtype != 'float':
                            filtered_segment[col] = filtered_segment[col].astype(float)
                    filtered_segment.loc[rows_to_nan, filter_columns] = np.nan
                else:
                    # Apply NaN to all columns
                    for col in filtered_segment.columns:
                        if col != 'cycle' and filtered_segment[col].dtype != 'float':
                            filtered_segment[col] = filtered_segment[col].astype(float)
                    filtered_segment.loc[rows_to_nan, :] = np.nan

                # Count rows and NaNs marked
                total_rows_marked_nan += rows_to_nan.sum()
                if filter_columns:
                    total_nans_marked += filtered_segment.loc[rows_to_nan, filter_columns].isna().sum().sum()
                else:
                    total_nans_marked += filtered_segment.loc[rows_to_nan, :].isna().sum().sum()

                filtered_segments.append(filtered_segment)

            filtered_segmented_data.setdefault(round_number, {})[device] = filtered_segments

    # Global summary
    if verbose and total_original_rows > 0:
        overall_percentage_nan = (total_rows_marked_nan / total_original_rows) * 100
        print('exclude_transition_cycles:')
        print(f"Total rows marked as NaN: {total_rows_marked_nan:.2f} out of {total_original_rows:.2f} ({overall_percentage_nan:.2f}%)")
        print(f"Total individual NaN values marked: {total_nans_marked:.2f}")
        print('----------------------------------------------------------------------------')

    return filtered_segmented_data

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def merge_segments(segmented_data, devices, rounds, columns_to_merge, mean_interval_proportion=0.1, min_interval=10, max_interval=500, transition_margin=10):
    merged_data = {}
    aligned_data = {}
    
    for round_number in rounds:
        round_data_merged = {}
        round_data_aligned = {}
        
        for device in devices:
            segments = segmented_data[round_number].get(device, [])
            if not segments or len(segments) == 1:
                round_data_merged[device] = (segments[0] if segments else None, [], min_interval, transition_margin)
                round_data_aligned[device] = segments
                continue
                
            merged_segment = segments[0].copy()
            aligned_segments = [segments[0].copy()]
            transition_means = []
            
            for i in range(1, len(segments)):
                transition_cycle = merged_segment['cycle'].max()
                next_segment = segments[i].copy()
                aligned_next_segment = segments[i].copy()
                
                segment_length = len(merged_segment)
                mean_interval = min(max(int(segment_length * mean_interval_proportion), min_interval), max_interval)
                
                # Get non-NaN indices directly
                before_transition = merged_segment[
                    (merged_segment['cycle'] >= transition_cycle - mean_interval) & 
                    (merged_segment['cycle'] <= transition_cycle - transition_margin)
                ]
                after_transition = next_segment[
                    (next_segment['cycle'] >= transition_cycle + transition_margin) & 
                    (next_segment['cycle'] <= transition_cycle + mean_interval)
                ]
                
                # Find valid rows with any non-NaN values
                valid_before = before_transition[columns_to_merge].notna().any(axis=1)
                valid_after = after_transition[columns_to_merge].notna().any(axis=1)
                
                mean_differences = {}
                for column in columns_to_merge:
                    first_data = before_transition.loc[valid_before, column].dropna()
                    second_data = after_transition.loc[valid_after, column].dropna()
                    
                    if len(first_data) > 0 and len(second_data) > 0:
                        first_mean = np.percentile(first_data, 50)
                        second_mean = np.percentile(second_data, 50)
                        transition_means.append((transition_cycle, column, first_mean, second_mean))
                        mean_differences[column] = second_mean - first_mean
                
                for column, mean_difference in mean_differences.items():
                    next_segment[column] -= mean_difference
                    aligned_next_segment[column] -= mean_difference
                
                merged_segment = pd.concat([merged_segment, next_segment]).sort_values('cycle').reset_index(drop=True)
                aligned_segments.append(aligned_next_segment)
            
            round_data_merged[device] = (merged_segment, transition_means, mean_interval, transition_margin)
            round_data_aligned[device] = aligned_segments
        
        merged_data[round_number] = round_data_merged
        aligned_data[round_number] = round_data_aligned
    
    return merged_data, aligned_data


#-----------------------------------------------------------------------------------------------------------------------------------------------------


def fill_gaps(merged_data, columns_to_fill, window_size=1000):
    filled_data = {}
    total_nans_filled = 0
    total_values = 0
    gap_sizes = []  # Store the sizes of gaps filled

    for round_number, devices in merged_data.items():
        filled_data[round_number] = {}
        
        for device, device_data in devices.items():
            if device_data is None:
                filled_data[round_number][device] = None
                continue
                
            merged_segment, transition_means, mean_interval, transition_margin = device_data
            filled_segment = merged_segment.copy()
            
            for column in columns_to_fill:
                before_fill = filled_segment[column].isna().sum()

                # Calculate rolling mean using center=False to use only prior values
                rolling_mean = filled_segment[column].rolling(window=window_size, min_periods=1, center=False).mean()
                
                # Detect and measure gaps before filling
                gap_start = None
                for idx, is_nan in enumerate(filled_segment[column].isna()):
                    if is_nan and gap_start is None:
                        gap_start = idx  # Mark the start of a gap
                    elif not is_nan and gap_start is not None:
                        gap_sizes.append(idx - gap_start)  # Store the gap size
                        gap_start = None
                # Handle if a gap continues to the end
                if gap_start is not None:
                    gap_sizes.append(len(filled_segment) - gap_start)

                filled_segment[column] = filled_segment[column].fillna(rolling_mean)
                after_fill = filled_segment[column].isna().sum()
                total_nans_filled += (before_fill - after_fill)
                total_values += len(filled_segment[column])
            
            filled_data[round_number][device] = (filled_segment, transition_means, mean_interval, transition_margin)

    # Calculate statistics for gaps
    if gap_sizes:
        max_gap = max(gap_sizes)
        min_gap = min(gap_sizes)
        mean_gap = sum(gap_sizes) / len(gap_sizes)
        std_gap = (sum((x - mean_gap) ** 2 for x in gap_sizes) / len(gap_sizes)) ** 0.5
    else:
        max_gap, min_gap, mean_gap, std_gap = 0, 0, 0, 0

    total_gaps = len(gap_sizes)

    # Print statistics
    percentage_filled = (total_nans_filled / total_values * 100) if total_values > 0 else 0
    print('fill_gaps:')
    print(f"Total NaN values filled: {total_nans_filled:,} out of {total_values:,} values ({percentage_filled:.2f}%)")
    print(f"Gap Statistics: Max={max_gap:.2f}, Min={min_gap:.2f}, Mean={mean_gap:.2f}, Std={std_gap:.2f}, Total Gaps={total_gaps:.2f}")
    print('----------------------------------------------------------------------------')

    # Return the filled data and gap statistics
    return filled_data, {'max_gap': max_gap, 'min_gap': min_gap, 'mean_gap': mean_gap, 'std_gap': std_gap, 'total_gaps': total_gaps}

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def calculate_moving_averages(merged_data, window_size=25):
    """
    Calculate the moving average of all numeric columns (except 'round' and 'device') 
    for each round and device in merged_data.
    
    Parameters:
    - merged_data: dict, nested dictionary with structure {round_number: {device: [DataFrame]}}.
    - window_size: int, the window size for calculating the moving averages (default is 25).
    
    Returns:
    - result_data: dict, nested dictionary with structure {round_number: {device: DataFrame}}.
    """
    # Initialize an empty dictionary to store the results
    result_data = {}
    
    # Columns to exclude from moving average calculation
    exclude_columns = ['round', 'device', 'cycle', 'consumed_life']
    
    total_removed_rows = 0  # Counter for removed rows across all DataFrames
    total_original_rows = 0  # Counter for total rows before processing

    # Iterate over each round in merged_data
    for round_number, devices in merged_data.items():
        # Initialize a dictionary for each round
        result_data[round_number] = {}
        
        # Iterate over each device in the round
        for device, data_list in devices.items():
            # Clone the DataFrame to avoid modifying the original data
            df = data_list[0].copy()
            
            # Keep track of original row count
            original_row_count = len(df)
            total_original_rows += original_row_count
            
            # Get all numeric columns except excluded columns
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
            numeric_columns = [col for col in numeric_columns if col not in exclude_columns]
            
            # Calculate the moving average for all selected numeric columns
            for column in numeric_columns:
                df[column] = df[column].rolling(window=window_size).mean()
            
            # Drop rows where any column contains NaN values
            df.dropna(inplace=True)
            
            # Count removed rows for this DataFrame
            removed_rows = original_row_count - len(df)
            total_removed_rows += removed_rows
            
            # Store the DataFrame directly in the result_data dictionary (not in a list)
            result_data[round_number][device] = df
    
    print('calculate_moving_averages')
    # Total rows removed summary
    print(f"Total rows removed: {total_removed_rows} out of {total_original_rows} "
          f"({100 * total_removed_rows / total_original_rows:.2f}%)")
    
    # Expected rows removed due to rolling window
    num_devices = sum(len(devices) for devices in merged_data.values())
    expected_rows_removed = (window_size - 1) * num_devices
    print(f"Expected rows removed due to window size ({window_size}): {expected_rows_removed:.2f}")
    print('--------------------------------------------------------------------------------------------')
    
    return result_data

#-------------------------------------------------------------------------------------------------
def add_initial_values(sampled_data_dict, verbose=False):
    """
    Add values for all numeric columns based on the lowest cycle to the sampled data,
    and print the cycle from which the value was taken.

    Parameters:
    - sampled_data_dict: Dictionary of format {round_number: {device: DataFrame}}
    - verbose: Boolean to control printing of processing details (default=False)

    Returns:
    - Dictionary of format {round_number: {device: DataFrame}} with added values based on the lowest cycle.
    """
    processed_data = {}

    # Columns to exclude from getting initial values
    exclude_columns = ['round', 'device'] # 'consumed_life', 'cycle', 'device_number'

    for round_number, devices in sampled_data_dict.items():
        processed_data[round_number] = {}

        if verbose:
            print(f"\nProcessing Round {round_number}")
            print("-" * 60)

        for device, df in devices.items():
            # Get all numeric columns
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
            columns_to_process = [col for col in numeric_columns if col not in exclude_columns]

            # Locate the row with the lowest cycle
            if 'cycle' in df.columns:
                lowest_cycle_row = df.loc[df['cycle'].idxmin()]
            else:
                if verbose:
                    print(f"  Device {device}: No 'cycle' column found.")
                processed_data[round_number][device] = df
                continue

            # Print values for this device if verbose is True
            if verbose:
                print(f"\nDevice {device} (Lowest Cycle = {lowest_cycle_row['cycle']}):")

            # Add values for each numeric column
            for col in columns_to_process:
                value_at_lowest_cycle = lowest_cycle_row[col]

                # Print details if verbose
                if verbose:
                    print(f"  {col}: {value_at_lowest_cycle:.6f} (from cycle: {lowest_cycle_row['cycle']})")

                # Add a new column with the value at the lowest cycle
                df[f'init_{col}'] = value_at_lowest_cycle

            # Store the updated DataFrame in the result dictionary
            processed_data[round_number][device] = df

        if verbose:
            print(f"\nRound {round_number} Summary:")
            print(f"Devices processed: {len(devices)}")

    return processed_data


#-----------------------------------------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------------------------------------------

def devices_combined_into_one_df(data_dict, rounds):
    """
    Combines device data by rounds into a dictionary of DataFrames.
    
    Parameters:
    - data_dict: Dictionary of format {round_number: {device: DataFrame}}
    - rounds: Integer or list of round numbers to combine
    
    Returns:
    - If single round: DataFrame with combined device data
    - If multiple rounds: Dictionary {round_number: DataFrame} with combined device data per round
    """
    if isinstance(rounds, (int, float)):
        rounds = [rounds]

    if len(rounds) == 1:
        combined_df = pd.DataFrame()
        round_number = rounds[0]

        if round_number not in data_dict:
            print(f"Round {round_number} not found in data")
            return combined_df

        for device, df in data_dict[round_number].items():
            df_copy = df.copy()
            df_copy['device'] = device
            combined_df = pd.concat([combined_df, df_copy], ignore_index=True)

        if not combined_df.empty:
            combined_df = combined_df.sort_values(['device']).reset_index(drop=True)

        return combined_df

    else:
        rounds_dict = {}
        for round_number in rounds:
            if round_number not in data_dict:
                print(f"Round {round_number} not found in data")
                continue

            combined_df = pd.DataFrame()
            for device, df in data_dict[round_number].items():
                df_copy = df.copy()
                df_copy['device'] = device
                combined_df = pd.concat([combined_df, df_copy], ignore_index=True)

            if not combined_df.empty:
                combined_df = combined_df.sort_values(['device']).reset_index(drop=True)
                rounds_dict[round_number] = combined_df

        return rounds_dict
    

#------------------------------------------------------------------------------------------------------------------------------------
def combine_all_rounds(rounds_dict):
    """
    Combines data from multiple rounds into a single DataFrame.

    Parameters:
    - rounds_dict: Dictionary of format {round_number: DataFrame}

    Returns:
    - DataFrame containing data from all rounds
    """
    combined_df = pd.concat(rounds_dict.values(), ignore_index=True)

    if not combined_df.empty:
        combined_df = combined_df.sort_values(['round', 'device']).reset_index(drop=True)

    return combined_df

#-----------------------------------------------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------
def systematic_sample_device_data(data_dict, rounds, devices, column_to_sample='min_temperature', n_samples=320, verbose=False):
    """
    Sample data points at regular intervals for multiple devices and rounds, based on a specified column.
    
    Parameters:
    - data_dict: Dictionary of format {round: {device: DataFrame}}
    - rounds: list of round numbers to sample
    - devices: list of device numbers to sample
    - column_to_sample: The column to use for sampling (default='min_temperature')
    - n_samples: number of samples per device (default=320)
    - verbose: Boolean to control printing of processing details (default=False)
    
    Returns:
    - Dictionary of format {round: {device: DataFrame}} containing sampled data
    """
    sampled_data_dict = {}
    failed_devices = []  # To track devices with fewer samples than expected
    
    for round_num in rounds:
        sampled_data_dict[round_num] = {}
        
        for device in devices:
            try:
                # Get data for current device and round
                device_data = data_dict[round_num][device]
                
                # Ensure the specified column exists
                if column_to_sample not in device_data.columns:
                    raise ValueError(f"Column '{column_to_sample}' not found in data for Round {round_num}, Device {device}")
                
                # Sort by the specified column
                device_data = device_data.sort_values(column_to_sample)
                column_values = device_data[column_to_sample].values
                
                # Create uniform bins based on the specified column
                col_min = column_values.min()
                col_max = column_values.max()
                
                # Create evenly spaced points in the specified column
                target_values = np.linspace(col_min, col_max, n_samples)
                
                # Find closest actual points in the column
                sampled_indices = []
                for target in target_values:
                    idx = np.argmin(np.abs(column_values - target))
                    if idx not in sampled_indices:  # Avoid duplicates
                        sampled_indices.append(idx)
                
                # Store sampled data in the same format as input
                sampled_df = device_data.iloc[sampled_indices]
                sampled_data_dict[round_num][device] = sampled_df
                
                # Check if the number of samples is less than expected
                if len(sampled_df) < n_samples:
                    failed_devices.append(
                        {
                            "round": round_num,
                            "device": device,
                            "expected_samples": n_samples,
                            "collected_samples": len(sampled_df),
                        }
                    )
                
            except KeyError:
                # If data is missing for a specific round/device, skip
                continue
            except ValueError as e:
                # Handle missing column error
                continue
    
    # Print a summary if there are devices with fewer samples than expected
    if failed_devices:
        print("\nSummary of Devices with Fewer Samples than Expected:")
        print("-" * 60)
        for failure in failed_devices:
            print(
                f"Round {failure['round']}, Device {failure['device']}: "
                f"Expected {failure['expected_samples']} samples, "
                f"Collected {failure['collected_samples']} samples"
            )
        print("-" * 60)
    else:
        print("All devices sampled successfully with the expected number of samples.")
    
    return sampled_data_dict
#--------------------------------------------------------------------------------------------


def systematic_sample_device_data2(data_dict, rounds, devices, column_to_sample='min_temperature', n_samples=320, verbose=False):
    """
    Sample data points at regular intervals for multiple devices and rounds, based on a specified column.
    
    Parameters:
    - data_dict: Dictionary of format {round: {device: DataFrame}}
    - rounds: list of round numbers to sample
    - devices: list of device numbers to sample
    - column_to_sample: The column to use for sampling (default='min_temperature')
    - n_samples: number of samples per device (default=320)
    - verbose: Boolean to control printing of processing details (default=False)
    
    Returns:
    - Dictionary of format {round: {device: DataFrame}} containing sampled data
    """
    sampled_data_dict = {}
    
    for round_num in rounds:
        if verbose:
            print(f"\nProcessing Round {round_num}")
            print("-" * 60)
        
        sampled_data_dict[round_num] = {}
        
        for device in devices:
            try:
                # Get data for current device and round
                device_data = data_dict[round_num][device]
                
                # Ensure the specified column exists
                if column_to_sample not in device_data.columns:
                    raise ValueError(f"Column '{column_to_sample}' not found in data for Round {round_num}, Device {device}")
                
                # Sort by the specified column
                device_data = device_data.sort_values(column_to_sample)
                column_values = device_data[column_to_sample].values
                
                # Create uniform bins based on the specified column
                col_min = column_values.min()
                col_max = column_values.max()
                
                # Create evenly spaced points in the specified column
                target_values = np.linspace(col_min, col_max, n_samples)

                print(target_values)
                
                # Find closest actual points in the column
                sampled_indices = []
                for target in target_values:
                    idx = np.argmin(np.abs(column_values - target))
                    if idx not in sampled_indices:  # Avoid duplicates
                        sampled_indices.append(idx)
                
                # Store sampled data in the same format as input
                sampled_df = device_data.iloc[sampled_indices]
                sampled_data_dict[round_num][device] = sampled_df
                
                if verbose:
                    print(f"Device {device}: {len(sampled_df)} samples")
                
            except KeyError:
                if verbose:
                    print(f"Warning: Data not found for Round {round_num}, Device {device}")
                continue
            except ValueError as e:
                if verbose:
                    print(str(e))
                continue
        
        if verbose:
            print('systematic_sample_device_data')
            print(f"\nRound {round_num} Summary:")
            print(f"Devices processed: {len(sampled_data_dict[round_num])}")
            print('-------------------------------------------------------')
    
    return sampled_data_dict

def systematic_sample_device_data_v2(data_dict, rounds, devices, column_to_sample, n_samples=320, verbose=False):
    """
    Sample data points at regular intervals for multiple devices and rounds, ensuring consistent sampling.
    
    Parameters:
    - data_dict: Dictionary of format {round: {device: DataFrame}}
    - rounds: List of round numbers to sample
    - devices: List of device numbers to sample
    - column_to_sample: Column to base sampling on (e.g., 'cycle' or 'min_temperature')
    - n_samples: Number of samples per device (default=320)
    - verbose: Boolean to control printing of processing details (default=False)
    
    Returns:
    - Dictionary of format {round: {device: DataFrame}} containing sampled data
    """
    sampled_data_dict = {}

    for round_num in rounds:
        if verbose:
            print(f"\nProcessing Round {round_num}")
            print("-" * 60)
        
        sampled_data_dict[round_num] = {}

        for device in devices:
            try:
                # Get data for current device and round
                device_data = data_dict[round_num][device]
                
                # Sort by the column to sample
                device_data = device_data.sort_values(column_to_sample)
                sample_values = device_data[column_to_sample].values
                
                # Ensure n_samples does not exceed the number of available values
                n_samples_to_take = min(n_samples, len(sample_values))
                
                # Select evenly spaced indices based on the sorted values
                sampled_indices = np.linspace(0, len(sample_values) - 1, n_samples_to_take, dtype=int)
                
                # Get sampled data
                sampled_df = device_data.iloc[sampled_indices]
                sampled_data_dict[round_num][device] = sampled_df
                
                if verbose:
                    print(f"Device {device}: {len(sampled_df)} samples")
            
            except KeyError:
                if verbose:
                    print(f"Warning: Data not found for Round {round_num}, Device {device}")
                continue

        if verbose:
            print('-------------------------------------------------------')

    return sampled_data_dict

#------------------------------------------------------------------------------------------------------
def convert_temperatures_to_kelvin(m_average_data, temp_columns):
    """
    Convert multiple temperature columns in all DataFrames of m_average_data to Kelvin.

    Parameters:
    - m_average_data: Nested dictionary {round_number: {device: DataFrame}}
    - temp_columns: List of temperature column names to convert (e.g., ['min_temperature', 'max_temperature'])
    """
    for round_num, devices in m_average_data.items():  # Loop through rounds
        for device_num, df in devices.items():  # Loop through devices and their DataFrames
            for temp_column in temp_columns:  # Loop through each temperature column
                if temp_column in df.columns:
                    df[temp_column] += 273.15  # Convert to Kelvin
                else:
                    print(f"    Warning: Column '{temp_column}' not found in DataFrame for round {round_num}, device {device_num}.")
