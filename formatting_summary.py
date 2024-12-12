### This file is used to transform a .txt summary file into a 
### .csv file that has the correct ID column

### The .txt summary files for the desired match need to have
### already been created and need to be placed in the same place as this file

import pandas as pd
import re
import csv

def txt_to_csv(input_file, output_file):
    """
    Convert a formatted .txt file to a .csv file.
    Args:
        input_file (str): Path to the input .txt file.
        output_file (str): Path to the output .csv file.
    """
    with open(input_file, 'r', encoding='utf-8') as txt_file:
        data = txt_file.read()
    
    # Regular expression to extract each timestamp block
    blocks = re.findall(r'\*{50,}\n\*.*?Timestamp: (\d+).*?\*\n(.*?)\*{50,}', data, re.DOTALL)
    
    # Clean up the content and prepare rows
    rows = [{"Timestamp": timestamp, "Content": content.strip().replace('\n', ' ')}
            for timestamp, content in blocks]
    
    # Write rows to a CSV file
    with open(output_file, 'w', encoding='utf-8', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["Timestamp", "Content"])
        writer.writeheader()
        writer.writerows(rows)

def zipping_IDs_with_dfs(match_df, summary_df):
    """
    Add the ID column to the summary dataframe.
    Args:
        match_df (pd.DataFrame): Dataframe containing the match data.
        summary_df (pd.DataFrame): Dataframe containing the summary data.
    Returns:
        pd.DataFrame: Dataframe with the ID column added.
    """
    # Extract the match ID from the match dataframe
    match_id = int(match_df["MatchID"].values[0])
    
    # Add the ID column to the summary dataframe
    summary_df["ID"] = summary_df["Timestamp"].apply(lambda x: f"{match_id}_{x-1}")
    
    return summary_df

def get_correct_summary_df(match,train_or_eval='train'):
    """
    Get the correct summary dataframe for a given match.
    Args:
        match (str): The match for which to get the summary dataframe.
    Returns:
        pd.DataFrame: The summary dataframe for the given match.
    """
    input_file = "summaries/converted_"+match+".txt"
    output_file_no_index = f"summaries/summary_no_index_{match}.csv"
    output_file = f"summaries/summary_{match}.csv"
    txt_to_csv(input_file, output_file_no_index)
    if train_or_eval=='train' or train_or_eval=='eval':
        match_df_loc = "challenge_data/"+train_or_eval+"_tweets/"+match+".csv"
    else:
        raise ValueError("train_or_eval must be either 'train' or 'eval'")
    match_df = pd.read_csv(match_df_loc)
    summary_df = pd.read_csv(output_file_no_index)
    
    # Add the ID column to the summary dataframe
    summary_df = zipping_IDs_with_dfs(match_df, summary_df)

    if train_or_eval=='train':
        summary_df = pd.merge(match_df.groupby("ID").agg({"EventType": "max"}),summary_df[["ID","Content"]], on='ID', how='inner')

    summary_df.to_csv(output_file, index=False)
    
    return summary_df