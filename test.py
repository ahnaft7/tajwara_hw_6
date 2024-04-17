"""
Ahnaf Tajwar
Class: CS 677
Date: 4/20/24
Homework Problems # 1-3
Description of Problem (just a 1-2 line summary!): These problems are to 
"""
import pandas as pd

# Define column names
column_names = ['f1', 'f2', 'C', 'f4', 'f5', 'f6', 'f7', 'L']

# Read the file using a regular expression pattern as the delimiter
df = pd.read_csv("seeds_dataset.txt", delimiter='\s+', header=None, names=column_names)

print(df)

buid = 0
R = buid % 3
print(f'\nR = {R}: class L = 1 (negative) and L = 2 (positive)')

# Filter rows where L is 1 or 2
df = df[df['L'].isin([1, 2])]

# Print the filtered DataFrame
print(df)

