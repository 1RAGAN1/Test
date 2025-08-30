import pandas as pd

# Melbourne data
melbourne_file_path = 'data/Melb/melb_data.csv'  
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 
# print a summary of the data in Melbourne data
#print(melbourne_data.describe())
#print(melbourne_data.head())
print(melbourne_data.columns)


# Iowa data
iowa_file_path = 'data/Iowa/train.csv'
# Fill in the line below to read the file into a variable home_data
home_data = pd.read_csv(iowa_file_path)
# Print summary statistics in next line
#print(home_data.describe())
# What is the average lot size (rounded to nearest integer)?
avg_lot_size = home_data['LotArea'].mean().round()
# As of today, how old is the newest home (current year - the date in which it was built)
newest_home_age = 2025 - home_data['YearBuilt'].max()