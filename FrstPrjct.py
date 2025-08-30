import sys
import pandas as pd
from sklearn.tree import DecisionTreeRegressor as dtr

# *****************Import Melbourne data********************
melbourne_file_path = 'data/Melb/melb_data.csv'  
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 
# print a summary of the data in Melbourne data
#print(melbourne_data.describe())
print(melbourne_data[['Price']].head())
#print(melbourne_data.columns)

# **********************Import Iowa data*********************
iowa_file_path = 'data/Iowa/train.csv'
# Fill in the line below to read the file into a variable iowa_data
iowa_data = pd.read_csv(iowa_file_path)
# Print summary statistics in next line
#print(iowa_data.describe())
# What is the average lot size (rounded to nearest integer)?
avg_lot_size = iowa_data['LotArea'].mean().round()
#print(avg_lot_size)
# As of today, how old is the newest home (current year - the date in which it was built)
newest_home_age = 2025 - iowa_data['YearBuilt'].max()
#print(newest_home_age)
# print cloumn names
#print(iowa_data.columns)
print(iowa_data[['SalePrice']].head())
#sys.exit(0)

# ****************Prepare melbourne data*********************
# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)
# assign Price col to y
yM = melbourne_data.Price
#choose features from melbourne_data and assign to x
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
XM = melbourne_data[melbourne_features]
#print(XM.describe())
#print(XM.head())

# *****************Define model Melborne*********************
melbourne_model = dtr(random_state=1)
# Fit model to data
melbourne_model.fit(XM, yM)
print("Making predictions for the following 5 houses:")
print(XM.head())
print("The predictions are")
print(melbourne_model.predict(XM.head()))

# *********************Prepare iowa data*********************
yI = iowa_data.SalePrice
iowa_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
XI = iowa_data[iowa_features]
#print(XI.describe())
#print(XI.head())

# ************************Define model Iowa******************
iowa_model = dtr(random_state=1)
# Fit model to data
iowa_model.fit(XI, yI)
print("Making predictions for the following 5 houses:")
print(XI.head())
print("The predictions are")
print(iowa_model.predict(XI.head()))
#print('full predection')
#print(iowa_model.predict(XI))