import sys
import pandas as pd
import getMAE as getMAE
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestRegressor as rfr

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
predicted_melb_prices = melbourne_model.predict(XM)


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

#**************Melbourne Validation(Bad Method)****************
print(mae(yM, predicted_melb_prices))

#**************Melbourne Validation(Good Method)****************
train_XM, val_XM, train_yM, val_yM = tts(XM, yM, random_state=1)
melbourne_model= dtr(random_state=1)
melbourne_model.fit(train_XM, train_yM)
val_melb_predictions = melbourne_model.predict(val_XM)
print(mae(val_yM, val_melb_predictions))

#**************Iowa Validation(Good Method)****************
train_XI, val_XI, train_yI, val_yI = tts(XI, yI, random_state=1)
iowa_model= dtr()
iowa_model.fit(train_XI, train_yI)
val_iowa_predictions = iowa_model.predict(val_XI)
print(mae(val_yI, val_iowa_predictions))

#*************Differ max_leaf_nodes*****************
for max_leaf_nodes in [400, 450, 500, 550, 600]:
    my_mae = getMAE.get_mae(max_leaf_nodes, train_XM, val_XM, train_yM, val_yM)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

#*************Differ max_leaf_nodes*****************
for max_leaf_nodes in [5, 50, 500, 1000, 5000]:
    my_mae = getMAE.get_mae(max_leaf_nodes, train_XI, val_XI, train_yI, val_yI)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
#*************Find best max_leaf_nodes Iowa*****************
candidate_max_leaf_nodes = [5, 50, 500, 1000, 5000]
scores = {leaf_size: getMAE.get_mae(leaf_size, train_XI, val_XI, train_yI, val_yI) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)
print(best_tree_size)

#****Forest Model Melborne****
forest_model = rfr(random_state=1)
forest_model.fit(train_XM, train_yM)
melb_preds = forest_model.predict(val_XM)
print(mae(val_yM, melb_preds))