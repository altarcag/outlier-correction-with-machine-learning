import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

original_dataset = pd.read_csv("./ankara_dataset.csv")
original_dataset

org_dataset_edited = original_dataset.drop(columns=['Country', 'City'])
org_time_set = org_dataset_edited.drop(columns=['AvgTemperature'])
org_temp_set = original_dataset['AvgTemperature']
org_dataset_edited

# name of the temp column
temperature_column = 'AvgTemperature'

# defining a threshold for outliers (which can be adjusted based on what interval is required)
lower_bound = -50
upper_bound = 50

# Filter rows based on the temperature column
outliers_excluded = original_dataset[(original_dataset[temperature_column] >= lower_bound) & (original_dataset[temperature_column] <= upper_bound)]

# Save the filtered DataFrame back to a new CSV file or update the existing one
outliers_excluded.to_csv('ankara_edited.csv', index=False)

outliers_included = original_dataset[(original_dataset[temperature_column] <= lower_bound) | (original_dataset[temperature_column] >= upper_bound)]
outliers_included.to_csv('outlier_dataset.csv', index=False)

outlier_data = pd.read_csv("./outlier_dataset.csv")

ankara_edited = pd.read_csv("./ankara_edited.csv")
ankara_edited

#making it all numeric now
# "THE" dataset ;)

dataset = ankara_edited.drop(columns=['Country', 'City'])
dataset

x = dataset.drop(columns=['AvgTemperature'])
y = dataset['AvgTemperature']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = DecisionTreeRegressor()
model.fit(x_train, y_train)

predictions = model.predict(x_test)


# Make predictions on the entire dataset
all_predictions = model.predict(org_time_set)
# Identify outliers in the test set
outlier_mask = (org_temp_set <= lower_bound) | (org_temp_set >= upper_bound)
# Replace outliers with the predicted data
org_temp_set[outlier_mask] = all_predictions[outlier_mask]


specific_prediction = model.predict([[1,4,2018]]) #this is one of the days with an outlier
print(specific_prediction)

# error
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')

# Display the updated dataset with predictions next to the average temperature values in the original dataset
org_dataset_w_prdctxs = org_dataset_edited.copy()  # creates a copy of the original dataset
org_dataset_w_prdctxs['PredictedTemperature'] = all_predictions  # adds a new column with the predictions

print("Updated Dataset with Predicted Temperatures:")
org_dataset_w_prdctxs

org_dataset_w_prdctxs.to_csv('ankara_with_predictions.csv', index=False)

# Create a copy of the original dataset to keep non-outliers unchanged
updated_dataset = org_dataset_edited.copy()

# Identify outliers in the dataset
outlier_mask = (updated_dataset['AvgTemperature'] <= lower_bound) | (updated_dataset['AvgTemperature'] >= upper_bound)

# Create a new column with the original values
updated_dataset['UpdatedTemperature'] = updated_dataset['AvgTemperature']

# Replace outliers with the predicted data
updated_dataset.loc[outlier_mask, 'UpdatedTemperature'] = all_predictions[outlier_mask]

# Display the updated dataset with the new column
print("Updated Dataset with Original and Predicted Temperatures:")

# Save the updated dataset to a new CSV file
updated_dataset.to_csv('ankara_updated.csv', index=False)

corrected_dataset = pd.read_csv("./ankara_updated.csv")
corrected_dataset



#plotting
plt.figure(figsize=(20, 4))

#Creating a datetime single-column combining year month and day
org_dataset_edited['Date'] = pd.to_datetime(org_dataset_edited[['Month', 'Day', 'Year']])

#plotting actual data now (with all the mfing outliers)
plt.plot(org_dataset_edited['Date'], org_dataset_edited['AvgTemperature'], label='Temperatures')
plt.show()

plt.figure(figsize=(20, 4))

dataset['Date'] = pd.to_datetime(dataset[['Month', 'Day', 'Year']])

#plotting
plt.plot(dataset['Date'], dataset['AvgTemperature'], label='Corrected Temperatures')
plt.show()

#plotting the mfing plots
plt.figure(figsize=(20, 4))

#Creating a datetime single-column combining year month and day
org_dataset_edited['Date'] = pd.to_datetime(org_dataset_edited[['Month', 'Day', 'Year']])

#plotting actual data now (with all the mfing outliers)
plt.plot(org_dataset_edited['Date'], org_dataset_edited['AvgTemperature'], label='Temperatures')

#creating a datetime ...
org_dataset_w_prdctxs['Date'] = pd.to_datetime(org_dataset_w_prdctxs[['Month', 'Day', 'Year']])

#plotting ...
plt.plot(org_dataset_w_prdctxs['Date'], org_dataset_w_prdctxs['PredictedTemperature'], label='Corrected Temperatures')

plt.show()

# Plotting the mfing plots
plt.figure(figsize=(20, 4))

# Creating a datetime single-column combining year month and day for org_dataset_edited
org_dataset_edited['Date'] = pd.to_datetime(org_dataset_edited[['Month', 'Day', 'Year']])

# Plotting actual data now (with all the outliers) for specific years
selected_years = [2011, 2012, 2013]  # Add the years you want to plot

# Filter data for the selected years
org_selected_years = org_dataset_edited[org_dataset_edited['Year'].isin(selected_years)]

plt.plot(org_selected_years['Date'], org_selected_years['AvgTemperature'], label='Temperatures (Selected Years)')

# Creating a datetime single-column combining year month and day for updated_dataset
org_dataset_w_prdctxs['Date'] = pd.to_datetime(org_dataset_w_prdctxs[['Month', 'Day', 'Year']])

# Filter data for the selected years
updated_selected_years = org_dataset_w_prdctxs[org_dataset_w_prdctxs['Year'].isin(selected_years)]

# Plotting corrected temperatures for specific years
plt.plot(updated_selected_years['Date'], updated_selected_years['PredictedTemperature'], label='Predicted Temperatures (Selected Years)')

plt.legend()
plt.show()

# Plotting the mfing plots
plt.figure(figsize=(20, 4))

# Creating a datetime single-column combining year month and day for org_dataset_edited
org_dataset_edited['Date'] = pd.to_datetime(org_dataset_edited[['Month', 'Day', 'Year']])

# Plotting actual data now (with all the outliers) for specific years
selected_years = [2011, 2012, 2013]  # Add the years you want to plot

# Filter data for the selected years
org_selected_years = org_dataset_edited[org_dataset_edited['Year'].isin(selected_years)]

plt.plot(org_selected_years['Date'], org_selected_years['AvgTemperature'], label='Temperatures (Selected Years)')


# Creating a datetime single-column combining year month and day for updated_dataset
corrected_dataset['Date'] = pd.to_datetime(corrected_dataset[['Month', 'Day', 'Year']])

# Filter data for the selected years
updated_selected_years = corrected_dataset[corrected_dataset['Year'].isin(selected_years)]

# Plotting corrected temperatures for specific years
plt.plot(updated_selected_years['Date'], updated_selected_years['UpdatedTemperature'], label='Corrected Temperatures (Selected Years)')

plt.legend()
plt.show()

#plotting the mfing plots
plt.figure(figsize=(20, 4))

#Creating a datetime single-column combining year month and day
org_dataset_edited['Date'] = pd.to_datetime(org_dataset_edited[['Month', 'Day', 'Year']])

#plotting
plt.plot(org_dataset_edited['Date'], org_dataset_edited['AvgTemperature'], label='Temperatures')

#...
corrected_dataset['Date'] = pd.to_datetime(corrected_dataset[['Month', 'Day', 'Year']])

#plotting
plt.plot(corrected_dataset['Date'], corrected_dataset['UpdatedTemperature'], label='Corrected Temperatures')

plt.legend()
plt.show()