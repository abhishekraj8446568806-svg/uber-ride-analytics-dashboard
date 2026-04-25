# Import Libraries & Load Dataset

# Import pandas, numpy, and matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Excel file and parse the correct sheet
path = 'Uber_ride_analysis_dataset.csv'
df = pd.read_csv(path,header=0)
df.head()

# Data Cleaning

# Standardize column names
# Convert to lowercase and replace spaces with underscores
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.columns

# Identify missing values in each column
print(df.isnull().sum())

# Drop rows with missing trip status or trip cost
df = df.dropna(subset=['trip_status', 'trip_cost'])

# Fill missing payment method with mode
df['payment_method'].fillna(df['payment_method'].mode()[0], inplace=True)

# Fill missing driver ID with -1 (indicates unassigned)
df['driver_id'].fillna(-1, inplace=True)

#  Handle timestamps conditionally
# Drop rows where timestamps are missing only for 'Trip Completed'
df = df[~((df['trip_status'] == 'Trip Completed') & (df['start_timestamp'].isna() | df['drop_timestamp'].isna()))]
# df = df[((df['trip_status']=='Trip Completed') & (df['start_timestamp'].isna() | df['drop_timestamp'].isna()))]

# Identify missing values in each column
print(df.isnull().sum())

# Convert date columns to datetime
# date_columns = ['request_timestamp', 'start_timestamp', 'drop_timestamp']
# for col in date_columns:
#     df[col] = pd.to_datetime(df[col])
    


#  List all numeric columns
numeric_columns = df.select_dtypes(include='number').columns.tolist()
print(numeric_columns)

## Outlier Detection and Treatment

# Plot boxplot for trip cost
plt.figure(figsize=(8, 4))
df['trip_cost'].plot(kind='box', vert=False)
plt.title('trip_cost Distribution')
plt.show()

# Detect and cap outliers in trip cost using IQR
Q1 = df['trip_cost'].quantile(0.25)
Q3 = df['trip_cost'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
df['trip_cost'] = np.where(df['trip_cost'] > upper_bound, upper_bound, df['trip_cost'])

# Plot boxplot for trip cost
plt.figure(figsize=(8, 4))
df['trip_cost'].plot(kind='box', vert=False)
plt.title('trip_cost Distribution - Outliers Handled')
plt.show()

#  Repeat for extra tip
plt.figure(figsize=(8, 4))
df['extra_tip'].plot(kind='box', vert=False)
plt.title('Extra Tip Distribution')
plt.show()


# 📊 Exploratory Data Analysis (EDA)

Now that the data cleaning is done, let's perform EDA to understand the dataset better and derive insights.


## 1. Dataset Overview

# Understand the shape of the dataset
df.shape

# Understand the datatypes of the columns
df.dtypes

## 2. Trip Status Breakdown

# Plot the value_counts for different trip statuses
df['trip_status'].value_counts().plot(kind='bar')
plt.title("Trip Status Distribution")

## 3. Trip Cost Distribution

# Plot Trip Cost Distribution
# Define bin edges at intervals of 100, starting from 0 up to the maximum trip cost
# Plot the histogram with the specified bins
# Turn off grid

import numpy as np
import matplotlib.pyplot as plt

max_cost = df['trip_cost'].max()
bins = np.arange(0, max_cost + 100, 100)

df['trip_cost'].hist(bins=bins, edgecolor='black')
plt.title("Trip Cost Distribution")
plt.xlabel("Trip Cost")
plt.ylabel("Frequency")
plt.grid(False)  
plt.show()

## 4. Analyze Payment Methods

# Analyze trip costs across payment methods by calculating the average, median, and number of trips for each method,
# and sort the results to highlight the most expensive payment types on average

df.groupby('payment_method')['trip_cost'].agg(['mean', 'median', 'count']).sort_values(by='mean', ascending=False)

## 5. Trip Duration Analysis


# Compute the trip duration in minutes by taking the difference between drop and start timestamps
df['trip_duration_minutes'] = (df['drop_timestamp'] - df['start_timestamp']).dt.total_seconds() / 60

# Plot distribution of trip duration
# Define bins with a fixed size (e.g., every 10 minutes)
# Grid should not be visible
import numpy as np
import matplotlib.pyplot as plt

max_duration = df['trip_duration_minutes'].max()
bins = np.arange(0, max_duration + 10, 10)  # bins of size 10 minutes

# Plot histogram
df['trip_duration_minutes'].hist(bins=bins, edgecolor='black')
plt.title("Trip Duration Distribution")
plt.xlabel("Trip Duration (minutes)")
plt.ylabel("Frequency")
plt.grid(False)  
plt.show()

# Transforming Data

## 1. Total Trip Cost

# Calculate Total cost of the trip

df['total_cost'] = df['trip_cost'] + df['extra_tip']

## 2. DateTime Columns

# Convert Time Columns to DateTime Format

df['request_timestamp'] = pd.to_datetime(df['request_timestamp'])
df['start_timestamp'] = pd.to_datetime(df['start_timestamp'])
df['drop_timestamp'] = pd.to_datetime(df['drop_timestamp'])

## 3. Extract Date Time

# Extract Date and Time Components
'''
From the relevant datetime column(s), extract:
1. Date (YYYY-MM-DD) (request_date, start_date, drop_date)
2. Day of the week (e.g., Monday, Tuesday) (request_day, start_day, drop_day)
3. Exact time (HH:MM:SS) (request_time, start_time, drop_time)
4. Hour of the day (0–23) (request_hour, start_hour, drop_hour)
'''

df['request_date'] = df['request_timestamp'].dt.date
df['start_date'] = df['start_timestamp'].dt.date
df['drop_date'] = df['drop_timestamp'].dt.date

df['request_day'] = df['request_timestamp'].dt.day_name()
df['start_day'] = df['start_timestamp'].dt.day_name()
df['drop_day'] = df['drop_timestamp'].dt.day_name()

df['request_time'] = df['request_timestamp'].dt.time
df['start_time'] = df['start_timestamp'].dt.time
df['drop_time'] = df['drop_timestamp'].dt.time

df['request_hour'] = df['request_timestamp'].dt.hour
df['start_hour'] = df['start_timestamp'].dt.hour
df['drop_hour'] = df['drop_timestamp'].dt.hour

## 4. Ride Delay

# Calculating the Ride delay.
'''
- Determine the delay between the ride request and actual trip start.
- Add a column ride_delay reflecting this delay in hour value.
'''

df["ride_delay"] = ((df["drop_timestamp"] - df["start_timestamp"]).dt.total_seconds() / 3600).round(2)

## 5. Cancellation Reasons

# Determine Cancellation reasons
# - Assign a cancellation reason to each trip based on the driver_id and trip_status columns using nested np.where() statements.
'''
1. If the driver_id is -1 and the trip status is 'No Cars Available', it indicates that no cab was assigned,
so the cancellation reason is set to 'No Cabs'.
2. If the driver_id is -1 and the trip status is 'Trip Cancelled', it means the passenger canceled the trip before a driver was assigned,
so the reason is 'Passenger'.
3. If a driver was assigned (driver_id not equal to -1) and the trip status is 'Trip Cancelled', it indicates that the driver canceled the trip,
so the reason is 'Driver'.
4. For all other cases—where the trip was completed successfully—the cancellation reason is set to 'Trip Completed'.

'''

df['cancellation_reason'] = np.where(
    (df['driver_id'] == -1) & (df['trip_status'] == 'No Cars Available'), 'No Cabs',
    np.where(
        (df['driver_id'] == -1) & (df['trip_status'] == 'Trip Cancelled'), 'Passenger',
        np.where(
            (df['driver_id'] != -1) & (df['trip_status'] == 'Trip Cancelled'), 'Driver',
            'Trip Completed'
        )
    )
)

# Analysis

Selecting columns relevant to our analysis

# Creating a New Dataframe selecting columns relevant to our analysis
# All analysis going forward will be done on this new dataframe created

new_df = df[['request_id', 'driver_id', 'trip_status', 'request_day', 'request_hour','start_day', 'start_hour', 'drop_day', 'drop_hour', 'trip_cost',
            'ride_delay', 'weather', 'cancellation_reason']]

new_df.columns

# Create a bar chart / Count plot (using Seaborn) that shows the number of ride requests for each day of the week (request_day column)
'''
1. Import Seaborn library as sns
2. Use .countlot() to create the chart
3. Assign Title to the chart
    - Title should be → “Request Count Vs Day”
4. Assign X & Y labels
5. (Optional) Assign Figure Size
    - Try (12, 5) or (10, 6) 
6. (Optional) Assign a colour Palette to the chart
    - Try 'Set1' or 'Pastel1' or 'coolwarm' or ‘Blues’, etc.
'''
import seaborn as sns

# plt.figure(figsize = (10, 6))

sns.countplot(
    new_df,
    x='request_day',
    hue='request_day',  # This applies palette correctly
    # palette='BuGn',  # Try 'Set1', 'Pastel1', 'husl', 'coolwarm', etc.
    legend=False
)

plt.title("Request Count Vs Day")
plt.xlabel("Day of the Week")
plt.ylabel("Number of Requests")
plt.show()

# Create a bar chart / Count plot (using Seaborn) that shows the number of ride requests for each day of the week (request_day column)
'''
1. Use .countlot() to create the chart
2. Assign Title to the chart
    - Title should be → “Request Count Vs Hour”
3. Assign X & Y labels
4. Assign Figure Size
    - Try (12 * 5), (10 * 6) 
5. Assign a colour Palette to the chart
    - Try 'Set1' or 'Pastel1' or 'coolwarm' or ‘Blues’, etc.
'''

plt.figure(figsize = (12, 5))

sns.countplot(
    new_df,
    x='request_hour',
    hue='request_hour',  # This applies palette correctly
    palette='Blues',  # Try 'Set1', 'Pastel1', 'husl', 'coolwarm', etc.
    legend=False
)

plt.title("Request Count Vs Hour")
plt.xlabel("Hour of the Day")
plt.ylabel("Number of Requests")
plt.show()

# Calculate the trip status bifurcation (normalize to percentage)
'''
1. Count the values to create a variable for each value in the trip status column
2. Normalize these to get the percentage of each
    - This calculates the percentage of each unique value in the respective column.
3. Store it in a variable → “trip_status_rates”
4. Print the variable to see the output
'''

trip_status_rates = new_df['trip_status'].value_counts(normalize=True) * 100
print(trip_status_rates)

# Plotting a pie chart for Trip Status
'''
1. Assign Figure Size
    - Try (10, 10) or (8,  8)
2. Use .pie() to create the chart
    - Add the labels as .index from the calculated variable above
    - Add Percentage of each segment on the chart
        - Use autopct = '%1.1f%%', within the pie chart
        - This formats the percentage to show one decimal point.
3. Assign Title to the chart
    - Title should be → “Trip Status Bifurcation”
'''

plt.figure(figsize=(8, 8))
plt.pie(trip_status_rates , labels = trip_status_rates.index, autopct = '%1.1f%%')

plt.title("Trip Status Bifurcation")
plt.show()

# Calculate the cancellation trends based on the 'cancellation_reason' column & plot the same
'''
1. Print the cancellation reason counts for cancelled trips.
    - Assign a variable → ‘cancellation_trends’ to find the value counts of each cancellation reason where trip status is ‘Trip Cancelled’
2. Plot a pie chart to visualize the same.
    - Assign Title → ‘Trip Cancellation Trend’ to the chart
4. (Optional) Plot a bar chart for the same.
    - Assign X & Y labels along with title
'''

cancellation_trends = new_df[new_df['trip_status'] == 'Trip Cancelled']['cancellation_reason'].value_counts()

# Print the cancellation trends
print(f"Cancellation Trends (Driver vs Passenger):")
print(cancellation_trends)

plt.figure(figsize=(3, 3))
plt.pie(cancellation_trends , labels = cancellation_trends.index, autopct = '%1.1f%%')

plt.title("Trip Cancellation Trend")
plt.show()

# Looking deeper into not completed trips

# TODO 1 : Creating a New Dataframe for only incomplete rides.
# Verify the columns
# df.columns
new_df = df[df['trip_status'] !='Trip Completed']
new_df = new_df[['request_id' , 'pickup_point' , 'drop_point' , 'driver_id' ,'trip_status' ,'payment_method' , 'weather' , 'request_day' , 'request_hour' ,'cancellation_reason' ]] 
new_df
# new_df.columns

# TODO 2 : Proportion of Incomplete Rides’ Trip Statuses
'''
Create a Pie Chart to see the proportion
1. Assign Figure size
2. Assign Title → “Proportion of Incomplete Rides’ Trip Statuses”
3. Assign labels
4. Assign Percentage values to the chart
5. Assign Colo r
6. (Optional) Assign Start Angle
'''

plt.figure(figsize=(8,8))

proportion_trip = new_df['trip_status'].value_counts(normalize=True)

plt.pie(
    proportion_trip,
    labels=proportion_trip.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=['red','green','blue']  # safer to match categories
)

plt.title("Proportion of Incomplete Ride Statuses")

plt.show()

# TODO 3 : Which is the day with the most rides cancelled?
# What is the split of cancelled trips' status across days?
'''
1. Assign Figure Size
    - Try (12, 5) or (10, 6)
2. Use .histplot() to create the chart
    - Request Day Vs Cancellation Reason
    - Bin the data into 7 (Number of days in a week)
3. Assign Title to the chart
    - Title should be → “Incomplete Rides by Day and Reason”
4. Assign X & Y labels
    - X Label → "Day"
    - Y Label → “Number of Incomplete Rides”
5. (Optional) Assign a colour Palette to the chart
    - Research on how to add colors to Stacked Histogram
'''

plt.figure(figsize=(12,6))

sns.histplot(
    data=new_df,
    x='request_day',
    hue='cancellation_reason',
    multiple='stack',
    bins = 7
)

plt.title('Incomplete Rides by Day and Reason')
plt.xlabel('Day')
plt.ylabel('Number of Incomplete Rides')

plt.show()

# TODO 4 : What are the times of the days with most rides cancelled?
# What are the reasons for the cancellations throughout the day?
'''
1. Assign Figure Size
    - Try (12, 5) or (10, 6)
2. Use .histplot() to create the chart
    - Request Day Vs Cancellation Reason
    - Bin the data into 24 (Number of hours in a day)
3. Assign Title to the chart.
    - Title should be → “Incomplete Rides by Hour and Reason”
4. Assign X & Y labels
    - X Label → "Hour of the Day"
    - Y Label → “Number of Incomplete Rides”
5. (Optional) Assign a colour Palette to the chart
    - Research on how to add colors to Stacked Histogram
'''
new_df.columns
plt.figure(figsize=(12, 5))

sns.histplot(x ='request_hour'  , hue = 'cancellation_reason', data = new_df , multiple = 'stack',bins=24)

plt.title('Incomplete Rides by Hour and Reason')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Incomplete Rides')


plt.show()

# TODO 5 : Relationship Between Weather & Cancellation Reason
'''
To see any relationship between weather and cancellation reason, let’s try to see the grouped count of cancellations.
1. Create a grouped dataframe with count using Weather and Cancellation Reasons as pivots.
2. Plot a bar chart with Weather and Count of grouped values
3. Assign Title to the chart
    - Title should be → ‘Cancellations by Weather and Reason per Ride Type’
4. Assign X & Y labels
    - X Label → ‘Weather Conditions’
    - Y Label → ‘Count of Cancellations’
'''
new_grouped_data = new_df.groupby(['weather','cancellation_reason']).size().reset_index(name='count')
new_grouped_data

sns.barplot(x ='weather' , y = 'count' , data = new_grouped_data , hue = 'cancellation_reason')
plt.title('Cancellations by Weather and Reason per Ride Type')
plt.xlabel('Weather Conditions')
plt.ylabel('Count of Cancellations')
plt.show()

