import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ipyleaflet import Map, basemaps, Marker, LayerGroup, CircleMarker, WidgetControl

data_set = pd.read_csv('assets/real_estate.csv', sep=';')
populations = data_set['level5'].drop_duplicates()

def get_address_with_price(data_frame : pd.DataFrame, price_value : np.int64) -> str:
    #Find the column of the highest price house. Using the head() function to prevent multiple values
    price_row = data_frame.loc[data_frame['price'] == price_value].head(1)
    #Converting the adress value to a String and modifying it for visuals
    return price_row['address'].to_string().replace(' ', ', ', 1)

#Exercise 01
max_price = data_set['price'].max()

print(f"The house at {get_address_with_price(data_set, max_price)} is the most expensive and its price is {max_price} USD")

#Exercise 02
min_price = data_set['price'].min()

print(f"The house at {get_address_with_price(data_set, min_price)} is the cheapest and its price is {min_price} USD")


def get_address_with_surface(data_frame : pd.DataFrame, surface_value : np.int64) -> str:
    #Find the column of the biggest surface house. Using the head() function to prevent multiple values
    surface_row = data_frame.loc[data_frame['surface'] == surface_value].head(1)
    #Converting the adress value to a String and modifying it for visuals
    return surface_row['address'].to_string().replace(' ', ',', 1)

#Exercise 03

biggest_surface = data_set['surface'].max()


smallest_surface = data_set['surface'].min()

print(f"The biggest house is located on {get_address_with_surface(data_set, biggest_surface)} and its surface is {biggest_surface} meters")
print(f"The smallest house is located on {get_address_with_surface(data_set, smallest_surface)} and its surface is {smallest_surface} meters")

#Exercise 04
def get_population_names() -> str:
    #Removing duplicate populations
    populations_string = ''
    #Looping through the new list to add commas
    for item in populations:
        populations_string += f"{item}, "
    return populations_string


print('Populations: ',get_population_names(), '\n')

#Exercise 05
def check_NAs():
    NAs_in_dataset = data_set.isnull().sum().sum()
    if NAs_in_dataset > 0:
        print(True)
        NAs_in_cols = []
        for column in data_set.columns:
            if data_set[column].isnull().values.any():
                NAs_in_cols.append(column)
        print(f"The data set contains {NAs_in_dataset} NaNs in columns : {', '.join(NAs_in_cols)}")
    else:
        print(False)
check_NAs()

#Exercise 06
#Removing columns with missing values
clean_data_set = data_set.dropna(axis = 1)
print('Dimensions before cleaning: ', data_set.shape, '\n', 'Dimensions after cleaning: ', clean_data_set.shape)

#Exercise 07
selected_rows = data_set.loc[data_set['level5'] == 'Arroyomolinos (Madrid)']
print(f"The mean of the prices in the population 'Arroyomolinos (Madrid)' is {selected_rows.price.mean(skipna = True)}")

#Exercise 08
plt.hist(selected_rows.price)
plt.title('Price Histogram')
plt.savefig('plots/Histogram.png')
plt.show()
#Conclusion:  The majority of the data falls into 300000

def get_population_mean(target_pop : str, target_column : str) -> float:
    selected_rows = data_set.loc[data_set['level5'] == target_pop]
    return selected_rows[target_column].mean(skipna = True)

# Exercise 09
avg_price_1 = get_population_mean('Valdemorillo', 'price')
avg_price_2 = get_population_mean('Galapagar', 'price')

print(f"The average prices in 'Valdemorillo' ({str(avg_price_1)}) and 'Galapagar' ({avg_price_2}) are {'the same' if avg_price_1 == avg_price_2 else 'different'}")

# Exercise 10
#data_set['surface'] = data_set['surface'].fillna(0)
data_set['pps'] = data_set.price / (data_set.surface * 2)
avg_pps_1 = get_population_mean('Valdemorillo', 'pps')
avg_pps_2 = get_population_mean('Galapagar', 'pps')

print(f"The average price per square meter in 'Valdemorillo' ({str(avg_pps_1)}) and 'Galapagar' ({avg_pps_2}) are {'the same' if avg_price_1 == avg_price_2 else 'different'}")

#Exercise 11
plt.scatter(data_set.surface, data_set.price)
plt.title('Surface/Price Scatter Plot')
plt.savefig('plots/Scatter Plot.png')
plt.show()
#Conclusion: The surface remains constant as the price increases, with a few exceptions

#Exercise 12
print(f"The data set contains {len(data_set['realEstate_name'].drop_duplicates())} real estate agencies")

#Exercise 13
pop_with_most_houses = ''
highest_pop_count = 0
for pop in populations:
    selected_pop = len(data_set.loc[data_set['level5'] == pop])
    if selected_pop > highest_pop_count:
        highest_pop_count = selected_pop
        pop_with_most_houses = pop
num_of_houses = len(data_set.loc[data_set['level5'] == pop_with_most_houses])
print(f"The population with most houses is {pop_with_most_houses} with {str(num_of_houses)} houses")


#Exercise 14
south_belt_df = data_set.query('level5 == ["Fuenlabrada", "Leganés", "Getafe", "Alcorcón"]')

#Exercise 15
median_values = south_belt_df.groupby('level5')['price'].median()
median_values_df = pd.DataFrame(median_values)
bar_plot = median_values_df.plot.bar(figsize=(20, 10))
plt.title('Population Price Median')
plt.savefig('plots/Bar Plot.png')


#Exercise 16
def calc_mean_of_column(data_frame : pd.DataFrame, column_name : str):
    return data_frame[column_name].mean()

def calc_variance_of_column(data_frame : pd.DataFrame, column_name : str):
    return data_frame[column_name].var()

for col in ['price', 'rooms', 'surface', 'bathrooms']:
    print(f"The mean of {col} is {str(calc_mean_of_column(south_belt_df, col))} and its variation is {str(calc_variance_of_column(south_belt_df, col))}")


#Exercise 17
for pop in south_belt_df.level5.drop_duplicates():
    max_price = south_belt_df.loc[south_belt_df['level5'] == pop]['price'].max()
    print(f"The most expensive house in {pop} is {get_address_with_price(south_belt_df, max_price)} with a price of {str(max_price)}")

#Exercise 18
#Values normalized using maximum absolute scaling
normalized_values = []
south_belt_df_scaled = south_belt_df.copy()
for pop in south_belt_df.level5.drop_duplicates():
    normalized_pop_group = south_belt_df.loc[south_belt_df['level5'] == pop]['price']  / south_belt_df.loc[south_belt_df['level5'] == pop]['price'].abs().max()
    normalized_values.append(pd.DataFrame(normalized_pop_group).to_numpy())

fig, axes = plt.subplots(nrows=2, ncols=2)
n_bins = 10
ax0, ax1, ax2, ax3 = axes.flat
plot_titles = south_belt_df.level5.drop_duplicates().values

ax0.hist(normalized_values[0], n_bins, histtype='bar')
ax0.set_title(plot_titles[0])

ax1.hist(normalized_values[1], n_bins, histtype='bar')
ax1.set_title(plot_titles[1])

ax2.hist(normalized_values[2], n_bins, histtype='bar')
ax2.set_title(plot_titles[2])

ax3.hist(normalized_values[3], n_bins, histtype='bar')
ax3.set_title(plot_titles[3])

plt.tight_layout()
plt.savefig('plots/MultiHistogram.png')
plt.show()

#Exercise 19
    
pps_mean1 = south_belt_df.loc[south_belt_df['level5'] == 'Getafe']['pps'].mean()
pps_mean2 = south_belt_df.loc[south_belt_df['level5'] == 'Alcorcón']['pps'].mean()

if pps_mean1 > pps_mean2:
    print(f"The price per square meter mean in Getafe({pps_mean1}) is greater than the value in Alcorcón({pps_mean2})")
else:
    print(f"The price per square meter mean in Alcorcón({pps_mean2}) is greater than the value in Getafe({pps_mean1})")
    
#Exercise 20
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
ax0, ax1, ax2, ax3 = axes.flat

ax0.scatter(normalized_values[0], normalized_values[0])
ax0.set_title(plot_titles[0])

ax1.scatter(normalized_values[1], normalized_values[1])
ax1.set_title(plot_titles[1])

ax2.scatter(normalized_values[2], normalized_values[2])
ax2.set_title(plot_titles[2])

ax3.scatter(normalized_values[3], normalized_values[3])
ax3.set_title(plot_titles[3])

plt.tight_layout()
plt.savefig('plots/MultiScatterPlot.png')
plt.show()

#Exercise 21
populations = ['Fuenlabrada', 'Leganés', 'Getafe', 'Alcorcón']
subset = data_set[data_set['level5'].isin(populations)]
y = subset['latitude'].str.replace(',', '.').astype(float).to_list()
x = subset['longitude'].str.replace(',', '.').astype(float).to_list()
population_coord = subset.groupby('level5').apply(lambda x: x[['latitude', 'longitude']].to_dict(orient='records')).to_dict()
colors = {
    'Fuenlabrada': 'red',
    'Leganés': 'blue',
    'Getafe': 'green',
    'Alcorcón': 'purple'
}
madrid_map = Map(center=(40.4168, -3.7038), zoom=12, basemap=basemaps.OpenStreetMap.Mapnik)
for population, coords in population_coord.items():
    for coord in coords:
        marker = CircleMarker(location=(coord['latitude'], coord['longitude']),
                              radius=5,
                              color=colors[population],
                              fill=True,
                              fill_color=colors[population])
        madrid_map.add_layer(marker)
