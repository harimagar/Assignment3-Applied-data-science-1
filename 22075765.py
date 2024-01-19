import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import cluster_tools as ct
import scipy.optimize as opt
import errors as err

def load_and_clean_data(file_path):
    """
    Reads the data from the given file, cleans it, and returns cleaned DataFrames.

    Parameters
    ----------
    file_path : str
        The file path to be read into a DataFrame.

    Returns
    -------
    cleaned_df : pandas DataFrame
        The cleaned version of the ingested DataFrame.
    transposed_df : pandas DataFrame
        The transposed version of the cleaned DataFrame.

    """
    if ".csv" in file_path:
        data_df = pd.read_csv(file_path, index_col=0)
    else:
        print("Invalid filetype")
        return None, None

    cleaned_df = data_df.dropna(axis=1, how="all").dropna()
    transposed_df = cleaned_df.transpose()

    return cleaned_df, transposed_df

# For reproducibility
np.random.seed(10)

def apply_kmeans_clustering(num_clusters, data):
    kmeans = cluster.KMeans(n_clusters=num_clusters, n_init=10)
    kmeans.fit(data)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    return labels, centers

def quadratic_curve(x, a, b, c):
    x = x - 2003
    f = a + b * x + c * x**2

    return f

# The CSV files are read into DataFrames
_, co2_data = load_and_clean_data("co2emissions.csv")
_, gdp_data = load_and_clean_data("gdppercapita.csv")

# Specific columns are extracted for Canada
co2_canada = co2_data.loc[:, "Canada"].copy()
gdp_per_capita_canada = gdp_data.loc["1990":"2019", "Canada"].copy()

# The extracted columns are merged into a DataFrame
canada_df = pd.merge(co2_canada, gdp_per_capita_canada, on=co2_canada.index, how="outer")
canada_df = canada_df.rename(columns={'key_0': "Year", 'Canada_x': "co2_emissions", 'Canada_y': "gdp_per_capita"})
canada_df = canada_df.set_index("Year")

# The scatter matrix of the DataFrame is plotted
pd.plotting.scatter_matrix(canada_df)

# The DataFrame for clustering is created
cluster_df = canada_df[["co2_emissions", "gdp_per_capita"]].copy()

# The data is normalized
cluster_df, min_values, max_values = ct.scaler(cluster_df)

# The number of clusters and respective silhouette scores are printed
for num_clusters in range(2, 10):
    labels, centroids = apply_kmeans_clustering(num_clusters, cluster_df)
    print(num_clusters, skmet.silhouette_score(cluster_df, labels))

# The cluster centers and labels are calculated using the function
cluster_labels, cluster_centers = apply_kmeans_clustering(5, cluster_df)
xcen = cluster_centers[:, 0]
ycen = cluster_centers[:, 1]

# The clustering is plotted
plt.figure()
color_map = plt.cm.get_cmap('viridis')  # Updated colormap to 'viridis'
plt.scatter(cluster_df['gdp_per_capita'], cluster_df["co2_emissions"], s=10,
            c=cluster_labels, marker='o', cmap=color_map)
plt.scatter(xcen, ycen, s=20, c="k", marker="d")
plt.title("CO2 emission vs GDP per capita of Canada", fontsize=20)
plt.xlabel("GDP per capita", fontsize=18)
plt.ylabel("CO2 emissions", fontsize=18)
plt.show()

# The cluster centers are rescaled to the original scale
rescaled_centers = ct.backscale(cluster_centers, min_values, max_values)
xcen = rescaled_centers[:, 0]
ycen = rescaled_centers[:, 1]

# The clustering is plotted with the original scale
plt.figure()
color_map = plt.cm.get_cmap('viridis')  # Updated colormap to 'viridis'
plt.scatter(canada_df['gdp_per_capita'], canada_df["co2_emissions"], 10,
            cluster_labels, marker='o', cmap=color_map)
plt.xlabel("GDP per capita")
plt.ylabel("CO2 emissions")
plt.title("CO2 emission vs GDP per capita of Canada")
plt.show()

# The DataFrame is prepared for fitting
canada_df = canada_df.reset_index()
canada_df["gdp_per_capita"] = pd.to_numeric(canada_df["gdp_per_capita"])
canada_df["Year"] = pd.to_numeric(canada_df["Year"])

# The fitting of the GDP per capita plot
# Calculates the parameters and covariance
params, covariance = opt.curve_fit(quadratic_curve, canada_df["Year"],
                                   canada_df["gdp_per_capita"])
# Calculates the standard deviation
sigma_values = np.sqrt(np.diag(covariance))
forecast_years = np.arange(1990, 2030)
# Calculates the fitting curve
gdp_forecast = quadratic_curve(forecast_years, *params)
# Calculates the confidence range
lower_bound, upper_bound = err.err_ranges(forecast_years, quadratic_curve, params, sigma_values)
canada_df["fit1"] = quadratic_curve(canada_df["Year"], *params)

# The fitting of CO2 Emissions plot
# Calculates the parameters and covariance
params, covariance = opt.curve_fit(quadratic_curve, canada_df["Year"], canada_df["co2_emissions"])
# Calculates the standard deviation
sigma_values = np.sqrt(np.diag(covariance))
# Calculates the fitting curve
co2_forecast = quadratic_curve(forecast_years, *params)
# Calculates the confidence range
lower_bound, upper_bound = err.err_ranges(forecast_years, quadratic_curve, params, sigma_values)
canada_df["fit2"] = quadratic_curve(canada_df["Year"], *params)

# Line plot for GDP per capita
plt.figure(figsize=(10, 6))
plt.plot(canada_df["Year"], canada_df["gdp_per_capita"], label="GDP", color='blue')
plt.plot(forecast_years, gdp_forecast, label="forecast", color='purple')  # Changed color to 'purple'
plt.fill_between(forecast_years, lower_bound, upper_bound, color="skyblue", alpha=0.8)
plt.xlabel("Year", fontsize=16)
plt.ylabel("GDP per capita", fontsize=14, color='blue')
plt.title("GDP per capita forecast of Canada", fontsize=18)
plt.legend()
plt.show()

# Line plot for CO2 Emissions
plt.figure(figsize=(10, 6))
plt.plot(canada_df["Year"], canada_df["co2_emissions"], label="CO2 emissions", color='purple')  # Changed color to 'orange'
plt.plot(forecast_years, co2_forecast, label="forecast", color='green')
plt.fill_between(forecast_years, lower_bound, upper_bound, color="orange", alpha=0.8)
plt.xlabel("Year", fontsize=16)
plt.ylabel("CO2 Emissions (metric tons per capita)", fontsize=12)
plt.title("CO2 Emissions forecast of Canada", fontsize=18)
plt.legend()
plt.show()


# The clustering is plotted
plt.figure()
color_map = plt.cm.get_cmap('viridis')  # Updated colormap to 'viridis'
plt.scatter(cluster_df['gdp_per_capita'], cluster_df["co2_emissions"], s=10,
            c=cluster_labels, marker='o', cmap=color_map)
plt.scatter(xcen, ycen, s=20, c="k", marker="d")
plt.title("CO2 emission vs GDP per capita of Canada", fontsize=20)
plt.xlabel("GDP per capita", fontsize=18)
plt.ylabel("CO2 emissions", fontsize=18)
plt.show()

# The clustering is plotted with the original scale
plt.figure()
color_map = plt.cm.get_cmap('viridis')  # Updated colormap to 'viridis'
plt.scatter(canada_df['gdp_per_capita'], canada_df["co2_emissions"], 10,
            cluster_labels, marker='o', cmap=color_map)
plt.xlabel("GDP per capita")
plt.ylabel("CO2 emissions")
plt.title("CO2 emission vs GDP per capita of Canada")
plt.show()

# Line plot for CO2 Emissions forecast
plt.figure(figsize=(10, 6))
plt.plot(canada_df["Year"], canada_df["co2_emissions"], label="CO2 emissions", color='purple')
plt.plot(forecast_years, co2_forecast, label="forecast", color='green')
plt.fill_between(forecast_years, lower_bound, upper_bound, color="orange", alpha=0.8)
plt.xlabel("Year", fontsize=16)
plt.ylabel("CO2 Emissions (metric tons per capita)", fontsize=12)
plt.title("CO2 Emissions forecast of Canada", fontsize=18)
plt.legend()
plt.show()

# Line plot for GDP per capita forecast
gdp_forecast = quadratic_curve(forecast_years, *params)

plt.figure(figsize=(10, 6))
plt.plot(canada_df["Year"], canada_df["gdp_per_capita"], label="GDP per capita", color='purple')
plt.plot(forecast_years, gdp_forecast, label="forecast", color='orange')
plt.fill_between(forecast_years, lower_bound, upper_bound, color="skyblue", alpha=0.8)
plt.xlabel("Year", fontsize=16)
plt.ylabel("GDP per capita", fontsize=14, color='purple')
plt.title("GDP per capita forecast of Canada", fontsize=18)
plt.legend()
plt.show()