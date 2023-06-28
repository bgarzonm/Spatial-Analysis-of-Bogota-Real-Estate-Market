# Data manipulation and analysis libraries
import numpy as np
import pandas as pd


# Spatial analysis libraries
import mapclassify
from numba import jit
from scipy.linalg import inv
from pysal.lib import weights
from pysal.model import spreg
from pysal.explore import esda
from libpysal.weights import Queen, KNN

# Geospatial and geographic data visualization libraries
import geopandas as gpd
from shapely import geometry
from geopandas.tools import sjoin
from shapely.geometry import Point, Polygon

# Data visualization libraries
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable


import re
import contextily
import statsmodels.formula.api as sm


# Load your CSV data
# df = pd.read_csv("../data/final_clean.csv")
df = pd.read_csv("../data/final_distances.csv")

del df["Unnamed: 0"]
df.info()

df["parking"] = df["parking"].replace("Más de 10", 10)
df["parking"] = pd.to_numeric(df["parking"])
df["parking"] = df["parking"].fillna(0, inplace=False)
df["l_price"] = np.log(df["price"])

# del df["price"]

neighborhoods = gpd.read_file(
    "/home/brayan/Desktop/python/Web_Scrapping/EM2021_UPZ.shp"
)

# # Create a figure with a larger size
# fig = plt.figure(figsize=(10, 8))

# # Plot the shapefile
# neighborhoods.plot()

# # Add a title
# plt.title("Shapefile of EM2021_UPZ")

# # Display the plot
# plt.show()

lat_min = 4.46
lat_max = 4.8
lon_min = -74.25
lon_max = -74

df_bogota = df[
    (df["latitude"] >= lat_min)
    & (df["latitude"] <= lat_max)
    & (df["longitude"] >= lon_min)
    & (df["longitude"] <= lon_max)
]

xys = df_bogota[["longitude", "latitude"]].apply(lambda row: Point(*row), axis=1)

gdb = gpd.GeoDataFrame(df_bogota.assign(geometry=xys), crs="+init=epsg:4326")

gdb.head()

del gdb["longitude"]
del gdb["latitude"]

gdb.to_file("df.geojson", driver="GeoJSON")

db = gpd.read_file("../data/df.geojson")


db.columns

variables = [
    "bedrooms",
    "bathrooms",
    "parking",
    "built_area",
    "private_area",
    "stratum",
    "age",
]

# Fit OLS model
m1 = spreg.OLS(
    # Dependent variable
    db[["l_price"]].values,
    # Independent variables
    db[variables].values,
    # Dependent variable name
    name_y="l_price",
    # Independent variable name
    name_x=variables,
)


print(m1.summary)


knn = weights.KNN.from_dataframe(db, k=1)

lag_residual = weights.spatial_lag.lag_spatial(knn, m1.u)
ax = sns.regplot(
    x=m1.u.flatten(),
    y=lag_residual.flatten(),
    line_kws=dict(color="orangered"),
    ci=None,
)
ax.set_xlabel("Model Residuals - $u$")
ax.set_ylabel("Spatial Lag of Model Residuals - $W u$")


# Re-weight W to 20 nearest neighbors
knn.reweight(k=20, inplace=True)
# Row standardize weights
knn.transform = "R"
# Run LISA on residuals
outliers = esda.moran.Moran_Local(m1.u, knn, permutations=9999)

# Select only LISA cluster cores
error_clusters = outliers.q % 2 == 1
# Filter out non-significant clusters
error_clusters &= outliers.p_sim <= 0.001
# Add `error_clusters` and `local_I` columns
ax = (
    db.assign(
        error_clusters=error_clusters,
        local_I=outliers.Is
        # Retain error clusters only
    )
    .query(
        "error_clusters"
        # Sort by I value to largest plot on top
    )
    .sort_values(
        "local_I"
        # Plot I values
    )
    .plot("local_I", cmap="bwr", marker=".")
)

# Add basemap
contextily.add_basemap(ax, crs=db.crs)
# Remove axes
ax.set_axis_off()

ax = db.plot("l_price", marker=".", s=5)
contextily.add_basemap(ax, crs=db.crs)
ax.set_axis_off()


# --------------------- Spatial Fixed Effects --------------------- #


joined = gpd.sjoin(db, neighborhoods, how="inner", op="intersects")

joined

f = "l_price ~ " + " + ".join(variables) + " + " "NombreUP_1 - 1"

print(f)


# Second OLS model

m2 = sm.ols(f, data=joined).fit()

# Store variable names for all the spatial fixed effects
sfe_names = [i for i in m2.params.index if "NombreUP_1[" in i]
# Create table
pd.DataFrame(
    {
        "Coef.": m2.params[sfe_names],
        "Std. Error": m2.bse[sfe_names],
        "P-Value": m2.pvalues[sfe_names],
    }
)

print(m2.summary())


# spreg spatial fixed effect implementation

m3 = spreg.OLS_Regimes(
    # Dependent variable
    joined[["l_price"]].values,
    # Independent variables
    joined[variables].values,
    # Variable specifying neighborhood membership
    joined["NombreUP_1"].tolist(),
    # Allow the constant term to vary by group/regime
    constant_regi="many",
    # Variables to be allowed to vary (True) or kept
    # constant (False). Here we set all to False
    cols2regi=[False] * len(variables),
    # Allow separate sigma coefficients to be estimated
    # by regime (False so a single sigma)
    regime_err_sep=False,
    # Dependent variable name
    name_y="l_price",
    # Independent variables names
    name_x=variables,
)

print(m3.summary)

np.round(m3.betas.flatten() - m2.params.values, decimals=12)

neighborhood_effects = m2.params.filter(like="NombreUP_1")
neighborhood_effects.head(100)


# Create a sequence with the variable names without
# `neighborhood[` and `]`
stripped = neighborhood_effects.index.str.replace(r"NombreUP_1\[|\]", "", regex=True)
# Reindex the neighborhood_effects Series on clean names
neighborhood_effects.index = stripped
# Convert Series to DataFrame
neighborhood_effects = neighborhood_effects.to_frame("fixed_effect")
# Print top of table
neighborhood_effects.head()

neighborhood_effects = neighborhood_effects.reset_index().rename(
    columns={"index": "NombreUP_1"}
)

# Merge neighborhoods and neighborhood_effects

merged_df = neighborhoods.merge(neighborhood_effects, how="left", on="NombreUP_1")

merged_df[merged_df.isna().any(axis=1)]


# Plot neighborhoods
ax = neighborhoods.plot(color="k", linewidth=0, alpha=0.5, figsize=(12, 6))

# Plot merged_df on the same axis (ax)
merged_df.plot(
    "fixed_effect",  # Variable to display
    scheme="quantiles",  # Choropleth scheme
    k=10,  # No. of classes in the choropleth
    linewidth=0.1,  # Polygon border width
    cmap="YlGnBu",  # Color scheme
    ax=ax,  # Axis to draw on (Same ax object)
)

# Add basemap
contextily.add_basemap(
    ax,
    crs=neighborhoods.crs,
    source=contextily.providers.CartoDB.PositronNoLabels,
)

# Remove axis
ax.set_axis_off()

# Display
plt.show()


# With the bar of fixed effects on the right

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
merged_df = merged_df.sort_values(by="fixed_effect", ascending=False)

# Plot merged_df on the same axis (ax)
merged_df.plot(
    "fixed_effect",  # Variable to display
    scheme="quantiles",  # Choropleth scheme
    k=10,  # No. of classes in the choropleth
    linewidth=0.1,  # Polygon border width
    cmap="YlGnBu",  # Color scheme
    legend=False,  # Add a legend
    ax=ax,  # Axis to draw on (Same ax object)
)

contextily.add_basemap(
    ax,
    crs=neighborhoods.crs,
    source=contextily.providers.CartoDB.PositronNoLabels,
)

# Create fake colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)

norm = mcolors.Normalize(
    vmin=merged_df["fixed_effect"].min(), vmax=merged_df["fixed_effect"].max()
)
cmap = plt.get_cmap("YlGnBu")
cbar = fig.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="vertical"
)
cbar.set_label("Fixed Effect Quantiles")

# Remove axis
ax.set_axis_off()

plt.show()


# Fit spatial lag model with `spreg`
# (GMM estimation)

knn = weights.KNN.from_dataframe(joined, k=1)
knn.reweight(k=20, inplace=True)
# Row standardize weights
knn.transform = "R"

m4 = spreg.GM_Lag(
    # Dependent variable
    joined[["l_price"]].values,
    # Independent variables
    joined[variables].values,
    # Spatial weights matrix
    w=knn,
    # Dependent variable name
    name_y="l_price",
    # Independent variables names
    name_x=variables,
)

print(m4.summary)
# Build full table of regression coefficients
pd.DataFrame(
    {
        # Pull out regression coefficients and
        # flatten as they are returned as Nx1 array
        "Coeff.": m4.betas.flatten(),
        # Pull out and flatten standard errors
        "Std. Error": m4.std_err.flatten(),
        # Pull out P-values from t-stat object
        "P-Value": [i[1] for i in m4.z_stat],
    },
    index=m4.name_z
    # Round to four decimals
).round(4)


rho = m4.rho
# Extracting the coefficients
beta_hat = m4.betas[1:]
W_matrix = knn.full()[0]  # Full weight matrix from PySAL weights object
I = np.eye(W_matrix.shape[0])
A = inv(I - rho * W_matrix)


# Create the design matrix
X = np.column_stack((np.ones((joined.shape[0], 1)), joined[variables].values))

# Compute the predicted values
y_hat_pre = A.dot(X.dot(beta_hat))


# Copy the dataframe


col_new = joined.copy()

col_new["NombreUP_1"].unique()

col_new.loc[col_new["NombreUP_1"] == "NIZA", "bedrooms"] += 1

# New design matrix
X_d = np.column_stack((np.ones((col_new.shape[0], 1)), col_new[variables].values))

# Recompute predicted values
y_hat_post = A.dot(X_d.dot(beta_hat))

# Calculate change in predicted values
delta_y = y_hat_post - y_hat_pre
col_new["delta_y"] = delta_y

# Print summary statistics and sum
print(col_new["delta_y"].describe())
print(col_new["delta_y"].sum())


classifier = mapclassify.NaturalBreaks.make(k=100)
col_new["delta_y_class"] = col_new[["delta_y"]].apply(classifier)

# Crear el mapa
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
col_new.plot(
    column="delta_y_class", cmap="Spectral", linewidth=0.8, ax=ax, edgecolor="0.8"
)

# Crear la leyenda manualmente
norm = plt.Normalize(
    vmin=col_new["delta_y_class"].min(), vmax=col_new["delta_y_class"].max()
)
cbar = plt.cm.ScalarMappable(norm=norm, cmap="Spectral")
fig.colorbar(cbar, ax=ax)

plt.show()


# Filter the data within the desired latitude and longitude range
filtered_data = col_new.cx[-74.11:-74.050, 4.67:4.76]

# Create the classifier
classifier = mapclassify.NaturalBreaks.make(k=100)

# Apply the classifier to the filtered data
filtered_data["delta_y_class"] = classifier(filtered_data["delta_y"])

# Create the map
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
filtered_data.plot(
    column="delta_y_class", cmap="Spectral", linewidth=0.8, ax=ax, edgecolor="0.8"
)

# Create the legend manually
norm = plt.Normalize(
    vmin=filtered_data["delta_y_class"].min(), vmax=filtered_data["delta_y_class"].max()
)
cbar = plt.cm.ScalarMappable(norm=norm, cmap="Spectral")
fig.colorbar(cbar, ax=ax)

plt.show()


# Now model with distances to the nearest Transmi station

# First a normal OLS model

variables2 = ["estacion_cercana", "distancia_minima"]


# Fit OLS model
m5 = spreg.OLS(
    # Dependent variable
    db[["l_price"]].values,
    # Independent variables
    db[variables2].values,
    # Dependent variable name
    name_y="l_price",
    # Independent variable name
    name_x=variables2,
)


print(m5.summary)

# complete variables

variables3 = [
    "bedrooms",
    "bathrooms",
    "parking",
    "built_area",
    "private_area",
    "stratum",
    "age",
    "distancia_minima",
    "estacion_cercana",
]

# Now with spatial lag and weights

knn = weights.KNN.from_dataframe(joined, k=1)
knn.reweight(k=20, inplace=True)
# Row standardize weights
knn.transform = "R"

# Fit spatial lag model with `spreg`
# (GMM estimation)
m6 = spreg.GM_Lag(
    # Dependent variable
    joined[["l_price"]].values,
    # Independent variables
    joined[variables3].values,
    # Spatial weights matrix
    w=knn,
    # Dependent variable name
    name_y="l_price",
    # Independent variables names
    name_x=variables3,
)


print(m6.summary)

rho = m6.rho
# Extracting the coefficients
beta_hat = m6.betas[1:]
W_matrix = knn.full()[0]  # Full weight matrix from PySAL weights object
I = np.eye(W_matrix.shape[0])
A = inv(I - rho * W_matrix)


# Create the design matrix
X = np.column_stack((np.ones((joined.shape[0], 1)), joined[variables3].values))

# Compute the predicted values
y_hat_pre = A.dot(X.dot(beta_hat))


# Copy the dataframe


col_new = joined.copy()

col_new["NombreUP_1"].unique()

x = col_new.loc[
    (col_new["NombreUP_1"] == "CHAPINERO: Chico Lago + El Refugio")
    & (col_new["price"] > 5000000),
    "distancia_minima",
]


a = x * -0.6
col_new.loc[
    (col_new["NombreUP_1"] == "CHAPINERO: Chico Lago + El Refugio")
    & (col_new["price"] > 5000000),
    "distancia_minima",
] += a


# New design matrix
X_d = np.column_stack((np.ones((col_new.shape[0], 1)), col_new[variables3].values))


# Recompute predicted values
y_hat_post = A.dot(X_d.dot(beta_hat))

# Calculate change in predicted values
delta_y = y_hat_post - y_hat_pre
col_new["delta_y"] = delta_y

# Print summary statistics and sum
print(col_new["delta_y"].describe())
print(col_new["delta_y"].sum())


classifier = mapclassify.NaturalBreaks.make(k=100)
col_new["delta_y_class"] = col_new[["delta_y"]].apply(classifier)

# Crear el mapa
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
col_new.plot(
    column="delta_y_class", cmap="Spectral", linewidth=0.8, ax=ax, edgecolor="0.8"
)

# Crear la leyenda manualmente
norm = plt.Normalize(
    vmin=col_new["delta_y_class"].min(), vmax=col_new["delta_y_class"].max()
)
cbar = plt.cm.ScalarMappable(norm=norm, cmap="Spectral")
fig.colorbar(cbar, ax=ax)

plt.show()


# Filter the data within the desired latitude and longitude range
filtered_data = col_new.cx[-73.9:-74.10, 4.64:4.76]

# Create the classifier
classifier = mapclassify.NaturalBreaks.make(k=100)

# Apply the classifier to the filtered data
filtered_data["delta_y_class"] = classifier(filtered_data["delta_y"])

# Create the map
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
filtered_data.plot(
    column="delta_y_class", cmap="Spectral", linewidth=0.8, ax=ax, edgecolor="0.8"
)

# Create the legend manually
norm = plt.Normalize(
    vmin=filtered_data["delta_y_class"].min(), vmax=filtered_data["delta_y_class"].max()
)
cbar = plt.cm.ScalarMappable(norm=norm, cmap="Spectral")
fig.colorbar(cbar, ax=ax)

plt.show()
