import geopandas as gpd
import folium
import pandas as pd
from shapely.geometry import Point
from branca.colormap import LinearColormap
import matplotlib.pyplot as plt

# Leer el archivo CSV con los datos de precios de apartamentos
df = pd.read_csv("final_clean.csv")
df.shape

df["price"].min()
df["price"].max()
df["price_millions"] = df["price"] / 1e6

# Exclude points above cut_off (in millions)
cut_off = df["price_millions"].quantile(0.99)  # 99th percentile
df_filtered = df[df["price_millions"] <= cut_off]

df_filtered["price"].min()
df_filtered["price"].max()
# Plot histogram
plt.hist(df_filtered["price_millions"], bins=30, color="blue", edgecolor="black")
plt.title("Price Distribution")
plt.xlabel("Price (in millions of pesos)")
plt.ylabel("Frequency")
plt.show()

df.sort_values(by="price", ascending=False).head(50)

# Calcular los límites de la escala de colores
precio_min = 250000
# precio_max = 25000000
precio_max = 10000000
# Crear una nueva columna en el DataFrame para clasificar los precios en una escala
df["price_range"] = pd.cut(df["price"], bins=5)

# Crear una lista de colores para la escala de precios
colormap = LinearColormap(
    ["green", "yellow", "orange", "red", "darkred"], vmin=precio_min, vmax=precio_max
)

# Crear un mapa con el fondo de relieve
m = folium.Map(location=[4.6097, -74.0817], zoom_start=12, tiles="Stamen Terrain")

# Cambiar el color y grosor del borde del mapa
folium.TileLayer("Stamen Terrain", overlay=True, control=False, attr="black").add_to(m)

# Leer el shapefile de las UPZs de Bogotá
shp = gpd.read_file("/home/brayan/Desktop/python/Web_Scrapping/EM2021_UPZ.shp")

# Crear un GeoDataFrame con los datos y las geometrías
geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
gdf = gpd.GeoDataFrame(df, geometry=geometry)

# Agregar las UPZs al mapa con el grosor de línea más delgado
folium.GeoJson(shp, style_function=lambda x: {"color": "black", "weight": 0.5}).add_to(
    m
)

# Agregar los puntos de los apartamentos al mapa con colores según la escala de precios
for i, row in gdf.iterrows():
    color = colormap(row["price"])
    folium.Circle(
        location=[row["latitude"], row["longitude"]],
        radius=20,
        color=color,
        fill=True,
        fill_color=color,
    ).add_to(m)

# Agregar la leyenda de la escala de precios al mapa
colormap.caption = "Precios de Apartamentos en Bogotá"
colormap.add_to(m)


m.add_child(colormap)
# Guardar el mapa en un archivo HTML
m.save("mapa_precios_apartamentos_p10.html")
