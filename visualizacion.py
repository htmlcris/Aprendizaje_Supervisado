import matplotlib.pyplot as plt
from data_loader import load_bike_data


# -----------------------------
# 1. CARGAR DATOS
# -----------------------------
bikes = load_bike_data()


# -----------------------------
# 2. CREAR FIGURA
# -----------------------------
fig, axes = plt.subplots(3, 2, figsize=(14, 14))


# ==========================================
# GRAFICO 1: temp vs cnt
# ==========================================
axes[0, 0].scatter(
    bikes["temp"],
    bikes["cnt"]
)

axes[0, 0].set_title("Relacion entre Temperatura y Alquileres")
axes[0, 0].set_xlabel("Temperatura (temp)")
axes[0, 0].set_ylabel("Cantidad de alquileres (cnt)")


# ==========================================
# GRAFICO 2: atemp vs cnt
# ==========================================
axes[0, 1].scatter(
    bikes["atemp"],
    bikes["cnt"]
)

axes[0, 1].set_title("Relacion entre Sensacion termica y Alquileres")
axes[0, 1].set_xlabel("Sensacion termica (atemp)")
axes[0, 1].set_ylabel("Cantidad de alquileres (cnt)")


# ==========================================
# GRAFICO 3: yr vs promedio de cnt
# ==========================================
yr_mean = bikes.groupby("yr")["cnt"].mean()

axes[1, 0].bar(
    yr_mean.index.astype(str),
    yr_mean.values
)

axes[1, 0].set_title("Promedio de alquileres por año")
axes[1, 0].set_xlabel("Año (0 = 2011, 1 = 2012)")
axes[1, 0].set_ylabel("Promedio de cnt")


# ==========================================
# GRAFICO 4: season vs promedio de cnt
# ==========================================
season_mean = bikes.groupby("season")["cnt"].mean()

axes[1, 1].bar(
    season_mean.index.astype(str),
    season_mean.values
)

axes[1, 1].set_title("Promedio de alquileres por estacion")
axes[1, 1].set_xlabel("Estacion (1=Primavera, 2=Verano, 3=Otoño, 4=Invierno)")
axes[1, 1].set_ylabel("Promedio de cnt")


# ==========================================
# GRAFICO 5: mnth vs promedio de cnt
# ==========================================
mnth_mean = bikes.groupby("mnth")["cnt"].mean()

axes[2, 0].plot(
    mnth_mean.index,
    mnth_mean.values,
    marker="o"
)

axes[2, 0].set_title("Promedio de alquileres por mes")
axes[2, 0].set_xlabel("Mes")
axes[2, 0].set_ylabel("Promedio de cnt")


# ==========================================
# GRAFICO 6: weathersit vs promedio de cnt
# ==========================================
weather_mean = bikes.groupby("weathersit")["cnt"].mean()

axes[2, 1].bar(
    weather_mean.index.astype(str),
    weather_mean.values
)

axes[2, 1].set_title("Promedio de alquileres por clima")
axes[2, 1].set_xlabel("Clima (weathersit)")
axes[2, 1].set_ylabel("Promedio de cnt")


# Ajustar espacios
plt.tight_layout()

# Mostrar
plt.show()