import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Paso 1: Cargar los datos
# Define el símbolo de la acción que deseas descargar
symbol = "AAPL"  # Por ejemplo, para Apple Inc.

# Define el rango de fechas que deseas descargar
start_date = "2023-01-01"
end_date = "2023-07-31"

# Descarga los datos de precios utilizando yfinance
data = yf.download(symbol, start=start_date, end=end_date)

# Guarda los datos en un archivo CSV
csv_filename = f"{symbol}_prices.csv"
data.to_csv(csv_filename)

print(f"Datos guardados en {csv_filename}")

csv_filename = '/Users/joseacevedo/Desktop/Stock/AAPL_prices.csv'
data = pd.read_csv(csv_filename)  # Asegúrate de tener un archivo CSV con tus datos históricos
# Paso 2: Preparar los datos
# Convertir la columna de fecha a tipo datetime
data['Date'] = pd.to_datetime(data['Date'])

# Convertir la fecha a una característica numérica (puede ser el número de días desde la primera fecha)
data['Days'] = (data['Date'] - data['Date'].min()).dt.days

# Definir X (características) e y (objetivo)
X = data[['Days']]
y =data['Close']  # Variable objetivo

# Paso 3: Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 4: Crear y entrenar el modelo 
model = LinearRegression()
model.fit(X_train, y_train)

# Paso 5: Realizar predicciones, generar una prediccion para el siguiente periodo 
y_pred = model.predict(X_test)

# Paso 6: Evaluar el rendimiento del modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
        
 
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Datos reales')
plt.plot(X_test, y_pred, color='red', label='Predicciones')
plt.xlabel('Días')
plt.ylabel('Precio de cierre')
plt.title('Predicciones de precio de cierre vs Datos reales')
plt.legend()
plt.show()
