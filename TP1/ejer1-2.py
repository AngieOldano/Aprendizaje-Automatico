import pandas as pd

df = pd.read_csv('prestamo.csv', encoding='latin-1')

# Filtrar por 50 años
df_50 = df[df['Edad'] == 50].copy()

columnas = ['Sexo', 'Mayor nivel educativo', 'Estado de vivienda', 'Préstamos previos impagos']
estado = 'Estado'

# Dividir en Entrenamiento (75%) y Prueba (25%)
limite  = int(len(df_50) * 0.75) 
entrenamiento = df_50.iloc[:limite] 
prueba = df_50.iloc[limite:]  

# Algoritmo FIND-S
def find_s(data, atributos, concepto):
    h = ['0'] * len(atributos)     # Hipótesis inicial
    positivos = data[data[concepto] == 'OTORGADO']  
    for j, fila in positivos.iterrows(): 
        for i, col in enumerate(atributos):
            valor_actual = str(fila[col])
            if h[i] == '0':
                h[i] = valor_actual
            elif h[i] != valor_actual:
                h[i] = '?'
    return h

hipotesis = find_s(entrenamiento, columnas, estado)
print(f"Hipótesis obtenida: {hipotesis}")


# Algoritmo de predicción
def predecir(hipotesis, registro, atributos):
    for i, valor in enumerate(hipotesis):
        if valor != '?' and valor != registro[atributos[i]]:
            return 'RECHAZADO'
    return 'OTORGADO'


# Predicción en el conjunto de prueba
aciertos = 0
no_aciertos = 0
for i, registro in prueba.iterrows():
    prediccion = predecir(hipotesis, registro, columnas) 
    es_correcto = prediccion == registro[estado] 
    if es_correcto:
        aciertos += 1
    else:
        no_aciertos += 1


    print(f"Real: {registro[estado]} | Predicción: {prediccion} | {'OK' if es_correcto else 'ERROR'}")
print(aciertos)
print(no_aciertos)
