import pandas as pd

df = pd.read_csv('prestamo.csv', encoding='latin-1') #pd.read_csv() es la función de la libreria pandas que se usa para leer un archivo CSV y convertirlo en un DataFrame (una tabla manipulable en Python)

# Filtrar por 50 años
df_50 = df[df['Edad'] == 50].copy() # Hago una copia del dataframe(tabla) filtrado para no modificar el df original

columnas = ['Sexo', 'Mayor nivel educativo', 'Estado de vivienda', 'Préstamos previos impagos'] #columnas a tener en cuenta para el algoritmo
estado = 'Estado' #estado del prestamo, columna que representa el concepto
#concepto a predecir, es decir, el resultado que queremos obtener con el algoritmo FIND-S (si el préstamo fue otorgado o rechazado)

# Dividir en Entrenamiento (75%) y Prueba (25%)
limite  = int(len(df_50) * 0.75) #de la cantidad de retistros de personas de 50 años saco el %75(son 42 registros)
#iloc de Panda es una propiedad utilizada para seleccionar filas y columnas de un DataFrame basándose en su posición entera (índice numérico), comenzando desde 0.
entrenamiento = df_50.iloc[:limite] # tomo los primeros 42 registros para entrenamiento(es el 75% de los registros de personas de 50 años)
prueba = df_50.iloc[limite:]  #desde el 42 hasta el final para prueba (es el 25% de los registros de personas de 50 años)

# Algoritmo FIND-S
def find_s(data, atributos, concepto): #recibe el dataframe, las columnas(atributos) a tener en cuenta(que nos pide el ejer) y el estado (del prestamo (concepto a predecir))
    h = ['0'] * len(atributos) #representa la hipótesis inicial, que es la más específica posible (todos los atributos son '0') 
    positivos = data[data[concepto] == 'OTORGADO']  # guarda solo los registros donde el estado del préstamo es 'OTORGADO', es decir, los ejemplos positivos que se utilizarán para generalizar la hipótesis
    # data[concepto] devuelve una serie con los valores de la columna 'Estado', y data[data[concepto] == 'OTORGADO'] devuelve un DataFrame que contiene solo las filas donde el valor de 'Estado' es 'OTORGADO'
    for j, fila in positivos.iterrows(): #.iterrows() → de la biblioteca Pandas, crea una tupla con un índice por fila y el contenido de la fila, lo que permite iterar sobre cada fila del DataFrame
        for i, col in enumerate(atributos): #el enumerate() enumera la posicion que tiene cada col en el array de columnas que vamos a pasarle
            valor_actual = str(fila[col]) #valor_actual es cada valor de la fila (por ej: 1ra vuelta: FEMENINO, 2da vuelta: UNIVERSITARIO, etc) 
            if h[i] == '0': #si en la posicion i de la hipotesis inicial es 0(porque todavia no se ha generalizado nada)
                h[i] = valor_actual #entonces se asigna el valor actual a la hipotesis
            elif h[i] != valor_actual: #si en la posicion i de la hipotesis ya hay un valor diferente al valor actual, 
                h[i] = '?' #entonces se generaliza esa posición a '?'
    return h

'''
EXPLIACION DEL ALGORITMO FIND-S:
1. Se inicializa la hipótesis h con el valor '0' (La cantidad de 0s depende de cuantas columnas tengamos, en este caso 4)
2. Nos quedamos nomas con los registros donde se les otora el préstamo (ejemplos positivos)
3. En la primera vuelta vamos a obtener la primera fila de los ejemplos positivos(ya que la condición if h[i] == '0': se cumple siempre) por ejemplo: ['FEMENINO', 'UNIVERSITARIO', 'PROPIO', 'NO']
4. En la segunda vuelta comparamos cada elemento de lista/array con la siguiente linea del regstro. Por ej:
si nuestra primera fila era ['FEMENINO', 'UNIVERSITARIO', 'PROPIO', 'NO'] y el registro que sigue es ['MASCULINO', 'SECUNDARIO', 'ALQUILA', 'NO']
    i[0]: ¿'FEMENINO' es igual a 'MASCULINO'?      No. Entonces h[0] = '?'.
    i[1]: ¿'SECUNDARIO' es igual a 'SECUNDARIO'?   Sí. No hace nada.
    i[2]: ¿'PROPIETARIO' es igual a 'ALQUILA'?     No. Entonces h[2] = '?'.
    i[3]: ¿'NO' es igual a 'NO'?                   Sí. No hace nada.
5. Al salir del for, tenemos la hipotesis (['?', '?', '?', 'NO'] en este caso)
'''


hipotesis = find_s(entrenamiento, columnas, estado)
print(f"Hipótesis obtenida: {hipotesis}")

# Algoritmo de predicción
def predecir(hipotesis, registro, atributos): # si es no(no tiene prestamos impagos) devuelve otorgado, si es sí(tiene prestamos impagos) devuelve rechazado
    for i, valor in enumerate(hipotesis):#enumerate() enumera la posicion de cada elemento de l hipotesis que obtuvimos antes (Hipótesis obtenida: ['?', '?', '?', 'NO']) 
        if valor != '?' and valor != registro[atributos[i]]:# Si en la posicion i la hipótesis obtenida no es '?' y no coincide el valor de la hipotesis obtenida con el valor del registro(el de prueba) en esa posición, el estado del préstamo se predice como 'RECHAZADO'
            return 'RECHAZADO'
    return 'OTORGADO' #si hay coincidencia en todas las posiciones, el estado del préstamo se predice como 'OTORGADO'



'''
para la prediccion le pasamos la hipotesis que obtuvimos ['?', '?', '?', 'NO'] y el conjunto de prueba, como son todos ? menos el ultimo va directamente a comparar ese,
osea compara el NO con los datos de la columna de prestamos impagos y busca el distinto a NO,
cuando lo encuentra devuelve rechazado porque predice que todos los que SI tengan prestamos impagos se les va a rechazar el la solicitud de otro prestamo
y por lo tanto supone que los que NO tienen prestamos impagos se les va a otorgar un nuevo prestamo
'''

# Predicción en el conjunto de prueba
aciertos = 0
no_aciertos = 0
for i, registro in prueba.iterrows(): #iterrows pa crear una tupla con un índice por fila y el contenido de la fila
    prediccion = predecir(hipotesis, registro, columnas) # pedicion es va a ser igual a otorgado o rechazado
    es_correcto = prediccion == registro[estado] #comparo la prediccion con el valor real del estado en nuesta tabla de prueba, si es correcto devuelve True, sino False
    if es_correcto: #y esto es un contador de aciertor y no aciertos de la comparacion de la anterior
        aciertos += 1
    else:
        no_aciertos += 1    
    
    print(f"Real: {registro[estado]} | Predicción: {prediccion} | {'OK' if es_correcto else 'ERROR'}")
print("Cantidad de aciertos: ", aciertos)
print("Cantidad de no aciertos: ", no_aciertos)

'''
usando el conjunto de prueba
aca se fija cuantas veces se falla y acierta en la preccion comparando la prediccion (OTORGADO O RECHAZADO) con el valor real del de la solicitud del prestamo
osea para los no acierto:
 1.cuanta las veces que NO tiene prestamos impagos e igualmente se le RECHAZA la solicitud de un nuevo prestamo
 2.y la veces que SI tiene prestamos impagos y se le OTORGA la solicitud de un nuevo prestamo 
para los aciertos:(como lo predecimos)
 1.cuanta las veces que NO tiene prestamos impagos y se le OTORGA
 2.y la veces que SI tiene prestamos impagos y se le RECHAZA
'''



'''
RESUMEN TOTAL:
Usamos la libreria panda para podes usar funciones que faciliten la manipulacion de tablas
1. Leemos el archivo CSV y lo convertimos en un DataFrame(que es una tabla manipulable en Python)
2. Filtramos el dataframe para quedarnos solo con los registros de personas de 50 años
3. Definimos las columnas que vamos a usar como atributos para el algoritmo y la columna que representa el concepto a predecir (estado de la solicitud del préstamo)
4. Dividimos el dataframe en un conjunto de entrenamiento (75% de los registros totales) y un conjunto de prueba (25%)

EXPLIACION DEL ALGORITMO FIND-S:
1. Se inicializa la hipótesis h con el valor '0' (La cantidad de 0s depende de cuantas columnas tengamos, en este caso 4)
2. Nos quedamos nomas con los registros donde se les otora el préstamo (ejemplos positivos)
3. En la primera vuelta vamos a obtener la primera fila de los ejemplos positivos(ya que la condición if h[i] == '0': se cumple siempre) por ejemplo: ['FEMENINO', 'UNIVERSITARIO', 'PROPIO', 'NO']
4. En la segunda vuelta comparamos cada elemento de lista/array con la siguiente linea del regstro. Por ej:
si nuestra primera fila era ['FEMENINO', 'UNIVERSITARIO', 'PROPIO', 'NO'] y el registro que sigue es ['MASCULINO', 'SECUNDARIO', 'ALQUILA', 'NO']
    i[0]: ¿'FEMENINO' es igual a 'MASCULINO'?      No. Entonces h[0] = '?'. (Consejo: Solo se mantiene el valor si todas las filas positivas coinciden en ese atributo.)
    i[1]: ¿'SECUNDARIO' es igual a 'SECUNDARIO'?   Sí. No hace nada.
    i[2]: ¿'PROPIETARIO' es igual a 'ALQUILA'?     No. Entonces h[2] = '?'.
    i[3]: ¿'NO' es igual a 'NO'?                   Sí. No hace nada.
5. Al salir del for, tenemos la hipotesis (['?', '?', '?', 'NO'] en este caso)


PREDICCION:
Para la prediccion le pasamos la hipotesis que obtuvimos ['?', '?', '?', 'NO'] y el conjunto de prueba
La condición primero descarta los atributos generalizados (?) para enfocarse únicamente en los atributos determinantes.
osea compara el NO con los datos de la columna de prestamos impagos 
Si es distinto entonces tiene prestamos impagos y devuelve rechazado 
y si no tiene prestamos impagos devuelve otorgado


usando el conjunto de prueba se fija cuantas veces se falla y acierta en la preccion 
comparando el resultado de la prediccion (OTORGADO O RECHAZADO) con el valor real del de la solicitud del prestamo
osea para los no acierto:
 1.cuanta las veces que NO tiene prestamos impagos e igualmente se le RECHAZA la solicitud de un nuevo prestamo
 2.y la veces que SI tiene prestamos impagos y se le OTORGA la solicitud de un nuevo prestamo 
para los aciertos:(como lo predecimos)
 1.cuanta las veces que NO tiene prestamos impagos y se le OTORGA
 2.y la veces que SI tiene prestamos impagos y se le RECHAZA


OTROS DATOS:
La hipotesis de todos 0 es la mas especifica SOLO al comienzo, rechaza todo
La vamos generalizando a medida que encontramos ejemplos positivos(OTORGADO de estado del prestamos para nosotros)
Al terminas de recorrer al algoritmo FINDS tenemos la hipotesis mas general PERO esta vez esta generalizada solo con los ejemplos positivos





Tu Resumen "Pulido" quedaría así:
RESUMEN TOTAL:
Usamos Pandas para manipular el CSV como un DataFrame.
Filtramos por Edad == 50.Definimos Atributos (X) y Target (y).
Dividimos en Entrenamiento (75%) y Prueba (25%).

ALGORITMO FIND-S:
Empezamos con $h = [0, 0, 0, 0]$ (máxima especificidad).
Solo miramos los OTORGADO.La primera fila positiva "pisa" los ceros.
Las siguientes filas positivas comparan: si el valor cambia, se pone ?. Si se mantiene igual, queda el valor.
Resultado: El patrón común más restrictivo de los casos exitosos.

PREDICCIÓN:
Usamos la hipótesis como un filtro.
Los ? dejan pasar cualquier cosa. 
Los valores fijos (como "NO") exigen igualdad.
Si la fila de prueba no encaja perfectamente en los valores fijos, se predice RECHAZADO.

EVALUACIÓN:Comparamos Predicción vs. Realidad.
Aciertos: Cuando ambos coinciden (ambos OTORGADO o ambos RECHAZADO).
Fallos: Cuando el modelo es muy optimista (predice OTORGADO pero era RECHAZADO) o cuando es muy estricto (predice RECHAZADO pero era OTORGADO).

DATO CLAVE:
Find-S ignora los ejemplos negativos durante el entrenamiento. Solo aprende de lo que "está bien" (los OTORGADO), 
por eso su hipótesis final es la más específica que puede existir sin contradecir a los ejemplos positivos.
¡Con este resumen estás más que lista para defender el TP!
'''

######################################################################
##########  Métricas  ##########

# Calculamos los 4 valores posibles de los datos: True Positive (TP), False Positive (FP), True Negative (TN), False Negative (FN)
# para un vector de predicciones y uno de esperados. OTORGADO == Positive, RECHAZADO == Negative
def calcularPositivosYnegativos(predicciones, esperados):
  TP = 0
  FP = 0
  TN = 0
  FN = 0
  # Recorremos el vector de predicciones y comparamos el resultado con el esperado
  for i in range(len(predicciones)):
      if predicciones[i] == 'OTORGADO':   # Predicciones POSITVAS
        if esperados[i] == 'OTORGADO':
          TP += 1
        else:
          FP += 1
      else:                               # Predicciones NEGATIVAS
        if esperados[i] == 'RECHAZADO':
          TN += 1
        else:
          FN += 1
  return TP,FP,TN,FN

'''
con calcularPositivosYnegativos contamos cuantas veces el modelo predice OTORGADO y es correcto (TP), 
cuantas veces predice OTORGADO pero era RECHAZADO (FP), cuantas veces predice RECHAZADO y es correcto (TN) y 
cuantas veces predice RECHAZADO pero era OTORGADO (FN)
'''

# Creamos los conjuntos de predicciones y de esperados
conj_predicciones = []
conj_esperados = []

for i, registro in prueba.iterrows():
   conj_predicciones.append(predecir(hipotesis, registro, columnas))
   conj_esperados.append(registro[estado])

# Calculamos sus variables
TP, FP, TN, FN = calcularPositivosYnegativos(conj_predicciones, conj_esperados)

# Matriz de confusión para un conjunto de predicciones y esperados
def matrizDeConfusion(predicciones, esperados):
  tp, fp, tn, fn = calcularPositivosYnegativos(predicciones, esperados)
  return [[tp, fn], [fp, tn]]

def accuracy(tp, fp, tn, fn):
  return (tp + tn) / (tp + tn + fp + fn)

def recall(tp, fn):
  return tp / (tp + fn)

def especificidad(fp, tn):
  return tn / (tn + fp)

def precision(tp, fp):
  return tp / (tp + fp)

def f1_score(tp, fp, fn):
  return 2 * (precision(tp, fp) * recall(tp, fn)) / (precision(tp, fp) + recall(tp, fn))

def tpr_fpr(tp, fp, tn, fn):
  return tp / (tp + fn), fp / (tn + fp)

print(f"Matriz de confusión:")
matriz = matrizDeConfusion(conj_predicciones, conj_esperados)
print(matriz[0])
print(matriz[1])
print(f"Accuracy: {accuracy(TP,FP,TN,FN)}")
print(f"Recall: {recall(TP,FN)}")
print(f"Especificidad: {especificidad(FP,TN)}")
print(f"Precisión: {precision(TP,FP)}")
print(f"F1-score: {f1_score(TP,FP,FN)}")
print(f"TPR, FPR: {tpr_fpr(TP,FP,TN,FN)}")
#############################################################################################################


df = pd.read_csv('prestamo.csv', encoding='latin-1')

# Filtrar por edad entre 40 y 45
df_filtrado = df[(df['Edad'] >= 40) & (df['Edad'] <= 45)].copy()

columnas = ['Sexo', 'Mayor nivel educativo', 'Estado de vivienda', 'Préstamos previos impagos']
estado = 'Estado'

# Dividir en Entrenamiento (80%) y Prueba (20%)
limite  = int(len(df_filtrado) * 0.80) # saco la cantidad de el %80 de la cantidad de registros
entrenamiento = df_filtrado.iloc[:limite] # tomo los primeros registros para entrenamiento
prueba = df_filtrado.iloc[limite:]  # y los demas hasta el final para prueba

clases = entrenamiento[estado].unique()
N = len(entrenamiento) # total de ejemplos
d = len(columnas) # cantidad de atributos
l = 1  # suavizado de Laplace
K = len(clases)  # cantidad de clases

'''
separa entre la cantidad de clases, cantidad de atributos, cantidad de ejemplos, y el suavizado de Laplace (que es un valor que se le suma a cada conteo para evitar problemas de probabilidad cero en el caso de que no haya ejemplos de una clase con un valor específico de un atributo)
'''


# probabilidades de cada clase
pi = {}
for c in clases:
    # π_k = (#{Y = Ck} + l) / (N + l*K) 
    pi[c] = (len(entrenamiento[entrenamiento[estado] == c]) + l) / (N + l * K)
'''
calcula la probabilidad a priori de cada clase (π_k) usando la fórmula del estimador de máxima verosimilitud con suavizado de Laplace, 
donde #{Y = Ck} es el número de ejemplos en el conjunto de entrenamiento que pertenecen a la clase Ck, 
N es el total de ejemplos en el conjunto de entrenamiento, l es el parámetro de suavizado de Laplace, y 
K es el número total de clases.

el +l es por el suavizado de Laplace para que en caso de que haya alguna clase 0(no tenga ningún ejemplo) no nos de una probabilidad de 0 
'''
# cantidad de posibles valores de cada atributo
x = {}  # x[atributo][valor][clase]
'''
para cada atributo hay se ve la probabilidad de OTORGADO o RECHAZADO para cada valor posible de ese atributo, por ejemplo:
x['Sexo']['FEMENINO']['OTORGADO'] 
'''
for col in columnas:
    x[col] = {}
    valores_posibles = entrenamiento[col].unique()  # Mj valores que puede tomar cada atributo
    Mj = len(valores_posibles)

    for val in valores_posibles:
        x[col][val] = {}
        for c in clases:
            filas_clase = entrenamiento[entrenamiento[estado] == c]
            num   = len(filas_clase[filas_clase[col] == val]) + l
            denom = len(filas_clase) + l * Mj
            x[col][val][c] = num / denom
'''
x={
    'Sexo': {
        'FEMENINO': {'OTORGADO': 0.3, 'RECHAZADO': 0.1},
        'MASCULINO': {'OTORGADO': 0.2, 'RECHAZADO': 0.4}
    },
    'Mayor nivel educativo': {
        'UNIVERSITARIO': {'OTORGADO': 0.5, 'RECHAZADO': 0.2},
        'SECUNDARIO
        
para cada atributo (columna), se calcula la probabilidad condicional de cada valor del atributo dado cada clase, utilizando la fórmula del estimador de máxima verosimilitud con suavizado de Laplace, donde num es el número de ejemplos en la clase c que tienen el valor val para el atributo col, y denom es el número total de ejemplos en la clase c más el producto del suavizado de Laplace y la cantidad de valores posibles para ese atributo (Mj).
'''
print(f"Limite: {limite}\nEntrenamiento shape: {entrenamiento.shape}\nPrueba shape: {prueba.shape}\nClases: {clases}\nN: {N}\nK: {K}\nPi: {pi}\nx: {x}")

def calculoBayes(fila, pi, x, clases, columnas):

    scores = {}
    for c in clases:
        score = pi[c]
        for col in columnas:
            val = fila[col]
            score *= x[col][val][c] # P(Y=Ck) * Π P(Xj=val|Y=Ck)
        scores[c] = score
    return max(scores, key=scores.get), scores

'''
calculo de bayes, para cada clase se calcula un score multiplicando la probabilidad a priori de la clase (pi[c]) por las probabilidades condicionales de cada atributo dado esa clase (x[col][val][c]) para los valores de los atributos en la fila de prueba.
Luego se devuelve la clase con el score más alto como la predicción, junto con el diccionario de scores para cada clase.

max(scores, key=scores.get), scores devuelve la clase con el score más alto, es decir, la clase que tiene la mayor probabilidad de ser la correcta según el modelo de Naïve Bayes, junto con el diccionario completo de scores para cada clase.

'''

y_pred  = []
y_prob  = []
y_real  = []

clase_rech = clases[0]
clase_otorg = clases[1]

for _, fila in prueba.iterrows():
    pred, scores = calculoBayes(fila, pi, x, clases, columnas)
    y_pred.append(pred)
    y_real.append(fila[estado])

    # P1 / (P1 + P2)
    total = sum(scores.values())
    y_prob.append(scores[clase_otorg] / total if total > 0 else 0)
    
'''
usa el calculoBayes para hacer predicciones en el conjunto de prueba, almacenando las predicciones, las probabilidades y los valores reales en listas separadas.
Para cada fila en el conjunto de prueba, se calcula la predicción utilizando el modelo de Naïve Bayes, se almacena la predicción en y_pred, el valor real del estado en y_real, y se calcula la probabilidad de la clase "OTORGADO" dividiendo
'''


# Predicciones de cada caso (log)
# for i, (_, fila) in enumerate(prueba.iterrows()):
#     pred, scores = calculoBayes(fila, pi, x, clases, columnas)
#     print(f"Ejemplo {i+1}: Real={fila[estado]} | Pred={pred} | "
#           f"Score OTORGADO={scores['OTORGADO']:.6f} | Score RECHAZADO={scores['RECHAZADO']:.6f}")

TP, FP, TN, FN = calcularPositivosYnegativos(y_pred, y_real)

print(f"\nClase: {clase_otorg}")
print(f"\nMatriz de confusión:")
print(f"                Pred {clase_otorg}   Pred {clase_rech}")
print(f"Real {clase_otorg}      TP={TP}          FN={FN}")
print(f"Real {clase_rech}    FP={FP}          TN={TN}")


print(f"\nAccuracy  : {accuracy(TP,FP,TN,FN):.4f}")
print(f"Precisión : {precision(TP,FP):.4f}")
print(f"Recall    : {recall(TP,FN):.4f}")
print(f"F1-score  : {f1_score(TP,FP,FN):.4f}")

# GRAFICO Curva ROC
# score y clase real en una lista de pares ordenados
pares = sorted(zip(y_prob, y_real), reverse=True)

# print("Score | Verdad | TPR  | FPR")
# print("-" * 35)

total_pos = TP + FN   # P = total positivos reales
total_neg = FP + TN   # N = total negativos reales

puntos_fpr = [0]
puntos_tpr = [0]

tp = 0
fp = 0

for score, real in pares:
    # Bajamos el umbral hasta este score → esta fila pasa a clasificarse como positiva
    if real == clase_otorg:
        tp += 1   # era positivo y ahora lo clasificamos como positivo → VP
    else:
        fp += 1   # era negativo pero lo clasificamos como positivo → FP

    tpr = tp / total_pos   # TPR = VP / P
    fpr = fp / total_neg   # FPR = FP / N

    puntos_tpr.append(tpr)
    puntos_fpr.append(fpr)

    # print(f"{score:.2f}  | {real:10} | {tpr:.2f} | {fpr:.2f}")

auc_val = 0
for i in range(len(puntos_fpr) - 1):
    base   = puntos_fpr[i+1] - puntos_fpr[i]        # ancho del trapecio
    altura = (puntos_tpr[i+1] + puntos_tpr[i]) / 2  # altura promedio
    auc_val += base * altura

print(f"\nAUC = {auc_val:.4f}")






plt.figure(figsize=(6, 5))

# Curva ROC del modelo
plt.plot(puntos_fpr, puntos_tpr, color='#2E7D32', lw=2,
         marker='o', markersize=4, label=f'Naïve Bayes (AUC = {auc_val:.3f})')

# Línea del clasificador aleatorio (TPR = FPR)
plt.plot([0, 1], [0, 1], color='gray', lw=1.5,
         linestyle='--', label='Clasificador aleatorio (AUC = 0.5)')

plt.xlabel("FPR (Tasa de Falsos Positivos)")
plt.ylabel("TPR (Tasa de Verdaderos Positivos)")
plt.title("Curva ROC — Naïve Bayes Categórico")
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


