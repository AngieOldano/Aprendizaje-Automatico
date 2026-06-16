import pandas as pd
import numpy as np
from graphviz import Digraph
import matplotlib.pyplot as plt
import random
#random.seed(42)  

#########  Métricas TP1 ##########
# Reutilizamos las funciones de métricas que definimos en el TP1 para evaluar el desempeño del Random Forest en el conjunto de prueba.
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


# Matriz de confusión para un conjunto de predicciones y esperados
#def matrizDeConfusion(predicciones, esperados):
#  tp, fp, tn, fn = calcularPositivosYnegativos(predicciones, esperados)
#  return [[tp, fn], [fp, tn]]

def accuracy(tp, fp, tn, fn): 
  return (tp + tn) / (tp + tn + fp + fn)

def recall(tp, fn):
  return tp / (tp + fn)

#def especificidad(fp, tn):
#  return tn / (tn + fp)

def precision(tp, fp):
  return tp / (tp + fp)

def f1_score(tp, fp, fn):
  return 2 * (precision(tp, fp) * recall(tp, fn)) / (precision(tp, fp) + recall(tp, fn))

#def tpr_fpr(tp, fp, tn, fn):
#  return tp / (tp + fn), fp / (tn + fp)

#########  Métricas TP1  ##########

df = pd.read_csv('prestamo.csv', encoding='latin-1')

estado = 'Estado'
atributos = ["Edad", "Sexo", "Mayor nivel educativo", "Ingreso anual", "Estado de vivienda", "Monto solicitado", "Préstamos previos impagos"]

# Filtrar por entre 40 y 45 años
df_40_45 = df[df['Edad'].between(40, 45)].copy()

# Dividir en Entrenamiento (80%) y Prueba (20%)
limite  = int(len(df_40_45) * 0.80)
entrenamiento = df_40_45.iloc[:limite]
prueba = df_40_45.iloc[limite:]


def entropia(atributo):
    _, cantidad = np.unique(atributo, return_counts=True)
    probabilidades = cantidad / np.sum(cantidad)
    entropia = -np.sum(probabilidades * np.log2(probabilidades))
    return entropia

def ganancia_info(conjunto, atributo_predictor, atributo_objetivo):
    entropia_total = entropia(conjunto[atributo_objetivo]) # entropia base del nodo

    valores_unicos, conteos = np.unique(conjunto[atributo_predictor], return_counts=True) # valores posibles de la columna (por ej. edad), cantidad de cada uno
    total_filas = len(conjunto)

    entropia_ramas = 0

    for i in range(len(valores_unicos)):
        valor = valores_unicos[i]
        peso = conteos[i] / total_filas

        subconjunto = conjunto[conjunto[atributo_predictor] == valor] # filtramos los datos que pertenecen a esta rama específica
        entropia_subconjunto = entropia(subconjunto[atributo_objetivo]) # calculamos la entropía del atributo

        entropia_ramas += peso * entropia_subconjunto

    return entropia_total - entropia_ramas

def ganancia_informacion_continua(df, atributo_predictor, atributo_objetivo):
    entropia_total = entropia(df[atributo_objetivo])
    total_filas = len(df)

    valores_ordenados = np.sort(df[atributo_predictor].unique())

    mejor_ganancia = -1
    mejor_corte = None

    # Iteración sobre los posibles puntos de corte
    for i in range(len(valores_ordenados) - 1):
        corte = (valores_ordenados[i] + valores_ordenados[i+1]) / 2.0

        grupo_izq = df[df[atributo_predictor] <= corte]
        grupo_der = df[df[atributo_predictor] > corte]

        # Pesos de cada rama
        peso_izq = len(grupo_izq) / total_filas
        peso_der = len(grupo_der) / total_filas

        entropia_ramas = (peso_izq * entropia(grupo_izq[atributo_objetivo]) +  # entropía ponderada
                          peso_der * entropia(grupo_der[atributo_objetivo]))

        ganancia_actual = entropia_total - entropia_ramas

        if ganancia_actual > mejor_ganancia: # si es la mejor ganancia, la guarda
            mejor_ganancia = ganancia_actual
            mejor_corte = corte

    return mejor_ganancia, mejor_corte

def mayor_ganancia(conjunto, atributos_disponibles, atributo_objetivo):
    mejor_ganancia = -1
    mejor_atributo = None
    mejor_corte_final = None
    es_atributo_continuo = False

    for atributo in atributos_disponibles:
        # Si es número (continuo)
        if pd.api.types.is_numeric_dtype(conjunto[atributo]):
            ganancia, corte = ganancia_informacion_continua(conjunto, atributo, atributo_objetivo)

            if ganancia > mejor_ganancia:
                mejor_ganancia = ganancia
                mejor_atributo = atributo
                mejor_corte_final = corte
                es_atributo_continuo = True

        # Si es texto/categoría
        else:
            ganancia = ganancia_info(conjunto, atributo, atributo_objetivo)

            if ganancia > mejor_ganancia:
                mejor_ganancia = ganancia
                mejor_atributo = atributo
                mejor_corte_final = None
                es_atributo_continuo = False

    return mejor_atributo, mejor_corte_final, es_atributo_continuo

def id3(conjunto, atributos_disponibles, atributo_objetivo):
    # Casos Base
    valores_objetivo = conjunto[atributo_objetivo].unique()
    if len(valores_objetivo) == 1:
        return valores_objetivo[0] # Pureza total

    if len(atributos_disponibles) == 0:
        return conjunto[atributo_objetivo].mode()[0] # No hay más atributos

    mejor_atributo, mejor_corte_final, es_atributo_continuo = mayor_ganancia(
        conjunto, atributos_disponibles, atributo_objetivo
    )

    if mejor_atributo is None:
        return conjunto[atributo_objetivo].mode()[0]

    arbol = {mejor_atributo: {}} # raíz del sub-arbol

    # Dividimos el conjunto dependiendo del tipo de dato
    if es_atributo_continuo and mejor_corte_final is not None:
        # CONTINUOS
        atributos_restantes = [a for a in atributos_disponibles if a != mejor_atributo] # guarda un array con los atributos pero dejando fuera el que fue usado

        sub_izq = conjunto[conjunto[mejor_atributo] <= mejor_corte_final]
        sub_der = conjunto[conjunto[mejor_atributo] > mejor_corte_final]

        # Rama Izquierda
        if len(sub_izq) == 0:
            arbol[mejor_atributo][f"<= {mejor_corte_final}"] = conjunto[atributo_objetivo].mode()[0] # Si no hay grupo a la izquierda, calcula el dato que más se repite en el grupo
        else:
            arbol[mejor_atributo][f"<= {mejor_corte_final}"] = id3(sub_izq, atributos_restantes, atributo_objetivo) # Si hay más elementos, usa la función recursiva en la rama izq.

        # Rama Derecha
        if len(sub_der) == 0:
            arbol[mejor_atributo][f"> {mejor_corte_final}"] = conjunto[atributo_objetivo].mode()[0] # Si no hay grupo a la derecha, calcula el dato que más se repite en el grupo
        else:
            arbol[mejor_atributo][f"> {mejor_corte_final}"] = id3(sub_der, atributos_restantes, atributo_objetivo) # Si hay más elementos, usa la función recursiva en la rama der.

    else:
        # CATEGÓRICOS
        atributos_restantes = [a for a in atributos_disponibles if a != mejor_atributo] # guarda un array con los atributos pero dejando fuera el que fue usado

        for valor in conjunto[mejor_atributo].unique():
            subconjunto = conjunto[conjunto[mejor_atributo] == valor]

            if len(subconjunto) == 0:
                arbol[mejor_atributo][valor] = conjunto[atributo_objetivo].mode()[0]
            else:
                arbol[mejor_atributo][valor] = id3(subconjunto, atributos_restantes, atributo_objetivo)

    return arbol

arbol = id3(entrenamiento, atributos, estado)

# Función para graficar el árbol
def exportar_a_graphviz(arbol, nombre_archivo="arbol_decision"):
    """
    Convierte el diccionario del árbol ID3 en un gráfico visual usando Graphviz
    y lo guarda como una imagen/PDF.
    """
    dot = Digraph(comment='Árbol de Decisión ID3')

    # Configuramos estilos bonitos para los nodos y flechas
    dot.attr('node', shape='box', style='filled, rounded', color='#1f77b4',
             fontname='Arial', fillcolor='#e1f5fe', penwidth='2')
    dot.attr('edge', fontname='Arial', fontsize='10', color='#757575')

    contador_nodos = 0

    def agregar_nodos_recursivo(sub_arbol, padre_id=None, etiqueta_flecha=""):
        nonlocal contador_nodos

        # Generar un ID único para este nodo
        nodo_id = f"nodo_{contador_nodos}"
        contador_nodos += 1

        # CASO BASE: Es una hoja (OTORGADO / RECHAZADO)
        if not isinstance(sub_arbol, dict):
            # Color verde para otorgado, rojo para rechazado
            color_hoja = '#c8e6c9' if sub_arbol == 'OTORGADO' else '#ffcdd2'
            texto_hoja = f"STATUS:\n{sub_arbol}"

            dot.node(nodo_id, texto_hoja, fillcolor=color_hoja, shape='ellipse', color='#2e7d32' if sub_arbol == 'OTORGADO' else '#c62828')
            if padre_id:
                dot.edge(padre_id, nodo_id, label=etiqueta_flecha)
            return

        # CASO RECURSIVO: Es un nodo de pregunta (Atributo)
        nodo_nombre = list(sub_arbol.keys())[0]
        dot.node(nodo_id, nodo_nombre, fillcolor='#e1f5fe')

        # Si tiene padre, lo conectamos con la flecha que trae la condición
        if padre_id:
            dot.edge(padre_id, nodo_id, label=etiqueta_flecha)

        # Recorrer las ramas de este atributo
        for condicion, hijo_arbol in sub_arbol[nodo_nombre].items():
            agregar_nodos_recursivo(hijo_arbol, nodo_id, str(condicion))

    # Iniciar la recursión desde la raíz del árbol
    agregar_nodos_recursivo(arbol)

    # Guardar y renderizar el gráfico
    dot.render(nombre_archivo, format='png', cleanup=True)
    print(f"¡Árbol exportado con éxito! Revisa el archivo '{nombre_archivo}.png'")

# --- CÓMO EJECUTARLO ---
exportar_a_graphviz(arbol, "mi_arbol_prestamos")

# Predecir con un árbol dado una fila
# Recorre el diccionario del árbol hasta llegar a una hoja
def predecir(arbol, fila):
    # Si no es diccionario, llegamos a una hoja → es la clase
    if not isinstance(arbol, dict):
        return arbol

    # Obtenemos el atributo de este nodo
    atributo = list(arbol.keys())[0]
    ramas    = arbol[atributo]
    valor    = fila[atributo]

    # Atributo continuo: busca la rama "<= X" o "> X"
    if pd.api.types.is_numeric_dtype(type(valor)): 
        for condicion, sub_arbol in ramas.items():
            if condicion.startswith("<="):
                corte = float(condicion.split("<=")[1].strip())
                if valor <= corte:
                    return predecir(sub_arbol, fila)
            elif condicion.startswith(">"):
                corte = float(condicion.split(">")[1].strip())
                if valor > corte:
                    return predecir(sub_arbol, fila)

    # Atributo categórico
    else:
        if valor in ramas:
            return predecir(ramas[valor], fila)

    # Si no encuentra el valor (no estaba en entrenamiento)
    # devolvemos la clase más frecuente
    return entrenamiento[estado].mode()[0]

# ---------------------------------------------------------
# 3. Matriz de confusión
# ---------------------------------------------------------
y_pred_id3 = []
y_real_id3 = list(prueba[estado])

for _, fila in prueba.iterrows():
    prediccion = predecir(arbol, fila)
    y_pred_id3.append(prediccion)

TP, FP, TN, FN = calcularPositivosYnegativos(y_pred_id3, y_real_id3)

print(f"\nMatriz de confusión — ID3:")
print(f"                Pred OTORGADO   Pred RECHAZADO")
print(f"Real OTORGADO      TP={TP}          FN={FN}")
print(f"Real RECHAZADO     FP={FP}          TN={TN}")

# ---------------------------------------------------------
# 4. Accuracy y F1-score
# ---------------------------------------------------------
print(f"\nAccuracy : {accuracy(TP, FP, TN, FN):.4f}")
print(f"F1-score : {f1_score(TP, FP, FN):.4f}")



# Bootstrap: muestra aleatoria de tamaño N con reemplazo
def bootstrap(conjunto):
    n = len(conjunto)
    indices = []
    for _ in range(n):
        indices.append(random.randint(0, n - 1))  # elegimos un índice al azar (puede repetirse)
    return conjunto.iloc[indices]


# 1. Entrenar el Random Forest con 10 árboles
N_ARBOLES = 10
bosque = []

for i in range(N_ARBOLES):
    muestra  = bootstrap(entrenamiento)          # muestra distinta para cada árbol
    arbol_i  = id3(muestra, atributos, estado)   # entrenamos el árbol con esa muestra
    bosque.append(arbol_i)
    print(f"Árbol {i+1} entrenado ✓")
    exportar_a_graphviz(arbol_i, f"mi_arbol_{i+1}")
    

#Predecir sobre el conjunto de prueba
y_pred_rf = []
y_real_rf = list(prueba[estado])

for _, fila in prueba.iterrows():
    votos = []
    for arbol_i in bosque:
        votos.append(predecir(arbol_i, fila))    # cada árbol emite su voto
    
    # La clase con más votos gana
    clase_ganadora = max(set(votos), key=votos.count)
    y_pred_rf.append(clase_ganadora)

# 2. Matriz de confusión
TP, FP, TN, FN = calcularPositivosYnegativos(y_pred_rf, y_real_rf)

print(f"\nMatriz de confusión — Random Forest:")
print(f"                Pred OTORGADO   Pred RECHAZADO")
print(f"Real OTORGADO      TP={TP}          FN={FN}")
print(f"Real RECHAZADO     FP={FP}          TN={TN}")

# 3. Accuracy y F1-score
print(f"\nAccuracy : {accuracy(TP, FP, TN, FN):.4f}")
print(f"F1-score : {f1_score(TP, FP, FN):.4f}")


# 4. Gráfico de precisión 
#calculamos accuracy en entrenamiento y en prueba
acc_train = []
acc_test  = []

for n in range(1, N_ARBOLES + 1):
    bosque_parcial = bosque[:n]  

    #Accuracy en ENTRENAMIENTO
    preds_train = []
    for _, fila in entrenamiento.iterrows():
        votos = []
        for arbol_i in bosque_parcial:
            votos.append(predecir(arbol_i, fila))
        preds_train.append(max(set(votos), key=votos.count))

    TP_tr, FP_tr, TN_tr, FN_tr = calcularPositivosYnegativos(preds_train, list(entrenamiento[estado]))
    acc_train.append(accuracy(TP_tr, FP_tr, TN_tr, FN_tr))

    #Accuracy en PRUEBA
    preds_test = []
    for _, fila in prueba.iterrows():
        votos = []
        for arbol_i in bosque_parcial:
            votos.append(predecir(arbol_i, fila))
        preds_test.append(max(set(votos), key=votos.count))

    TP_te, FP_te, TN_te, FN_te = calcularPositivosYnegativos(preds_test, list(prueba[estado]))
    acc_test.append(accuracy(TP_te, FP_te, TN_te, FN_te))

    print(f"Bosque de {n:2d} árbol(es) → Train: {acc_train[-1]:.4f} | Test: {acc_test[-1]:.4f}")

# Gráfico
plt.figure(figsize=(8, 5))
plt.plot(range(1, N_ARBOLES + 1), acc_train, marker='o', color='#1565C0', lw=2, label='Entrenamiento')
plt.plot(range(1, N_ARBOLES + 1), acc_test,  marker='s', color='#2E7D32', lw=2, label='Prueba')
plt.xlabel("Cantidad de árboles en el bosque")
plt.ylabel("Accuracy")
plt.title("Precisión vs Tamaño del Bosque — Random Forest")
plt.xticks(range(1, N_ARBOLES + 1))
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


#¿Cuál de los métodos se recomienda utilizar para predecir si se otorgaría el crédito bancario solicitado por una persona? Justificar la respuesta.
#Conclusión:
#Se recomienda Random Forest porque:
#  1. Promedia las predicciones de 10 árboles → reduce el overfitting del ID3.
#  2. Cada árbol se entrena con una muestra bootstrap diferente → más diversidad.
#  3. En general obtiene mejor accuracy y F1 que un único árbol ID3.
#  4. El ID3 solo puede memorizar el conjunto de entrenamiento (overfitting),
#     mientras que el bosque generaliza mejor a datos nuevos.
