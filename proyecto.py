
######################################################################################################
################################################ PASO 1 ##############################################
# Crear un entorno de anaconda con los paquetes que considere necesarios.
######################################################################################################

#------------------------------------- inicio paquetes necesarios -------------------------------------

import numpy as np

import pandas as pd

import random

#------------------------------------- fin paquetes necesarios -------------------------------------

######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################


######################################################################################################
################################################ PASO 2 ##############################################
# Usando sclicing con NumPy separar los datos en 2 datasets: entrenamiento(80 %) y validación
# y pruebas(20 %).
######################################################################################################

#-------------------------------------  inicio data frame cargar datos -------------------------------------

#cargar datos con el nombre del archivo y el numero de columnas, tambien puede reducir en un porcentaje 

class Datos_Proyecto:

    def __init__(self, nombre_archivo,lista_columnas): # todos los atributos del objeto

        self.array_total=np.load(nombre_archivo) #cargar el archivo usando el nombre, debe estar en la carpeta y deber ser un array

        self.lista_columnas_1=lista_columnas #lista del nombre de las columnas de array

        rows, columns = self.array_total.shape #obtener cantidad de filas, columnas del array

        self.cantidad_filas=rows #cantidad de filas del array

        self.array_tolist=self.array_total.tolist() # convertir en lista el array para poder usarlo en pandas

        self.dataframe_total=pd.DataFrame(self.array_tolist, columns=self.lista_columnas_1) # crear el dataframe y nombrar las columnas

        self.data_Reducido_trabajar=self.dataframe_total

        self.data_Reducido_comparar=self.dataframe_total




    def reducir_data(self,porcentaje_reducir): #metodo para reducir el dataframe, porcentaje a reducir

        lista_reducir=[]

        if porcentaje_reducir<1 and porcentaje_reducir>0: # condicionante para reducir

            cantidad_filas_porcentaje=int((self.cantidad_filas)*porcentaje_reducir) #cantidad de filas del porcentaje a reducir


            while len(lista_reducir)<cantidad_filas_porcentaje: # terminar el proceso hasta que se tengan todos los valores de la lista

                valor_random=random.randint(0,self.cantidad_filas-1) # seleccionar un valor random, entre 0 y la cantidad de datos

                if lista_reducir.count(valor_random)==0: # comparar que no exista el valor random
                
                    lista_reducir.append(valor_random) #agregar a la lista el valor random
            
            
            lista_reducir.sort() # ordenar la lista a reducir

            set_range_cantida_filas=set(list(range(self.cantidad_filas-1))) # conjunto de todas las filas dataframe

            lista_reducida_trabajar=list(set_range_cantida_filas.difference(set(lista_reducir))) #diferencia de conjuntos, lista de los valores a trabajar

            self.data_Reducido_trabajar=self.dataframe_total.drop(lista_reducir, axis=0) #dataframe de las filas que se trabajaran

            self.data_Reducido_comparar=self.dataframe_total.drop(lista_reducida_trabajar, axis=0) #dataframe de las filas que se compararan

            return self.data_Reducido_trabajar, self.data_Reducido_comparar

        else:

            print("Valor incorrecto")



    def media_columnas(self): #metodo para obtener los valores medios

        dic_valores_medios={}

        for i in self.lista_columnas_1:

            data_columna=(self.data_Reducido_trabajar[i].dropna()).values.tolist()

            valor_medio=sum(data_columna)/len(data_columna)

            dic_valores_medios[i]=round(valor_medio, 3)

        return dic_valores_medios




#-------------------------------------  fin data frame cargar datos -------------------------------------

lista_1=["PRECIO",
    "CALIDAD_MATERIAL",
    "AREA_PISO",
    "TOTAL_HABITACIONES",
    "AÑO_CONSTRUCCION",
    "FRENTE"] # LISTA DE LOS NOMBRES DE LAS COLUMNAS

datos_1=Datos_Proyecto("proyecto_training_data.npy",lista_1) # leer los datos


data_Reducido_from_datos_1=datos_1.reducir_data(0.2) # reducir los datos en el 20%


print(datos_1.dataframe_total) # DATAFRAME DE TODOS LOS DATOS 

print(data_Reducido_from_datos_1[0]) # DATAFRAME  entrenamiento(80 %)

print(data_Reducido_from_datos_1[1]) # DATAFRAME pruebas(20 %)

print("\n")

######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################



######################################################################################################
################################################ PASO 3 ##############################################
# Análisis exploratorio de datos: Para cada variable en el dataset calcular((usando numpy o
# pandas):

# media
# valor máximo
# valor mínimo
# valor rango(peak to peak, no el rango del tensor que por ser vector sabemos que es 1)
# desvianción estándar

######################################################################################################

#------------------------------------- inicio data frame cargar datos -------------------------------------

dic_valores_medios=datos_1.media_columnas()

for i in dic_valores_medios:

    print(i)
    print(dic_valores_medios[i])
    print()






#------------------------------------- fin data frame cargar datos -----------------------------------------



######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################


input()


#matrix_1Reducida=dataframe_elemento.drop(lista_surtida, axis=1)


