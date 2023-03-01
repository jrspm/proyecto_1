
######################################################################################################
################################################ PASO 1 ##############################################
# Crear un entorno de anaconda con los paquetes que considere necesarios.
######################################################################################################

#------------------------------------- inicio paquetes necesarios -------------------------------------

import numpy as np # paquete para arrays

import pandas as pd # paquete para dataframes

import random # se uso este paquete porue se queria que los valores para reducir el porcentaje fuera aleatorio

import seaborn as sns

import matplotlib.pyplot as plt

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


    def valor_max_med_min_ptp_desv(self,list_num_int,redondear): #metodo para obtener los valores max_min_medio_ptp_desvi, ingresar 

        dic_val_max_med_min_ptp_desv={"VALORES":("MAXIMO","MEDIA","MINIMO","PEAK TO PEAK","DESVIACION")}

        for i in self.lista_columnas_1:

            data_1=self.data_Reducido_trabajar[i] #DATAFRAME DE CADA COLUMNA i 

            val_max=data_1.max() # obtener el valor maximo del dataframe
            val_med=data_1.mean() # obtener el valor medio del dataframe
            val_min=data_1.min() # obtener el valor minimo del dataframe 
            val_des=data_1.dropna().std() # obtener la desviacion estandar quitando los nan
            val_ptp=np.ptp(data_1.dropna()) # obtener el peak to peak quitando los nan

            val_cancatenados=[val_max,val_med,val_min,val_ptp,val_des] # hacer una lista de los valores

            if list_num_int.count(i)>0:

                val_cancatenados = list(map(int, val_cancatenados)) # redondear enteros cuando la columna lo requiera

            else:

                val_cancatenados =[round(i,redondear) for i in val_cancatenados] # redondear hasta la cifra indicada


            val_cancatenados_1=tuple(val_cancatenados) # convertir en tupla la lista

            dic_val_max_med_min_ptp_desv[i]=val_cancatenados_1 # guardar en el diccionario la tupla con los valores segun la columna

        data_valores = pd.DataFrame(dic_val_max_med_min_ptp_desv) # crear un dataframe apartir de un diccionario
            
        return data_valores #regresa un dataframe con todos los valores

    def histograma_parejas(self,list_graf): #metodo para obtener los valores max_min_medio_ptp_desvi, ingresar 

        for i in list_graf:

            #------- inicio comprobar distribucion --------------------

            #self.data_Reducido_trabajar.sort_values(i[1])

            #data_1=self.data_Reducido_trabajar[i].dropna() #DATAFRAME DE CADA COLUMNA i

            #data_1=data_1.value_counts() #para contar la catidad de valores que hay de un mismo valor 

            #data_1_dic=data_1.to_dict()

            #keys = data_1_dic.keys()
            #sorted_keys = sorted(keys)

            #data_2_dic = {key:data_1_dic[key] for key in sorted_keys}

            #print(data_2_dic)

            #------- fin comprobar distribucion --------------------              

            #------- inicio forma 1 ------------

            #self.data_Reducido_trabajar[i].dropna().hist()

            #plt.show()

            #------- fin forma 1 ------------

            #------- inicio forma 2 ------------

            #data_1=self.data_Reducido_trabajar[i].dropna() #DATAFRAME DE CADA COLUMNA i
       
            #res = pd.Series(data_1,name="Range")

            #plot = sns.distplot(res,rug=False,hist=True)
            
            #plt.show()

            #------- fin forma 2 ------------

            #------- inicio forma 3 ------------

            data_1=self.data_Reducido_trabajar[i].dropna().values.tolist() #DATAFRAME DE CADA COLUMNA i    

            df = pd.DataFrame(data_1, columns=[i])

            plot = sns.displot(data = df, x=i )

            plt.title("")
            plt.xticks(fontsize=9, rotation=45)
            plt.xlabel(i,fontsize=12)
            plt.ylabel('Counted',fontsize=12)
            plt.tight_layout()
            
        plt.show()
        
            #------- fin forma 3 ------------

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

#------------------------------------- inicio max_min_medio_ptp_desvi -------------------------------------

lista_enteros=["AÑO_CONSTRUCCION"] # columnas que deben ser enteros


#ingresar los valores de la lista de las columnas que deben ser enteros y el valor de los decimales
data_valores=datos_1.valor_max_med_min_ptp_desv(lista_enteros,4) #obtener la media de cada columna en un diccionario

#------------------------------------- fin  max_min_medio_ptp_desvi ------------------------------------------

print(data_valores)

######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################

######################################################################################################
################################################ PASO 4 ##############################################

#------------------------------------- inicio imprimir histogramas -------------------------------------

#ingresar el listado de las parejas de las columnas que se desean graficar [[y1,x1],[y2,x2]]

lista_of_columns=["PRECIO","CALIDAD_MATERIAL","AREA_PISO","TOTAL_HABITACIONES","AÑO_CONSTRUCCION","FRENTE"]

datos_1.histograma_parejas(lista_of_columns)

#------------------------------------- fin imprimir histogramas------------------------------------------


######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################

######################################################################################################
################################################ PASO 5 ##############################################




######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################

input()


#matrix_1Reducida=dataframe_elemento.drop(lista_surtida, axis=1)


