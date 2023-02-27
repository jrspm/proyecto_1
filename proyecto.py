import numpy as np

import pandas as pd

import random

lista_1=["PRECIO",
    "CALIDAD_MATERIAL",
    "AREA_PISO",
    "TOTAL_HABITACIONES",
    "AÑO_CONSTRUCCION",
    "FRENTE"]

#print(lista_1)

class Datos_Proyecto:

    def __init__(self, nombre_archivo,lista_columnas):

        self.array_total=np.load(nombre_archivo)
        self.lista_columnas_1=lista_columnas

        rows, columns = self.array_total.shape

        self.cantidad_filas=rows

        self.cantidad_filas_porcentaje=0



    def funcion_1(self,porcentaje_reducir):

        if porcentaje_reducir<1 and porcentaje_reducir>0:

            self.cantidad_filas_porcentaje=(self.cantidad_filas)*porcentaje_reducir

        else:

            print("Valor incorrecto")

        return self.array_total


datos_1=Datos_Proyecto("proyecto_training_data.npy",lista_1)


array_1=datos_1.funcion_1(0.2)

print(array_1)






#matrix_1Reducida=dataframe_elemento.drop(lista_surtida, axis=1)


