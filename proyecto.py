import numpy as np

import pandas as pd

lista_1=["PRECIO",
    "CALIDAD_MATERIAL",
    "AREA_PISO",
    "TOTAL_HABITACIONES",
    "AÑO_CONSTRUCCION",
    "FRENTE"]

#print(lista_1)


datos=np.load('proyecto_training_data.npy') # cargar proyecto_training_data.npy


datos_1=datos.tolist()

#print(datos_1)


dataframe_elemento=pd.DataFrame(datos_1,
    columns=lista_1)



print(dataframe_elemento)

