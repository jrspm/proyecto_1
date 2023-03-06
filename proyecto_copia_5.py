
######################################################################################################
################################################ PASO 1 ##############################################
# Crear un entorno de anaconda con los paquetes que considere necesarios.
######################################################################################################

# py -m pip install seaborn  para actulizar paquetes en python CMD, cuando si funciona en anaconda

# si no corre usar power shell para ver los errores

#------------------------------------- inicio paquetes necesarios -------------------------------------

import numpy as np # paquete para arrays

import pandas as pd # paquete para dataframes

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

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

            lista_total_valores=range(self.cantidad_filas)


            while len(lista_reducir)<cantidad_filas_porcentaje: # terminar el proceso hasta que se tengan todos los valores de la lista

                valor_random=np.random.choice(lista_total_valores) # seleccionar un valor random, entre 0 y la cantidad de datos

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

            if list_num_int.count(i)>0: #comparar si una de las columnas esta en el listado de enteros

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

            try:

                data_1=self.data_Reducido_trabajar[i].dropna().values.tolist() #DATAFRAME DE CADA COLUMNA i    

                df = pd.DataFrame(data_1, columns=[i]) # nombrar a la columna 

                plot = sns.displot(data = df, x=i )

                plt.title("")
                plt.xticks(fontsize=9, rotation=45)
                plt.xlabel(i,fontsize=12)
                plt.ylabel('Counted',fontsize=12)
                plt.tight_layout()
                
            except:
                print("ERROR AL GENERAR LA GRAFICA DE LOS HISTOGRAMAS")
                pass

            #------- fin forma 3 ------------

        plt.show()

    def factor_correlacion(self,lista_graf_2,num_parejas):

        list_corr=[]

        cont_1=0

        lista_columns_2=[]

        if num_parejas>len(lista_graf_2):

            print("ERROR EL VALOR DE LAS PAREJAS ES MAYOR A LA CANTIDAD DE LAS LISTAS")

        else:

            try:

                for i in lista_graf_2:

                    cont_1+=1

                    data_1=self.data_Reducido_trabajar[[i[1],i[0]]].dropna()

                    corr_1=data_1.corr().values.tolist()

                    key=i[1]+" VS "+i[0]

                    corr_value=round(corr_1[0][1],4)

                    lista_columns_2.append(key)

                    list_corr.append(corr_value)

                    array_data_1=data_1.values.tolist()

                    array2=np.reshape(array_data_1,-1)

                    lista_x=array2[0:-1:2]

                    lista_y=array2[1::2]          


                    plt.scatter(lista_x,lista_y)
                    plt.title(f"{key} ---- Corr:{corr_value}", fontsize = 12)
                    plt.xticks(fontsize=9, rotation=45)
                    plt.xlabel(i[1],fontsize=12)
                    plt.ylabel(i[0],fontsize=12)
                    plt.tight_layout()
                    plt.show()


                data_valores_corr = pd.DataFrame(list_corr, index=lista_columns_2, columns=["CORRELATIVO"])

                data_valores_corr = data_valores_corr.sort_values('CORRELATIVO',ascending=False)

                data_valores_entre_0_1=data_valores_corr.loc[data_valores_corr["CORRELATIVO"]!= 1, :]

                if (data_valores_entre_0_1.shape[0])>=num_parejas:

                    row_1=data_valores_entre_0_1.head(num_parejas)

                else:
        
                    row_1=data_valores_corr

                return data_valores_corr, row_1

            except: 

                print("Error al generar las graficas de las correlaciones")
                
                pass

    def entrenamiento_80(self,lista_graf_2,b1,b0,error,epchs,alpha):

        data_1=self.data_Reducido_trabajar[lista_graf_2].dropna()

        data_1_comparar=self.data_Reducido_comparar[lista_graf_2].dropna()

        num=data_1.shape[0]

        array_list_1=[1]*(num)

        data_1.insert(2, "ones", array_list_1)

        data_2=pd.DataFrame(columns=["b1","b0","error","epoch","pend_1","pend_2"])

        def det_error(data_1,lista_graf_2,b0_0,b1_1,b0_ant,b1_ant,error_ant,error_ant_x,alpha_x_y):

            array_temp_x_1=np.array(data_1[[lista_graf_2[1],"ones"]].values.tolist())

            array_temp_y=np.array(data_1[lista_graf_2[0]].values.tolist())

            array_b1_bo=np.array([b1_1,b0_0])

            array_y_result=np.dot(array_temp_x_1,array_b1_bo)

            data_temp=pd.DataFrame({'x':array_temp_x_1[:, :1].reshape(-1).tolist(),'y_result':array_y_result.tolist(), "y":array_temp_y.tolist()})

            data_temp['(y_result - y)'] = data_temp['y_result'] - data_temp['y']

            data_temp['(y_result - y)^2']=data_temp['(y_result - y)']*data_temp['(y_result - y)']

            data_temp['x(y_result - y)']=data_temp['x']*data_temp['(y_result - y)']

            n_1=float(data_temp.shape[0])

            sum_3=(-1)*(data_temp['(y_result - y)'].sum())
            sum_3_x=(-1)*(data_temp['x(y_result - y)'].sum())


            error_actual=sum_3
            error_actual_x=sum_3_x


            pend_1=(b1_ant-b1_1)/(error_ant-error_actual)
            pend_2=(b0_ant-b0_0)/(error_ant-error_actual)

            b_1_next=b1_1-alpha_x_y*sum_3_x*pend_1
            b_0_next=b0_0-alpha_x_y*sum_3*pend_2

            return b_1_next, b_0_next, error_actual, b1_1, b0_0, pend_1, pend_2,error_actual_x


        def det_error_2(data_1,lista_graf_2,b0_train,b1_train):

            array_temp_x_1=np.array(data_1[[lista_graf_2[1],"ones"]].values.tolist())

            array_temp_y=np.array(data_1[lista_graf_2[0]].values.tolist())

            array_b1_bo=np.array([b1_train,b0_train])

            array_y_result=np.dot(array_temp_x_1,array_b1_bo)

            data_temp=pd.DataFrame({'x':array_temp_x_1[:, :1].reshape(-1).tolist(),'y_result':array_y_result.tolist(), "y":array_temp_y.tolist()})

            data_temp['(y_result - y)'] = data_temp['y_result'] - data_temp['y']

            error_train=abs((data_temp['(y_result - y)'].sum()))

            return error_train


        cont_1=0

        cont_2=0

        cont_3=0

        list_epocs=[]

        dic_epoca_bo_b1={}

        var_1=False

        cont_true=0

        cont_4=0

        cont_idex=0

        last_value_b1=0 
        last_epoch=0
        last_value_b0=0
        last_value_error=0
        error_train=0
        train_b1=0
        train_b0=0

        while True:

            if cont_1==0:

                # print("..................................................................")


                # print("ingreso")

                # print(f"  b0_next={b0}")
                # print(f"  b1_next={b1}")
                # print(f"  error_ant_actual={alpha}")

                                                                                            #det_error(data_1,lista_graf_2,b0_0,b1_1,b0_ant,b1_ant,error_ant,error_ant_x)
                b_1_next, b_0_next, error_actual, b1_1, b0_0, pend_1, pend_2,error_actual_x=det_error(data_1,lista_graf_2,b0,b1,0,0,error,error,alpha)

                error_actual_abs=abs(error_actual)

                data_2.loc[cont_idex]=[b1_1, b0_0,error_actual_abs,cont_1,pend_1,pend_2]

                cont_idex+=1
                
                dic_epoca_bo_b1[cont_1]=[b0_0,b1_1,error_actual_abs]

                # print("resultados")

                # print(f"  b0_next={b_0_next}")
                # print(f"  b1_next={b_1_next}")
                # print(f"  error_actual={error_actual}")
                # print(f"  b0_ant={b0_0}")
                # print(f"  b1_ant={b1_1}")
                # print(f"  pend_b1={pend_1}")
                # print(f"  pend_b0={pend_2}")
                # print(f"  error_actual_x={error_actual_x}")

                # print("..................................................................")



            else:

                pent_ant_1=pend_1
                pent_ant_2=pend_2
                error_ant_x=error_actual_x
                error_ant=error_actual
                b1_1_ant=b1_1
                b0_0_ant=b0_0

                # print("..................................................................")


                # print("ingreso")

                # print(f"  b0_next={b_0_next}")
                # print(f"  b1_next={b_1_next}")
                # print(f"  b0_ant={b0_0_ant}")
                # print(f"  b1_ant={b1_1_ant}")
                # print(f"  error_ant={error_ant}")
                # print(f"  error_ant_x={error_ant_x}")   

                                                                                            #det_error(data_1,lista_graf_2,b0_0,b1_1,b0_ant,b1_ant,error_ant,error_ant_x)
                b_1_next, b_0_next, error_actual, b1_1, b0_0, pend_1, pend_2,error_actual_x=det_error(data_1,lista_graf_2,b_0_next,b_1_next,b0_0_ant,b1_1_ant,error_ant,error_ant_x,alpha)

                error_actual_abs=abs(error_actual)

                data_2.loc[cont_idex]=[b1_1, b0_0,error_actual_abs,cont_1,pend_1,pend_2]

                cont_idex+=1
                
                dic_epoca_bo_b1[cont_1]=[b0_0,b1_1,error_actual_abs]

                # print("resultados")

                # print(f"  b0_next={b_0_next}")
                # print(f"  b1_next={b_1_next}")
                # print(f"  error_actual={error_actual}")
                # print(f"  b0_ant={b0_0}")
                # print(f"  b1_ant={b1_1}")
                # print(f"  pend_b1={pend_1}")
                # print(f"  pend_b0={pend_2}")
                # print(f"  error_actual_x={error_actual_x}")
                # input()
                # print("..................................................................")

                finalizar=""

                if cont_3==epchs:

                    cont_3=0
 
                    error_actual_abs=abs(error_actual)

                    data_2.loc[cont_idex]=[b1_1, b0_0,error_actual_abs,cont_1,pend_1,pend_2]

                    cont_idex+=1
                    
                    dic_epoca_bo_b1[cont_1]=[b0_0,b1_1,error_actual_abs]

                    data_3 = data_2.sort_values('error',ascending=True)

                    data_3.replace([np.inf, -np.inf], np.nan, inplace=True)

                    data_3=data_3.dropna()

                    val_med=data_3["error"].mean()

                    data_3 = data_3.loc[data_3['error'] >0].drop_duplicates(subset = ['error'])

                    print(data_2)

                    #print(data_menor_media)

                    data_menor_media_snan=data_3.dropna()

                    print()
                    print()
                    print(data_3)

                    #print(data_menor_media_snan)

                    selected_3_rows=data_menor_media_snan[['epoch']].head(n=1)

                    list_epochs_menor_media=set(selected_3_rows['epoch'].values.tolist())

                    keys_dic=set(dic_epoca_bo_b1.keys())

                    keys_dic=list_epochs_menor_media.intersection(keys_dic)

                    keys_dic.add(0)

                    #keys_dic.add(3)
                    
                    fig, ax = plt.subplots()
                    ax.set_title(f'ENTRENAMIENTO DEL MODELO {lista_graf_2[0]} vs {lista_graf_2[1]}')
                    ax.set_xlabel(lista_graf_2[1])
                    ax.set_ylabel(lista_graf_2[0])
                    lista_y=np.array(data_1[lista_graf_2[0]].values.tolist())
                    lista_x=np.array(data_1[lista_graf_2[1]].values.tolist())
                    array_list_1=[1]*(len(lista_x))
                    ax.scatter(lista_x,lista_y)

                    list_keys_dic=list(keys_dic)

                    list_keys_dic.sort(reverse=True)

                    max_epoch=max(list_keys_dic)

                    

                    cont_5=0

                    data_3 = data_3.loc[data_3['epoch'] < max_epoch].drop_duplicates(subset = ['epoch'])


                    array_temp_x_1=np.array(data_1[lista_graf_2[1]].values.tolist())

                    array_temp_y=np.array(data_1[lista_graf_2[0]].values.tolist())

                    array_temp_x_1=array_temp_x_1.reshape(-1,1)
                    array_temp_y=array_temp_y.reshape(-1,1)


                    model=LinearRegression()

                    model.fit(array_temp_x_1,array_temp_y)
                    R_sq = model.score(array_temp_x_1,array_temp_y)

                    train_b1=(model.coef_).tolist()[0][0]
                    train_b0=(model.intercept_).tolist()[0]

                    error_train=det_error_2(data_1,lista_graf_2,train_b0,train_b1)

                    dic_epoca_bo_b1["Train_sklearn"]=[train_b0,train_b1,error_train]

                    list_keys_dic.append("Train_sklearn")


                    for i in set(list_keys_dic):

                        val=""
                        try:

                            val=f"EP = {int(i)}"

                        except:

                            val=f"{i}"

                            pass


                        print("-------------------------------------------------------------")

                        print(f"EPOCA = {i}")
                        print(f"ERROR = {dic_epoca_bo_b1[i][2]}")
                        vect_1=np.array([dic_epoca_bo_b1[i][1]])
                        print(f"VALOR DE B1 ={dic_epoca_bo_b1[i][1]}")
                        vect_2=np.reshape(lista_x,(-1,1))
                        vect_b1x=np.dot(vect_2,vect_1)
                        vect_3=np.array([dic_epoca_bo_b1[i][0]])
                        print(f"VALOR DE B0 ={dic_epoca_bo_b1[i][0]}")
                        vect_4=np.reshape(array_list_1,(-1,1))
                        vect_b0=np.dot(vect_4,vect_3)
                        y = vect_b1x+vect_b0
                        ax.plot(lista_x, y, label = f"{val} | ERROR ={round(dic_epoca_bo_b1[i][2],1)}")
                        plt.tight_layout()
                        ax.legend()

                        cont_5+=1

                        if cont_5==6:

                            break

                    print("-------------------------------------------------------------")




                    plt.show()




                    #data_2=pd.DataFrame(columns=["b1","b0","error","epoch","pend_1","pend_2"])

                    def graficas(data_graf,lista_graf):

                        data_2_reducido=data_graf[lista_graf].dropna()

                        key=f"{lista_graf[0]} vs {lista_graf[1]}"

                        array_data_1=data_2_reducido.values.tolist()

                        array2=np.reshape(array_data_1,-1)

                        lista_y=array2[0:-1:2]

                        lista_x=array2[1::2]

                        last_value_x=max(lista_x)

                        lista_1_columnas_todos = data_graf.loc[data_graf[lista_graf[1]] == last_value_x]

                        last_value_y = lista_1_columnas_todos[lista_graf[0]].tolist()

                        val_1=data_graf[lista_graf].dropna()

                        fig, ax = plt.subplots()         
                        ax.plot(lista_x,lista_y,label = f"{lista_graf[1]} = {last_value_x} | {lista_graf[0]} = {last_value_y[0]} ")
                        plt.title(key, fontsize = 12)
                        plt.xticks(fontsize=9, rotation=45)
                        plt.xlabel(lista_graf[1],fontsize=12)
                        plt.ylabel(lista_graf[0],fontsize=12)
                        plt.tight_layout()
                        ax.legend()
                        plt.show()

                        last_value_y_1=last_value_y[0]

                        last_value_x_1=last_value_x

                        return last_value_y_1, last_value_x_1

                    
                    last_value_b1, last_epoch=graficas(data_3,["b1","epoch"])
                    last_value_b0, last_epoch=graficas(data_3,["b0","epoch"])
                    last_value_error, last_epoch=graficas(data_3,["error","epoch"])



                    finalizar=input("Escriba Y si desea terminar o presione ENTER para repertir la cantidad de epocas = ")



                if finalizar=="Y":

                    break


                #if cont_1>3:

                    # if abs(pent_ant_1)*pend_1==abs(pend_1)*pent_ant_1 or abs(pent_ant_2)*pend_2==abs(pend_2)*pent_ant_2:

                    #     nuevo_registro = {"b1":b1_1,"b0":b0_0,"error":error_actual,"epoch":cont_1,"pend_1":pend_1,"pend_2":pend_2}
     
                    #     data_2 = data_2.append(nuevo_registro, ignore_index=True)

                    #     list_epocs.append(cont_1)

                    #     print(data_2)

                    #     list_bo.append(b0_0)
                    #     list_b1.append(b1_1)


                    #     break


                    # if error_ant>error_actual or abs(error_ant)*error_actual==abs(error_actual)*error_ant:         

                    #     data_2.loc[cont_idex]=[b1_1, b0_0,error_actual,cont_1,pend_1,pend_2] #Añadiendo una fila al dataframe

                    #     cont_idex+=1

                    #     list_epocs.append(cont_1)

                    #     #print(data_2)

                    #     dic_epoca_bo_b1[cont_1]=[b0_0,b1_1]

                        # fig = plt.figure(figsize = (9,6))
                        # ax =  fig.add_axes([0.1,0.1,0.7,0.7])
                        # ax.set_title('ENTRENAMIENTO DEL MODELO')
                        # ax.set_xlabel(lista_graf_2[1])
                        # ax.set_ylabel(lista_graf_2[0])
                        # lista_y=np.array(data_1[lista_graf_2[0]].values.tolist())
                        # lista_x=np.array(data_1[lista_graf_2[1]].values.tolist())
                        # array_list_1=[1]*(len(lista_x))
                        # ax.scatter(lista_x,lista_y)

                        # vect_1=np.array([b1_1])
                        # vect_2=np.reshape(lista_x,(-1,1))
                        # vect_b1x=np.dot(vect_2,vect_1)
                        # vect_3=np.array([b0_0])
                        # vect_4=np.reshape(array_list_1,(-1,1))
                        # vect_b0=np.dot(vect_4,vect_3)
                        # y = vect_b1x+vect_b0
                        # ax.plot(lista_x, y, '-r', label='Y=b1x+b0')

                        # plt.show()

                        #break

                    # if error_actual<=0 or float(pend_1)==float("NaN"):

                    #     nuevo_registro = {"b1":b1_1,"b0":b0_0,"error":error_actual,"epoch":cont_1,"pend_1":pend_1,"pend_2":pend_2}
     
                    #     data_2 = data_2.append(nuevo_registro, ignore_index=True)

                    #     list_epocs.append(cont_1)

                    #     print(data_2)
                    #     list_bo.append(b0_0)
                    #     list_b1.append(b1_1)

                    #     break


            cont_1+=1

            cont_2+=1

            cont_3+=1

            
        # fig = plt.figure(figsize = (9,6))
        # ax =  fig.add_axes([0.1,0.1,0.7,0.7])
        # ax.set_title('ENTRENAMIENTO DEL MODELO')
        # ax.set_xlabel(lista_graf_2[1])
        # ax.set_ylabel(lista_graf_2[0])
        # lista_y=np.array(data_1[lista_graf_2[0]].values.tolist())
        # lista_x=np.array(data_1[lista_graf_2[1]].values.tolist())
        # array_list_1=[1]*(len(lista_x))
        # ax.scatter(lista_x,lista_y)

        # print(lista_y)

        # print(lista_x)

        # for contador, i in enumerate(list_b1, start=0):

        #     vect_1=np.array([i])
        #     vect_2=np.reshape(lista_x,(-1,1))
        #     vect_b1x=np.dot(vect_2,vect_1)
        #     vect_3=np.array([list_bo[contador]])
        #     vect_4=np.reshape(array_list_1,(-1,1))
        #     vect_b0=np.dot(vect_4,vect_3)
        #     y = vect_b1x+vect_b0
        #     ax.plot(lista_x, y, '-r', label='Y=b1x+b0')

        # plt.show()


        print(f"Con un error inicial de alpha={alpha} y con valores de b1={b1} y b0={b0} se necesitaron {len(list_epocs)} epocas")


        #regressor = linear_model.LinearRegression()
        

        dic_result_train={"last_b1":last_value_b1, "last_b0":last_value_b0, "last_error":last_value_error, "last_epchs":last_epoch, "train_b0":train_b0, "train_b1":train_b1,"ERROR_train":error_train}

        return dic_result_train





        input("Enter para continuar")

               


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

#print(data_valores)



######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################

######################################################################################################
################################################ PASO 4 ##############################################

#------------------------------------- inicio imprimir histogramas -------------------------------------

#ingresar el listado de las parejas de las columnas que se desean graficar [[y1,x1],[y2,x2]]

lista_of_columns=["PRECIO","CALIDAD_MATERIAL","AREA_PISO","TOTAL_HABITACIONES","AÑO_CONSTRUCCION","FRENTE"]

#datos_1.histograma_parejas(lista_of_columns)

#------------------------------------- fin imprimir histogramas------------------------------------------


######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################

######################################################################################################
################################################ PASO 5 ##############################################

#Para cada variable independiente x :

#1. Calcular el coeficiente de correlación entre x y y.
#2. Graficar x vs y(scatterplot) usando matplotlib.
#3. Colocar el coeficiente de correlación y colocarlo como parte del título de la gráfica.
#4. Basado en la gráfica y el coeficiente de correlación de cada par x,y elegir las 2 variables
#   con más potencial predictivo es decir las 2 variables que presentan mayor correlación
#   entre dicha variable y la variable dependiente.

lista_graf_2=[["PRECIO","PRECIO"],
    ["PRECIO","CALIDAD_MATERIAL"],
    ["PRECIO","AREA_PISO"],
    ["PRECIO","TOTAL_HABITACIONES"],
    ["PRECIO","AÑO_CONSTRUCCION"],
    ["PRECIO","FRENTE"]] 

#ingresa las listas de las tablas que se compararan y se agrega la cantidad de 
#parejas a analizar, quienes serán las mas cercanas a 1, pero no 1 
#data_valores_corr,data_valores_corr_1=datos_1.factor_correlacion(lista_graf_2,2)

#print("TODOS_LOS_VALORES_CORRELATIVOS")
#print(data_valores_corr)
#print()
#print("VALORES_CORRELATIVOS_SELECCIONADOS")
#print(data_valores_corr_1)

######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################
print("PRIMER_ANALISIS")
lista_graf_2=["PRECIO","CALIDAD_MATERIAL"]
epchs=10000
b1=10000
b0=-500
error_1=500
alpha_1=0.00001

data_entrenamiento=datos_1.entrenamiento_80(lista_graf_2,b1,b0,error_1,epchs,alpha_1)

print(data_entrenamiento)


lista_graf_3=["PRECIO","AREA_PISO"]
print("SEGUNDO_ANALISIS")
epchs_2=3500
b1_2=-0.61
b0_2=1000
error_2=0
alpha_2=0.00001

#data_entrenamiento_2=datos_1.entrenamiento_80(lista_graf_3,b1_2,b0_2,error_2,epchs_2,alpha_2)


lista_graf_4=["PRECIO","TOTAL_HABITACIONES"]
print("TERCER_ANALISIS")
epchs_3=6000
b1_3=10
b0_3=500
error_3=5
alpha_3=0.2

#data_entrenamiento_3=datos_1.entrenamiento_80(lista_graf_4,b1_3,b0_3,error_3,epchs_3,alpha_3)


lista_graf_5=["PRECIO","AÑO_CONSTRUCCION"]
print("CUARTO_ANALISIS")
epchs_4=2000
b1_4=-0.2
b0_4=-500
error_4=5
alpha_4=0.001

data_entrenamiento_4=datos_1.entrenamiento_80(lista_graf_5,b1_4,b0_4,error_4,epchs_4,alpha_4)

lista_graf_6=["PRECIO","FRENTE"]
print("QUINTO_ANALISIS")
epchs_5=2000
b1_5=-0.2
b0_5=-500
error_5=5
alpha_5=0.0001

data_entrenamiento_5=datos_1.entrenamiento_80(lista_graf_6,b1_5,b0_5,error_5,epchs_5,alpha_5)

input()


#matrix_1Reducida=dataframe_elemento.drop(lista_surtida, axis=1)