import math
import os
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfilenames
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import itertools as IT

import numpy as np
import pandas as pd
from scipy import optimize
import csv
import pathlib
from scipy.optimize import fsolve
from pathlib import PurePath

# Denoting fitting function
def fit_func(x,a, b):
    return a * x + b

#def findIntersection(fun1,fun2,x0):
#     return fsolve(lambda x :fun1(x) - fun2(x),x0)


def getIndexes(dfObj, value):
    ''' Get index positions of value in dataframe i.e. dfObj.'''
    listOfPos = list()
    # Get bool dataframe with True at positions where the given value exists
    result = dfObj.isin([value])
    # Get list of columns that contains the value
    seriesObj = result.any()
    columnNames = list(seriesObj[seriesObj == True].index)
    # Iterate over list of columns and fetch the rows indexes where value exists
    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
        for row in rows:
            listOfPos.append((row, col))
    # Return a list of tuples indicating the positions of value in the dataframe
    return listOfPos

##Filechooser
root = tk.Tk()
root.withdraw() # we don't want a full GUI, so keep the root window from appearing
path=pathlib.Path().absolute()
path_par=PurePath(path.parents[2], 'flute1/VDPStrip')
file_names = filedialog.askopenfilenames(initialdir=path,parent=root,title='Choose a file')
Rsh_data_file_names = filedialog.askopenfilenames(initialdir=path_par,parent=root,title='Choo   se a file')

data_VDPStrip=pd.read_csv(Rsh_data_file_names[0], sep=",")
VDP_name_r=[]
VDP_name_s=[]
Rsh_s=[]
Rsh_r=[]
delta_Rsh_s=[]
delta_Rsh_r=[]
for index, row in data_VDPStrip.iterrows():
    #print(measurment)
    #name=data_VDPStrip.iloc[index:,0]
    name=row[0]
    if "_r" in name:
        #print(measurment)
        Rsh_r.append(row[1])
        delta_Rsh_r.append(row[2])
        VDP_name_r.append(name)
    elif "_s" in name:
        Rsh_s.append(row[1])
        delta_Rsh_s.append(row[2])
        VDP_name_s.append(name)

d = {'VDP_name_r': VDP_name_r, 'Rsh_r': Rsh_r,'delta_Rsh_r':delta_Rsh_r }
data_VDPStrip_r=pd.DataFrame(data=d)
d = {'VDP_name_s': VDP_name_s, 'Rsh_s': Rsh_s,'delta_Rsh_s':delta_Rsh_s }
data_VDPStrip_s = pd.DataFrame(data=d)

print(data_VDPStrip_r)
print(data_VDPStrip_s)
Results = pd.DataFrame(columns = ['Name'])
#Results = pd.DataFrame(columns = ['Name','Rc_s','deltaRc_s','Rc_r','deltaRc_r'])

#print(data_VDPStrip_r)
#print(data_VDPStrip_s)

#Rsh=2290 # Rsh is extracted from VDP Strip measurment from flute1
#dRsh=0.57 # calculated error of Rsh
W = 33; # um from geomrtry
d = 13; # um from geometry

########Plot labels and title ####################################

fig1=plt.figure()
# canvas1=FigureCanvas(fig1)
#fig1.suptitle('CBKR-Strip')
ax1= fig1.add_subplot()
ax1.set_xlabel('Current [A]')
ax1.set_ylabel('Voltage [V]')
#ax1.set_ylim(-0.05,0.05)
#ax1.set_xlim(0,10)

fig2=plt.figure()
#FigureCanvas(fig2)
#fig2.suptitle('CBKR-Strip')
ax2= fig2.add_subplot()
ax2.set_xlabel('Current [A]')
ax2.set_ylabel('Voltage [V]')
#ax2.set_ylim(-0.05,0.05)
#ax2.set_xlim(0,10)
print(plt.get_backend())

label_list=[]
Rc_s_list=[]
Rc_r_list=[]
deltaRc_s_list=[]
deltaRc_r_list=[]
##Loop in each file
for file_name in file_names:
    pos=file_name.find(".txt")
    #pos=file_name.find(".csv")
    if (pos!=-1):
        ##read data
        data=pd.read_csv(file_name, sep="\t",skiprows=4)
        current=data.iloc[:,0]
        voltage = data.iloc[:,1]
        file_name = os.path.splitext(file_name)[0]
        ##extract important information from file names
        label = file_name[0:pos].split('/')[-1]
        label_contents = label.split('_')
        label_check = label_contents[3] + "_" + label_contents[4] \
                      + "_" + label_contents[2] + "_" + label_contents[1] \
                      + "_" + label_contents[6]
        if label_contents[-1] == "r":
            for measurment in data_VDPStrip_r['VDP_name_r']:
                if label_check in measurment:
                    if label not in label_list:
                        label_list.append(label)
                    listOfPositions = getIndexes(data_VDPStrip_r,measurment)
                    #print(type(listOfPositions))
                    flat_1 = [x for l in listOfPositions for x in l]
                    #print(flat_1[0])
                    #print(flat_1[1])
                    Rsh_r=data_VDPStrip_r.at[flat_1[0],'Rsh_r']
                    dRsh_r=data_VDPStrip_r.at[flat_1[0],'delta_Rsh_r']
                    name = label_contents[4] + "_" + label_contents[1] + "_" \
                           + label_contents[6] + "_" + label_contents[8] + "_" + "rotated"
                    ax2.plot(current, voltage, linestyle='solid', marker='o', label=name)
                    ax2.title.set_text(
                        "CBKR-Strip" + " : " + label_contents[2] + "_" + label_contents[3] + "_" + "rotated")
                    ##fit the data
                    # position_current_fit1 = int(np.where(current == -9.000000e-08)[0]) #position in which the fit starts
                    pars1, cov = optimize.curve_fit(fit_func, current[3:],
                                                    voltage[
                                                    3:])  # pars1 is the parameters of the fit and cov is the convolution
                    ##Resistivity calculation
                    slope = pars1[0]  # takes the slope
                    stdevs = np.sqrt(np.diag(cov))
                    dslope = stdevs[0]
                    Rgeom = (4 * Rsh_r* pow(d, 2) / (3 * pow(W, 2))) * (1 + d / (2 * (W - d)))
                    dRgeom = (4 * dRsh_r* pow(d, 2) / (3 * pow(W, 2))) * (1 + d / (2 * (W - d)))
                    Rc_s = slope - Rgeom
                    deltaRc_s = math.sqrt(pow(dslope, 2) + pow(dRgeom, 2))

                    print(label)
                    #label_list.append(label)
                    Rc_s_list.append(Rc_s)
                    deltaRc_s_list.append(deltaRc_s)

                    ##plot fit results
                    label2 = "Rc={} \u00B1 {} \u03A9/square \n".format(round(Rc_s, 2), round(deltaRc_s, 2))
                    ax2.plot(current[3:],
                             fit_func(current[3:], *pars1),
                             linestyle='--', linewidth=2, color='black', label=label2)  # plot also the fit curve
                    ax2.legend(loc="best", ncol=5, fontsize="x-small")
        if label_contents[-1] == "s":
            for measurment in data_VDPStrip_s['VDP_name_s']:
                if label_check in measurment:
                    if label not in label_list:
                        label_list.append(label)
                    listOfPositions = getIndexes(data_VDPStrip_s,measurment)
                    #print(type(listOfPositions))
                    flat_1 = [x for l in listOfPositions for x in l]
                    #print(flat_1[0])
                    #print(flat_1[1])
                    Rsh_s=data_VDPStrip_s.at[flat_1[0],'Rsh_s']
                    dRsh_s=data_VDPStrip_s.at[flat_1[0],'delta_Rsh_s']
                    name = label_contents[4] + "_" + label_contents[1] + "_" \
                           + label_contents[6] + "_" + label_contents[8] + "_" + "standard"
                    ax1.plot(current, voltage, linestyle='solid', marker='o', label=name)
                    ax1.title.set_text(
                        "CBKR-Strip" + " : " + label_contents[2] + "_" + label_contents[3] + "_" + "standard")
                    ##fit the data
                    # position_current_fit1 = int(np.where(current == -9.000000e-08)[0]) #position in which the fit starts
                    pars1, cov = optimize.curve_fit(fit_func, current[3:],
                                                    voltage[
                                                    3:])  # pars1 is the parameters of the fit and cov is the convolution
                    ##Resistivity calculation
                    slope = pars1[0]  # takes the slope
                    stdevs = np.sqrt(np.diag(cov))
                    dslope = stdevs[0]
                    Rgeom = (4 * Rsh_s* pow(d, 2) / (3 * pow(W, 2))) * (1 + d / (2 * (W - d)))
                    dRgeom = (4 * dRsh_s* pow(d, 2) / (3 * pow(W, 2))) * (1 + d / (2 * (W - d)))
                    Rc_s=slope - Rgeom
                    deltaRc_s=math.sqrt(pow(dslope, 2) + pow(dRgeom, 2))

                    #label_list.append(label)
                    #label_list.append(label)
                    #print(type(label))
                    #label_list.append(label)
                    Rc_s_list.append(Rc_s)
                    deltaRc_s_list.append(deltaRc_s)

                    ##plot fit results
                    label2 = "Rc={} \u00B1 {} \u03A9/square \n".format(round(Rc_s, 2), round(deltaRc_s, 2))
                    ax1.plot(current[3:],
                             fit_func(current[3:], *pars1),
                             linestyle='--', linewidth=2, color='black', label=label2)  # plot also the fit curve
                    ax1.legend(loc="best", ncol=5, fontsize="x-small")
                    ######## Write results to CBKRStrip.csv ####################################

        label =" "

######## Write results to VDPStrip.csv ####################################
print(label_list)
print(len(label_list))
print(len(Rc_s_list))
Results['Name']=label_list
Results['Rc_s [Omega/square]']=Rc_s_list
Results['deltaRc_s [Omega/square]'] = deltaRc_s_list
#Results['Rc_r']=Rc_r_list
#Results['deltaRc_r'] = deltaRc_r_list

print("Results=",Results)
Results.to_csv(r'CBKRStrip.csv', index = False)

width = root.winfo_screenwidth()
height = root.winfo_screenheight()
print("width=",width)

fig1.set_size_inches(width/100,height/100)
fig2.set_size_inches(width/100,height/100)

fig1.savefig('CBKRStrip-standard.png',dpi=300)
fig2.savefig('CBKRStrip-rotated.png',dpi=300)

plt.show()







