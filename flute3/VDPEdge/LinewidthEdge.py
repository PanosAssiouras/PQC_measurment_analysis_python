import math
import os
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfilenames
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.optimize import fsolve
import csv
import pathlib
from pathlib import PurePath

# Denoting fitting function
def fit_func(x,a, b):
    return a * x + b


# function to return key for any value
def get_key(name,data):
    k
    for key, value in my_dict.items():
        if val == value:
            return key

    return "key doesn't exist"

#def findIntersection(fun1,fun2,x0):
#     return fsolve(lambda x :fun1(x) - fun2(x),x0)

##Filechooser
root = tk.Tk()
root.withdraw() # we don't want a full GUI, so keep the root window from appearing
path=pathlib.Path().absolute()
path_par=PurePath(path.parents[1])
file_names = filedialog.askopenfilenames(initialdir=path,parent=root,title='Choose a file')
Rsh_data_file_names = filedialog.askopenfilenames(initialdir=path_par,parent=root,title='Choose a file')

#print(Rsh_data_file_names)
#print(file_names)


data_Rsh=pd.read_csv(Rsh_data_file_names[0], sep=",",skiprows=0)
#print(data_Rsh)


Rsh_measurment_name=data_Rsh.iloc[:,0]
#print(type(Rsh_measurment_name))
Rsh=data_Rsh.iloc[:,1]
delta_Rsh=data_Rsh.iloc[:,2]# Rsh is extracted from VDP poly measurment from flute1
dRsh=6.42
W = 33; # um from geomrtry
d = 13; # um from geometry


########Plot labels and title ####################################

fig1=plt.figure()
# canvas1=FigureCanvas(fig1)
#fig1.suptitle('LinewidthEdge')
ax1= fig1.add_subplot()
ax1.set_xlabel('Current [A]')
ax1.set_ylabel('Voltage [V]')
#ax1.set_ylim(-0.01,0.01)
#ax1.set_xlim(0,10)

fig2=plt.figure()
#FigureCanvas(fig2)
#fig2.suptitle('LinewidthEdge')
ax2= fig2.add_subplot()
ax2.set_xlabel('Current [A]')
ax2.set_ylabel('Voltage [V]')
#ax2.set_ylim(-0.025,0.025)
#ax2.set_xlim(0,10)
#print(plt.get_backend())

t_s_list = []
t_r_list = []
label_list=[]
##Loop in each file
for file_name in file_names:
    pos=file_name.find(".txt")
    #pos=file_name.find(".csv")
    if (pos!=-1):
        ##read data
        data=pd.read_csv(file_name, sep="\t",skiprows=4)
        file_name = os.path.splitext(file_name)[0]
        current = data.iloc[:,0]
        voltage = data.iloc[:,1]
        #print(data)
        ##extract important information from file names
        label=file_name[0:pos].split('/')[-1]
        label_list.append(label)
        label_contents = label.split('_')
#        print(key_list[val_list.index(label)])
        label_check=label_contents[0]+"_"+label_contents[1]\
                    +"_"+label_contents[2]+"_"+label_contents[3]\
                    +"_"+label_contents[4]+"_"+label_contents[5]\
                    +"_"+label_contents[6]
        #print("label_check=", label_check)

        for measurement in Rsh_measurment_name:
            #print(measurement)
            #print(label_check)
            if(label_check in measurement):
                #print("checked")
                position=Rsh_measurment_name[Rsh_measurment_name == measurement].index[0]
                #print("position=",position)
                #print(Rsh[int(position)])
                if "_s" in measurement:
                    #print(measurement)
                    name = label_contents[4] + "_" + label_contents[1] + "_" \
                           + label_contents[6] + "_" + label_contents[8] + "_"+"s"
                    ax1.plot(current, voltage, linestyle='solid', marker='o', label=name)
                    ax1.title.set_text(
                        "LinewidthEdge" + " : " + label_contents[2] + "_" + label_contents[3] + "_" + "standard")
                    ##fit the data
                   # position in which the fit starts
                    pars1, cov = optimize.curve_fit(fit_func, current[3:], voltage[3:])  # pars1 is the parameters of the fit and cov is the convolution
                    ax1.plot(current[3:],
                             fit_func(current[3:], *pars1),
                             linestyle='--', linewidth=2, color='black')  # plot also the fit curve
                    ##Resistivity calculation
                    slope = pars1[0]  # takes the slope
                    stdevs = np.sqrt(np.diag(cov))
                    dslope = stdevs[0]
                    # Rsh=(math.pi/math.log(2))*slope  # Rsh=(pi/ln(2))*slope
                    ##Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
                    # delta_Rsh=(math.pi/math.log(2))*stdevs[0]
                    t_s = (1 / slope) * Rsh[int(position)] * 128.5 * 1E-6
                    t_s_list.append(t_s)
                    deltat=math.sqrt(pow(dslope,2)+pow(dRsh,2))
                    print(t_s)
                    ax1.legend(loc="best", ncol=3, fontsize="small")
                elif "_r" in measurement:
                    name = label_contents[4] + "_" + label_contents[1] + "_" + \
                           label_contents[6] + "_" + label_contents[8] + "_"+"r"
                    #print(name)
                    ax2.plot(current, voltage, linestyle='solid', marker='o', label=name)
                    ax2.title.set_text(
                        "LinewidthEdge" + " : " + label_contents[2] + "_" + label_contents[3] + "_" + "rotated")
                    # position in which the fit starts
                    pars1, cov = optimize.curve_fit(fit_func, current[3:], voltage[3:])  # pars1 is the parameters of the fit and cov is the convolution
                    ax2.plot(current[3:],
                             fit_func(current[3:], *pars1),
                             linestyle='--', linewidth=2, color='black')  # plot also the fit curve
                    ##Resistivity calculation
                    slope = pars1[0]  # takes the slope
                    stdevs = np.sqrt(np.diag(cov))
                    dslope = stdevs[0]
                    # Rsh=(math.pi/math.log(2))*slope  # Rsh=(pi/ln(2))*slope
                    ##Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
                    # delta_Rsh=(math.pi/math.log(2))*stdevs[0]
                    t_r = (1 / slope) * Rsh[int(position)] * 128.5 * 1E-6
                    t_r_list.append(t_r)
                    # deltat=math.sqrt(pow(dslope,2)+pow(dRgeom,2))
                    print(t_r)
                    ax2.legend(loc="best", ncol=3, fontsize="small")


file_exists = os.path.isfile('./LinewidthEdge.csv')
print("file_exists=", file_exists)
with open('LinewidthEdge.csv', newline='', mode='a') as csv_file:
    fieldnames = ['Name', 'linewidth_Edge-standard', 'linewidth_Edge-rotated']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not file_exists:
        print("Write header")
        writer.writeheader()  # file doesn't exist yet, write a header
    written_data = pd.read_csv("./LinewidthEdge.csv", sep=",")
    writer.writerow({'Name': label_list[0], 'linewidth_Edge-standard': t_s_list[0],
                     'linewidth_Edge-rotated': t_r_list[0]})
    for i in range(1,len(label_list)-1):
        if label_list[i] not in written_data.values:
            print(label_list[i])
            writer.writerow({'Name': label_list[i], 'linewidth_Edge-standard': t_s_list[i],
                            'linewidth_Edge-rotated': t_r_list[i]})
            





width = root.winfo_screenwidth()
height = root.winfo_screenheight()
print("width=",width)

fig1.set_size_inches(width/100,height/100)
fig2.set_size_inches(width/100,height/100)

fig1.savefig('LinewidthEdge-standard.png',dpi=300)
fig2.savefig('LinewidthEdge-rotated.png',dpi=300)

plt.show()