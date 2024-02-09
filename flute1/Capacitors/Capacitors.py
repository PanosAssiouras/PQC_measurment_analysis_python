import math
import os
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfilenames
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from decimal import Decimal
from scipy.optimize import fsolve
import csv
import pathlib

# Denoting fitting function
def fit_func(x,a, b):
    return a * x + b

#def findIntersection(fun1,fun2,x0):
#     return fsolve(lambda x :fun1(x) - fun2(x),x0)

fig1, ax1 = plt.subplots()


##Filechooser
root = tk.Tk()
root.withdraw() # we don't want a full GUI, so keep the root window from appearing
path=pathlib.Path().absolute()
file_names = filedialog.askopenfilenames(initialdir=path,parent=root,title='Choose a file')

##Loop in each file
for file_name in file_names:
    pos=file_name.find(".txt")
    #pos=file_name.find(".csv")
    if (pos!=-1):
        ##read data
        data=pd.read_csv(file_name, sep="\t",skiprows=4)
        file_name = os.path.splitext(file_name)[0]
        voltage = data.iloc[:,0]
        capacitance = data.iloc[:,1]
        print(data)
        ##make the plot
        label = file_name[0:pos].split('/')[-1]
        print(label)
        label_contents=label.split('_')
        name=label_contents[4]+"_"+label_contents[1]+"_"+label_contents[6]+"_"+label_contents[8]
        print(name)
        ax1.plot(voltage,capacitance,linestyle='solid',marker='o',label=name)
        #ax1.title("Capacitor measurment")
        ax1.set_xlabel('Voltage [V]')
        ax1.set_ylabel('Capacitance [F]')
        ax1.title.set_text("Capacitors"+" : "+label_contents[2]+"_"+label_contents[3])
        ax1.legend(loc="best", ncol=6, fontsize="xx-small")
        ax1.set_ylim(2.0E-12, 2.8E-12)
        ##fit the data
        pars1, cov = optimize.curve_fit(fit_func,voltage,capacitance) #pars1 is the parameters of the fit and cov the covariance
        ##Capacitance calculation
        Caps=pars1[1]
        ##Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
        stdevs = np.sqrt(np.diag(cov))
        delta_Caps=stdevs[0]
        print(Caps)
        print(cov)
        ##print results in Caps.txt
        label2="{}={} \u00B1 {} F \n".format(label_contents[8],'%.2E' % Caps,'%.2E' % delta_Caps,2)
        fig1,ax1.plot(voltage,
                 fit_func(capacitance, *pars1),
                 linestyle='--', linewidth=2, color='black',label=label2) # plot also the fit curve

######## Write results to Caps.csv ####################################
        label=label_contents[3]+'_'+label_contents[4]+'_'+label_contents[2]+'_'+\
             label_contents[1]+'_'+label_contents[6]+'_'+label_contents[8]

        file_exists = os.path.isfile('./Caps.csv')
        print("file_exists=",file_exists)
        with open('Caps.csv', newline='', mode='a') as csv_file:
            fieldnames = ['Name', 'Caps [F]', 'delta_Caps [F]']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not file_exists:
                print("Write header")
                writer.writeheader()  # file doesn't exist yet, write a header
                writer.writerow({'Name': label, 'Caps [F]': Caps, 'delta_Caps [F]': delta_Caps})
            else:
                written_data=pd.read_csv("./Caps.csv",sep=",")
                if label not in written_data.values:
                    writer.writerow({'Name': label, 'Caps [F]': Caps, 'delta_Caps [F]': delta_Caps})

width = root.winfo_screenwidth()
height = root.winfo_screenheight()
print("width=",width)

fig1.set_size_inches(width/100,height/100)
fig1.savefig('Caps.png',dpi=300)

plt.show()

#f.close()