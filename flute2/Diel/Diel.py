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

# Denoting fitting function
def fit_func(x,a, b):
    return a * x + b

#def findIntersection(fun1,fun2,x0):
#     return fsolve(lambda x :fun1(x) - fun2(x),x0)

##Filechooser
root = tk.Tk()
root.withdraw() # we don't want a full GUI, so keep the root window from appearing
path=pathlib.Path().absolute()
file_names = filedialog.askopenfilenames(initialdir=path,parent=root,title='Choose a file')

fig1=plt.figure()
# canvas1=FigureCanvas(fig1)
#fig1.suptitle('CBKR-Strip')
ax1= fig1.add_subplot()
ax1.set_xlabel('Voltage [V]')
ax1.set_ylabel('Current [A]')


##Loop in each file
for file_name in file_names:
    pos=file_name.find(".txt")
    #pos=file_name.find(".csv")
    if (pos!=-1):
        ##read data
        data=pd.read_csv(file_name, sep="\t",skiprows=4)
        file_name = os.path.splitext(file_name)[0]
        voltage = data.iloc[:,0]
        current = data.iloc[:,1]
        print(data)
        ##make the plot
        label = file_name[0:pos].split('/')[-1]
        print(label)
        label = file_name[0:pos].split('/')[-1]
        label_contents=label.split('_')
        name=label_contents[4]+"_"+label_contents[1]+"_"+label_contents[6]+"_"+label_contents[8]
        print(name)
        ax1.plot(voltage,current,linestyle='solid',marker='o',label=name)
        ax1.title.set_text("Dielectric Breakdown"+" : "+label_contents[2]+"_"+label_contents[3])
        ax1.legend(loc="best", ncol=3, fontsize="small")
        ##fit the data
        #position_voltage_fit1 = int(np.where(voltage == (-8.5e-06))[0]) #position in which the fit starts
        #pars1, cov = optimize.curve_fit(fit_func,voltage[position_voltage_fit1:],voltage[position_voltage_fit1:]) #pars1 is the parameters of the fit and cov is the convolution
        ##Resistivity calculation
        #slope=pars1[0] # takes the slope
        #Rsh=(math.pi/math.log(2))*slope  # Rsh=(pi/ln(2))*slope
        ##Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
        #stdevs = np.sqrt(np.diag(cov))
        #delta_Rsh=(math.pi/math.log(2))*stdevs[0]
       # print(Rsh)
        ##print results in Results.txt
        #print(cov)

        #label2="Rsh={} \u00B1 {} \u03A9/square \n".format(round(Rsh, 2),round(delta_Rsh, 2))
       # ax1.plot(voltage[position_voltage_fit1:],
        #         fit_func(voltage[position_voltage_fit1:], *pars1),
        #         linestyle='--', linewidth=2, color='black',label=label2) # plot also the fit curve
        #ax1.legend(loc="upper left")
        Oxide_Vbd=voltage[len(voltage)-1]
        label = label_contents[3] + '_' + label_contents[4] + '_' + label_contents[2] + '_' + \
                label_contents[1] + '_' + label_contents[6] + '_' + label_contents[8]

        file_exists = os.path.isfile('./Dielectric.csv')
        print("file_exists=",file_exists)
        with open('Dielectric.csv', newline='', mode='a') as csv_file:
            fieldnames = ['Name', 'Oxide_Vbd [V]']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not file_exists:
                print("Write header")
                writer.writeheader()  # file doesn't exist yet, write a header
                writer.writerow({'Name': label, 'Oxide_Vbd [V]': Oxide_Vbd})
            else:
                written_data=pd.read_csv("./Dielectric.csv",sep=",")
                if label not in written_data.values:
                    writer.writerow({'Name': label, 'Oxide_Vbd [V]': Oxide_Vbd})

        #split=label.split("_")
        #serialnumber=split[4]
        #orientation=split[5]
        #print("orientation=",orientation)
        #fig2,ax2.bar(serialnumber+" "+orientation,Caps,1.0)


width = root.winfo_screenwidth()
height = root.winfo_screenheight()
print("width=",width)

fig1.set_size_inches(width/100,height/100)
fig1.savefig('Diel.png',dpi=300)

plt.show()


#f.close()