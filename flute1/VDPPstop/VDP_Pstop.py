import math
import os
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfilenames
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy import optimize
import csv
import pathlib
from scipy.optimize import fsolve

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


########Plot labels and title ####################################

fig1=plt.figure()
# canvas1=FigureCanvas(fig1)
#fig1.suptitle('VDPPstop')
ax1= fig1.add_subplot()
ax1.set_xlabel('Current [A]')
ax1.set_ylabel('Voltage [V]')
ax1.set_ylim(-0.01,0.01)
#ax1.set_xlim(0,10)

fig2=plt.figure()
#FigureCanvas(fig2)
#fig2.suptitle('VDPPstop')
ax2= fig2.add_subplot()
ax2.set_xlabel('Current [A]')
ax2.set_ylabel('Voltage [V]')
ax2.set_ylim(-0.025,0.025)
#ax2.set_xlim(0,10)
print(plt.get_backend())

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
        print(data)
        ##extract important information from file names
        label = file_name[0:pos].split('/')[-1]
        label_contents = label.split('_')
        ##if it is standard meseurment
        if label_contents[-1]=="s":
            name = label_contents[4] + "_" + label_contents[1] + "_" \
                   + label_contents[6] + "_" + label_contents[8]+ "_" + "standard"
            ax1.plot(current,voltage,linestyle='solid',marker='o',label=name)
            ax1.title.set_text("VDPPstop"+" : "+label_contents[2]+"_"+label_contents[3]+"_"+"standard")
            ##fit the data
            # position_current_fit1 = int(np.where(current == -9.000000e-08)[0]) #position in which the fit starts
            pars1, cov = optimize.curve_fit(fit_func, current[3:],
                                            voltage[3:])  # pars1 is the parameters of the fit and cov is the convolution
            ##Resistivity calculation
            slope = pars1[0]  # takes the slope
            Rsh = (math.pi / math.log(2)) * slope  # Rsh=(pi/ln(2))*slope
            ##Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
            stdevs = np.sqrt(np.diag(cov))
            deltaRsh = (math.pi / math.log(2)) * stdevs[0]
            print(Rsh)
            label2 = "Rsh={} \u00B1 {} \u03A9/square \n".format(round(Rsh, 2), round(deltaRsh, 2))
            ax1.plot(current[3:],
                     fit_func(current[3:], *pars1),
                     linestyle='--', linewidth=2, color='black', label=label2)  # plot also the fit curve
            ax1.legend(loc="best", ncol=3, fontsize="small")
        elif label_contents[-1]=="r":
            name = label_contents[4] + "_" + label_contents[1] + "_" + \
                   label_contents[6] + "_" + label_contents[8]+ "_" + "rotated"
            print(name)
            ax2.plot(current,voltage,linestyle='solid',marker='o',label=name)
            ax2.title.set_text("VDPPstop"+" : "+label_contents[2]+"_"+label_contents[3]+"_"+"rotated")
            ##fit the data
            # position_current_fit1 = int(np.where(current == -9.000000e-08)[0]) #position in which the fit starts
            pars1, cov = optimize.curve_fit(fit_func, current[3:],
                                            voltage[3:])  # pars1 is the parameters of the fit and cov is the convolution
            ##Resistivity calculation
            slope = pars1[0]  # takes the slope
            Rsh = (math.pi / math.log(2)) * slope  # Rsh=(pi/ln(2))*slope
            ##Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
            stdevs = np.sqrt(np.diag(cov))
            deltaRsh = (math.pi / math.log(2)) * stdevs[0]
            print(Rsh)
            label2 = "Rsh={} \u00B1 {} \u03A9/square \n".format(round(Rsh, 2), round(deltaRsh, 2))
            ax2.plot(current[3:],
                     fit_func(current[3:], *pars1),
                     linestyle='--', linewidth=2, color='black', label=label2)  # plot also the fit curve
            ax2.legend(loc="best", ncol=3, fontsize="small")




######## Write results to VDPPstop.csv ####################################

        label = label_contents[3] + '_' + label_contents[4] + '_' + label_contents[2] + '_' + \
                label_contents[1] + '_' + label_contents[6]+ '_' + label_contents[9]

        file_exists = os.path.isfile('./VDPPstop.csv')
        print("file_exists=",file_exists)
        with open('VDPPstop.csv', newline='', mode='a') as csv_file:
            fieldnames = ['Name', 'R_sh [Ohm/square]', 'delta_Rsh [Ohm/square]']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not file_exists:
                print("Write header")
                writer.writeheader()  # file doesn't exist yet, write a header
                writer.writerow({'Name': label, 'R_sh [Ohm/square]': Rsh, 'delta_Rsh [Ohm/square]': deltaRsh})
            else:
                written_data=pd.read_csv("./VDPPstop.csv",sep=",")
                if label not in written_data.values:
                    writer.writerow({'Name': label, 'R_sh [Ohm/square]': Rsh, 'delta_Rsh [Ohm/square]': deltaRsh})

#manager = plt.figure(1).get_current_fig_manager()
#manager.resize(*manager.window.maxsize())
#plt.figure(1).savefig('VDPPstop-standard.png')
#manager = plt.get_current_fig_manager()
#manager.resize(*manager.window.maxsize())
#fig2.savefig('VDPPstop-rotated.png')
#plt.show()


#plt.figure(figsize=(10,10), dpi=

#manager=fig1.canvas.manager
#canvas1.resize(*manager.window.maxsize())
#<matplotlib.backends.backend_tkagg.FigureManagerTkAgg instance at 0x1c3e170>
#manager = plt.get_current_fig_manager()
#manager=fig2.canvas.manager
#manager.resize(*manager.window.maxsize())

width = root.winfo_screenwidth()
height = root.winfo_screenheight()
print("width=",width)

fig1.set_size_inches(width/100,height/100)
fig2.set_size_inches(width/100,height/100)

fig1.savefig('VDPPstop-standard.png',dpi=300)
fig2.savefig('VDPPstop-rotated.png',dpi=300)

plt.show()
#fig2.savefig('VDPPstop-rotated.png',bbox_inches='tight',pad_inches=0.1)




