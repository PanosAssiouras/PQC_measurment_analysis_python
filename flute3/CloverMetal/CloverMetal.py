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
#fig1.suptitle('CloverMetal')
ax1= fig1.add_subplot()
ax1.set_xlabel('Current [A]')
ax1.set_ylabel('Voltage [V]')
#ax1.set_ylim(-0.01,0.01)
#ax1.set_xlim(0,10)

fig2=plt.figure()
#FigureCanvas(fig2)
#fig2.suptitle('CloverMetal')
ax2= fig2.add_subplot()
ax2.set_xlabel('Current [A]')
ax2.set_ylabel('Voltage [V]')
#ax2.set_ylim(-0.025,0.025)
#ax2.set_xlim(0,10)
print(plt.get_backend())

##Loop in each file
label_list=[]
Rsh_s_list=[]
Rsh_r_list=[]
deltaRsh_s_list=[]
deltaRsh_r_list=[]
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
            label = label_contents[3] + '_' + label_contents[4] + '_' + label_contents[2] + '_' + \
                    label_contents[1] + '_' + label_contents[6]
            if label not in label_list:
                label_list.append(label)
            name = label_contents[4] + "_" + label_contents[1] + "_" \
                   + label_contents[6] + "_" + label_contents[8]+ "_" + "standard"
            ax1.plot(current,voltage,linestyle='solid',marker='o',label=name)
            ax1.title.set_text("CloverMetal"+" : "+label_contents[2]+"_"+label_contents[3]+"_"+"standard")
            ##fit the data
            # position_current_fit1 = int(np.where(current == -9.000000e-08)[0]) #position in which the fit starts
            pars1, cov = optimize.curve_fit(fit_func, current[3:],
                                            voltage[3:])  # pars1 is the parameters of the fit and cov is the convolution
            ##Resistivity calculation
            slope = pars1[0]  # takes the slope
            Rsh_s=(math.pi / math.log(2))*slope # Rsh=(pi/ln(2))*slope
            Rsh_s_list.append(Rsh_s)
            ##Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
            stdevs = np.sqrt(np.diag(cov))
            deltaRsh_s=(math.pi / math.log(2)) * stdevs[0]
            deltaRsh_s_list.append(deltaRsh_s)

            #print(Rsh)
            label2 = "Rsh={} \u00B1 {} \u03A9/square \n".format(round(Rsh_s, 4), round(deltaRsh_s, 5))
            ax1.plot(current[3:],
                     fit_func(current[3:], *pars1),
                     linestyle='--', linewidth=2, color='black', label=label2)  # plot also the fit curve
            ax1.legend(loc="best", ncol=3, fontsize="x-small")
        elif label_contents[-1]=="r":
            label = label_contents[3] + '_' + label_contents[4] + '_' + label_contents[2] + '_' + \
                    label_contents[1] + '_' + label_contents[6]
            if label not in label_list:
                label_list.append(label)
            name = label_contents[4] + "_" + label_contents[1] + "_" + \
                   label_contents[6] + "_" + label_contents[8]+ "_" + "rotated"
            print(name)
            ax2.plot(current,voltage,linestyle='solid',marker='o',label=name)
            ax2.title.set_text("CloverMetal"+" : "+label_contents[2]+"_"+label_contents[3]+"_"+"rotated")
            ##fit the data
            # position_current_fit1 = int(np.where(current == -9.000000e-08)[0]) #position in which the fit starts
            pars1, cov = optimize.curve_fit(fit_func, current[3:],
                                            voltage[3:])  # pars1 is the parameters of the fit and cov is the convolution
            ##Resistivity calculation
            slope = pars1[0]  # takes the slope
            Rsh_r=(math.pi / math.log(2))*slope # Rsh=(pi/ln(2))*slope
            Rsh_r_list.append(Rsh_r)
            ##Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
            stdevs = np.sqrt(np.diag(cov))
            deltaRsh_r=(math.pi / math.log(2))*stdevs[0]
            deltaRsh_r_list.append(deltaRsh_r)

            #print(Rsh)
            label2 = "Rsh={} \u00B1 {} \u03A9/square \n".format(round(Rsh_r,4), round(deltaRsh_r, 5))
            ax2.plot(current[3:],
                     fit_func(current[3:], *pars1),
                     linestyle='--', linewidth=2, color='black', label=label2)  # plot also the fit curve
            ax2.legend(loc="best", ncol=3, fontsize="x-small")


file_exists = os.path.isfile('./CloverMetal.csv')
#label = label_list[3] + '_' + label_list[4] + '_' + label_list[2] + '_' + \
#                label_list[1] + '_' + label_list[6] + '_' + label_list[8]

print(len(label_list))
print(len(Rsh_s_list))
print(len(Rsh_r_list))

print("file_exists=", file_exists)
with open('CloverMetal.csv', newline='', mode='a') as csv_file:
    fieldnames = ['Name', 'CloverMetal-standard [Ohm/square]', 'CloverMetal-standard-err [Ohm/square]',
                  'CloverMetal-rotated [Ohm/square]','CloverMetal-rotated-err [Ohm/square]']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not file_exists:
        print("Write header")
        writer.writeheader()  # file doesn't exist yet, write a header
    writer.writerow({'Name': label_list[0], 'CloverMetal-standard [Ohm/square]': Rsh_s_list[0],
                     'CloverMetal-standard-err [Ohm/square]' : deltaRsh_s_list[0],
                     'CloverMetal-rotated [Ohm/square]': Rsh_r_list[0],
                    'CloverMetal-rotated-err [Ohm/square]' : deltaRsh_r_list[0]})
    #written_data = pd.read_csv("./CloverMetal.csv", sep=",")
    for i in range(1,len(label_list)):
        #if label_list[i] not in written_data.values:
        print(label_list[i])
        writer.writerow({'Name': label_list[i], 'CloverMetal-standard [Ohm/square]': Rsh_s_list[i],
                         'CloverMetal-standard-err [Ohm/square]': deltaRsh_s_list[i],
                         'CloverMetal-rotated [Ohm/square]': Rsh_r_list[i],
                         'CloverMetal-rotated-err [Ohm/square]': deltaRsh_r_list[i]})


#manager = plt.figure(1).get_current_fig_manager()
#manager.resize(*manager.window.maxsize())
#plt.figure(1).savefig('CloverMetal-standard.png')
#manager = plt.get_current_fig_manager()
#manager.resize(*manager.window.maxsize())
#fig2.savefig('CloverMetal-rotated.png')
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

fig1.savefig('CloverMetal-standard.png',dpi=500)
fig2.savefig('CloverMetal-rotated.png',dpi=500)

plt.show()
#fig2.savefig('CloverMetal-rotated.png',bbox_inches='tight',pad_inches=0.1)




