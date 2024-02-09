import os
import numpy as np
import pylab
import matplotlib.pyplot as plt
import math
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
from scipy.interpolate import CubicSpline
import csv
import pathlib



# Denoting fitting function
def fit_func(x, a, b):
    return a * x + b

##Filechooser
root = tk.Tk()
root.withdraw() # we don't want a full GUI, so keep the root window from appearing
path=pathlib.Path().absolute()
file_names = filedialog.askopenfilenames(initialdir=path,parent=root,title='Choose a file')


fig1=plt.figure()
# canvas1=FigureCanvas(fig1)
#fig1.suptitle('CBKR-Strip')
ax1= fig1.add_subplot()

##Loop in each file
for file_name in file_names:
    pos=file_name.find(".txt")
    if (pos!=-1):
        ##Read Data
        label = file_name[0:pos].split('/')[-1]
        print(label)
        data=pd.read_csv(file_name, sep="\t",skiprows=4)
        file_name = os.path.splitext(file_name)[0]
        current = data.iloc[:,1]
        voltage = data.iloc[:,0]
        ##CubicSpline
        cs=CubicSpline(voltage,current)
        ##calculate derivatives
        diff__volt=voltage[:-1]
        diff_current=()
        #diff_current=np.diff(current,1)/np.diff(voltage,1)
        diff_current = np.diff(cs(voltage), 1) / np.diff(voltage, 1)
        print(diff_current)
        for i in range(len(diff_current)):
            #print(diff_current[i])
            if (diff_current[i]<0.0):
                diff_current[i]=0

        label = file_name[0:pos].split('/')[-1]
        label_contents=label.split('_')
        name=label_contents[4]+"_"+label_contents[1]+"_"+label_contents[6]+"_"+label_contents[8]
        ax1.title.set_text("FET"+" : "+label_contents[2]+"_"+label_contents[3])
        #plot data
        ax1.plot(voltage, current, linestyle='solid', marker='o', label=name)

        ##plot derivatives
        ax1.plot(diff__volt, diff_current, label="Transconductance", linestyle='dashed', marker='+') # transcoductance plot (first derivative)
        max_der=max(diff_current)                                   #calculate maximum of first derivative
        position_max_der=np.where(diff_current==max(diff_current)) #find index of maximum first derivative
        voltage_max_der=voltage[int(position_max_der[0])]          #find the correspoding Voltage value at maximum first derivative
        current_max_der = current[int(position_max_der[0])]        #find maximum Current at the maximum first derivative
        b=current_max_der-max_der*voltage_max_der                  #b=y-ax calculate the y intecept of the fitting curve
        Threshold_Voltage=-b/max_der+(100E-03/2.0)                 # Threshold voltage= (Intercept value at x-axis)-((Drain-Source Voltage)/2.0)
        position_voltage_at4 = np.where(voltage == 4.0)            #index at 4.0V
        position_voltage_at6 = np.where(voltage == 6.0)            #index at 6.0V
        ax1.plot(voltage[int(position_voltage_at4[0]):int(position_voltage_at6[0])],
                   fit_func(voltage[int(position_voltage_at4[0]):int(position_voltage_at6[0])],max_der,b)
                   ,label="Fit at maximum slope, ThresholdVoltage={}".format(round(Threshold_Voltage, 2)),
                   color='red',linestyle='dashed',marker='*')       #plot fit curve from Voltage=4.0 to Voltage=6.0

        print("The maximum first derivative is {} uA with index {} at {} V".format(max_der,
                                                                                   int(position_max_der[0]),
                                                                                   voltage_max_der))
        ax1.title.set_text("FET"+" : "+label_contents[2]+"_"+label_contents[3])
        print("Threshold Voltage is {}".format(Threshold_Voltage))

######## Write results to FET.csv ####################################

        label = label_contents[3] + '_' + label_contents[4] + '_' + label_contents[2] + '_' + \
                label_contents[1] + '_' + label_contents[6]

        file_exists = os.path.isfile('./FET.csv')
        print("file_exists=",file_exists)
        with open('FET.csv', newline='', mode='a') as csv_file:
            fieldnames = ['Name', 'V_th [V]']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not file_exists:
                print("Write header")
                writer.writeheader()  # file doesn't exist yet, write a header
                writer.writerow({'Name': label, 'V_th [V]': round(Threshold_Voltage, 2)})
            else:
                written_data=pd.read_csv("./FET.csv",sep=",")
                if label not in written_data.values:
                    writer.writerow({'Name': label, 'V_th [V]': round(Threshold_Voltage,2)})

#pylab.legend()      #plots the lendend for all the plots
ax1.set_xlabel("Voltage [V]")
ax1.set_ylabel("Current [A]")
ax1.legend(loc="best", ncol=3, fontsize="x-small")

width = root.winfo_screenwidth()
height = root.winfo_screenheight()
print("width=",width)

fig1.set_size_inches(width/100,height/100)
fig1.savefig('FET.png',dpi=300)

plt.show()
