import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from scipy.optimize import curve_fit
from scipy import optimize
from scipy.optimize import fsolve
from decimal import Decimal
import operator
import tkinter as tk
from tkinter import filedialog
from scipy import stats
import csv
import pathlib

# Denoting fitting function
def fit_func1(x,a,b):
    return  a*x + b

def fit_func0(x,b):
    return  0*x + b

#find intersection points from two given functions
def findIntersection(fun1,fun2,x0):
     return fsolve(lambda x :fun1(x) - fun2(x),x0)

# calculates r-squared
def rsquared(f,popt,xdata,ydata):
    residuals = ydata - f(xdata, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
    r_squared =1-ss_res / ss_tot
    return r_squared

def optimizefit_horizontal_curve_with_zero_slope(xdata,ydata,start_posfit1,start_posfit2):
    #optimize fit curve
    #first an initial position of the fit positions is determined to a fixed amount of data pionts
    #The number of data points is increased iteratively and for each interation a new fit is calculated
    #Those values with the maximym r-squared are kept to the final outpout
    position_voltage_horizontal_fit1 =int(np.where(xdata == (start_posfit2-100.0))[0])
    position_voltage_horizontal_fit2 = int(np.where(xdata == (start_posfit2))[0])
    r_squared={}
    intercepts={}
#    print(int(np.where(xdata == start_posfit1)[0]))
    while (xdata[position_voltage_horizontal_fit1] > start_posfit1):
            pars1, cov1 = optimize.curve_fit(fit_func0, xdata[int(position_voltage_horizontal_fit1):int(position_voltage_horizontal_fit2)],
                                             ydata[int(position_voltage_horizontal_fit1):int(position_voltage_horizontal_fit2)])
            r = rsquared(fit_func0, pars1, xdata[int(position_voltage_horizontal_fit1):int(position_voltage_horizontal_fit2)],
                                 ydata[int(position_voltage_horizontal_fit1):int(position_voltage_horizontal_fit2)])
            r_squared[int(position_voltage_horizontal_fit1)]=r
            intercepts[position_voltage_horizontal_fit1]=pars1[0]
            position_voltage_horizontal_fit1-=10
            print(r_squared)
    position_at_maximum_rsquared=max(r_squared.items(), key=operator.itemgetter(1))[0]
    print(position_at_maximum_rsquared)
    return intercepts[position_at_maximum_rsquared], position_at_maximum_rsquared , position_voltage_horizontal_fit2

def optimizefit_incline_curve(xdata,ydata,start_posfit1,start_posfit2):
    #optimize fit curve
    #first an initial position of the fit positions is determined to a fixed amount of data pionts
    #The number of data points is increased iteratively and for each interation a new fit is calculated
    #Those values with the maximym r-squared are kept to the final outpout
    position_voltage_incline_fit1 = int(np.where(voltage == start_posfit1)[0])
    position_voltage_incline_fit2 = int(np.where(voltage == start_posfit2)[0])
    r_squared={}
    while (xdata[int(position_voltage_incline_fit2)] < 500):
        slope, intercept, r, p, std_err = stats.linregress(
            xdata[int(position_voltage_incline_fit1):int(position_voltage_incline_fit2)],
            ydata[int(position_voltage_incline_fit1):int(position_voltage_incline_fit2)])
        r_squared[position_voltage_incline_fit2]=r**2
        position_voltage_incline_fit2+=1
    max_rsquared = max(r_squared.values())
    position_at_maximum=max(r_squared.items(), key=operator.itemgetter(1))[0]
    return max_rsquared, position_at_maximum

def optimizefit_horizontal_curve(xdata,ydata,start_posfit1,start_posfit2):
    #optimize fit curve
    #first an initial position of the fit positions is determined to a fixed amount of data pionts
    #The number of data points is increased iteratively and for each interation a new fit is calculated
    #Those values with the maximym r-squared are kept to the final outpout
    position_voltage_horizontal_fit1 = int(np.where(voltage == start_posfit1)[0])
    position_voltage_horizontal_fit2 = int(np.where(voltage == start_posfit2)[0])
    r_squared={}
    while (xdata[position_voltage_horizontal_fit1]>250):
        slope, intercept, r, p, std_err = stats.linregress(xdata[position_voltage_horizontal_fit1:position_voltage_horizontal_fit2],
                                                           ydata[position_voltage_horizontal_fit1:position_voltage_horizontal_fit2])
        r_squared[position_voltage_horizontal_fit1]=r**2
        position_voltage_horizontal_fit1 -= 1
    print(r_squared)
    max_rsquared = max(r_squared.values())
    position_at_maximum=max(r_squared.items(), key=operator.itemgetter(1))[0]
    return max_rsquared, position_at_maximum

##Filechooser#####################################################
root = tk.Tk()
root.withdraw() # we don't want a full GUI, so keep the root window from appearing
path=pathlib.Path().absolute()
file_names = filedialog.askopenfilenames(initialdir=path,parent=root,title='Choose a file')


########Plot labels and title ####################################
fig, ax1 = plt.subplots()
#fig.suptitle('IV Diodes')
ax1.set_xlabel('|Voltage| [V]')
ax1.set_ylabel('Capacitance [F]')

#######Define constants ##########################################
e0=8.8541878128E-2 #pF/cm vacuum permitivity
esi=11.68            # relative permitivity of silicon
esi02=3.9         # relative permitivity of SiO2
q=1.602E-7 #pFV charge unit
#Area=0.125*0.125 -math.pi*(math.pow((0.04/2.0),2)) #cm^2 Diode Area of flute1
Area=0.125*0.125  #cm^2 Diode Area of flute1
k=1.38064852*1E-23
T=273+24.7      #J/K
Q=k*T           #Thermal energy [FV^2]
Q=k*T*1E+12    #Thermal energy [pFV^2]
Ni=1.45*1e+10  # intristic concentration


##Loop in each file
for file_name in file_names:
    pos=file_name.find(".txt")
    if (pos!=-1):
        ###############Read data######################################
        label = file_name[0:pos].split('/')[-1]
        label_contents = label.split('_')
        name = label_contents[4] + "_" + label_contents[1] + "_" \
               + label_contents[6] + "_" + label_contents[8] + "_" + "standard"
        #ax1.plot(current, voltage, linestyle='solid', marker='o', label=name)
        ax1.title.set_text("IV_Diodes" + " : " + label_contents[2] + "_" + label_contents[3])


        data=pd.read_csv(file_name, sep="\t",skiprows=4)
        file_name = os.path.splitext(file_name)[0]
        voltage = abs(data.iloc[:,0])
        current = data.iloc[:,1]

       # type(voltage)
        print(data)
        ###############Plot data######################################
        ax1.plot(abs(voltage),current,linestyle='solid',marker='o',label=name)
        ax1.legend(loc="upper right")




        ########################## Nsub ####################################
        position_voltage_at_600 = int(np.where(voltage == 600.0)[0])
        current_at_600=current[int(position_voltage_at_600)]
        print(current_at_600)


        label = label_contents[3] + '_' + label_contents[4] + '_' + label_contents[2] + '_' + \
                label_contents[1] + '_' + label_contents[6]

        file_exists = os.path.isfile('./IV_Diodes.csv')
        with open('IV_Diodes.csv', newline='', mode='a') as csv_file:
            fieldnames = ['Name', 'I_600 [A]']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()  # file doesn't exist yet, write a header
                writer.writerow({'Name': label, 'I_600 [A]': '%.2E' % current_at_600 })
            else:
                written_data = pd.read_csv("./IV_Diodes.csv", sep=",")
                if label not in written_data.values:
                    writer.writerow({'Name': label, 'I_600 [A]': '%.2E' % current_at_600 })

        ax1.legend(loc="best", ncol=2, fontsize="small")



width = root.winfo_screenwidth()
height = root.winfo_screenheight()
print("width=",width)

fig.set_size_inches(width/100,height/100)
fig.savefig('IV_Diodes.png',dpi=300)


plt.show()


