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
     return fsolve(lambda x :fun1(x) - fun2(x),200)

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
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('CV Diodes')
ax1.set_xlabel('|Voltage| [V]')
ax1.set_ylabel('Capacitance [F]')
ax2.set_xlabel('$|Voltage|$ [V]')
ax2.set_ylabel('$1/C^{2} $ [F]')
#######Define constants ##########################################
e0=8.854E-14 #F/cm vacuum permitivity
esi=11.68            # relative permitivity of silicon
esi02=3.9         # relative permitivity of SiO2
q=1.602E-19 #FV charge unit
Area=0.125*0.125 -math.pi*(math.pow((0.04/2.0),2)) #cm^2 Diode Area of flute1
#Area=0.125*0.125  #cm^2 Diode Area of flute1
k=1.38064852*1E-23
T=273+24.7      #J/K
Q=k*T           #Thermal energy [FV^2]
Q=k*T*1E+12    #Thermal energy [pFV^2]
Ni=1.45*1e+10  # intristic concentration

m_h=450    #cm^2/Vs


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
        ax1.title.set_text("CV_Diodes" + " : " + label_contents[2] + "_" + label_contents[3])
        ax2.title.set_text("CV_Diodes" + " : " + label_contents[2] + "_" + label_contents[3])

        data=pd.read_csv(file_name, sep="\t",skiprows=4)
        file_name = os.path.splitext(file_name)[0]
        voltage = abs(data.iloc[:,0])
        capacitance = ((data.iloc[:,1]/Area))     #capacitance in pF
        inv_cap_squred=(1/(capacitance))**2
       # type(voltage)
        data['1/C2'] =inv_cap_squred
        print(data)
        ###############Plot data######################################
        ax1.plot(abs(voltage),capacitance,linestyle='solid',marker='o',label=name)
        ax1.legend(loc="upper right")
        ax2.plot(abs(voltage),inv_cap_squred, linestyle='solid', marker='o',label=name)
        ax2.legend(loc="lower right")

        ############Fit in the incline region##########################
        max_rsquared, position_at_maximum = optimizefit_incline_curve(voltage, inv_cap_squred,0.0,200.0) #Firt fit at position 0.0 and 100.0
        position_voltage_incline_fit1 = int(np.where(voltage == 0.0)[0])
        position_voltage_incline_fit2 = position_at_maximum
        print("r_squared=", max_rsquared)
        print(max_rsquared, position_at_maximum, voltage[position_at_maximum])
        slope1, intercept1, r, p, std_err = stats.linregress(
            voltage[position_voltage_incline_fit1:position_voltage_incline_fit2],
            inv_cap_squred[position_voltage_incline_fit1:position_voltage_incline_fit2])
        ax2.plot(abs(voltage[position_voltage_incline_fit1:position_voltage_incline_fit2]),
                 fit_func1(abs(voltage[position_voltage_incline_fit1:position_voltage_incline_fit2]),slope1,intercept1),
                 linestyle='--', linewidth=2, color='black')
        f1 = lambda x: slope1 * x + intercept1
        ############Fit in the horizontal region##########################
        max_rsquared_horizontal, position_at_maximum_rsquared_horizontal = optimizefit_horizontal_curve(voltage, inv_cap_squred,300.0, 500.0)#Firt fit at position 500.0 and 1000.0
        print(max_rsquared_horizontal,position_at_maximum_rsquared_horizontal, voltage[position_at_maximum_rsquared_horizontal])
        position_voltage_horizontal_fit1 = position_at_maximum_rsquared_horizontal
        position_voltage_horizontal_fit2 = int(np.where(voltage == 500.0)[0])
        slope2, intercept2, r, p, std_err = stats.linregress(
            voltage[position_voltage_horizontal_fit1:position_voltage_horizontal_fit2],
            inv_cap_squred[position_voltage_horizontal_fit1:position_voltage_horizontal_fit2])
        #pars2, cov2 = optimize.curve_fit(fit_func1,
        #                               voltage[position_voltage_horizontal_fit1:position_voltage_horizontal_fit2],
        #                               inv_cap_squred[position_voltage_horizontal_fit1:position_voltage_horizontal_fit2])
        ax2.plot(abs(voltage[position_voltage_horizontal_fit1:position_voltage_horizontal_fit2]),
                 fit_func1(abs(voltage[position_voltage_horizontal_fit1:position_voltage_horizontal_fit2]), slope2,intercept2),
                 linestyle='--', linewidth=2, color='black')
        f2=lambda x: slope2*x+intercept2
        print("f1=", slope1, " ", intercept1)
        print("f2=",slope2," ",intercept2)
        ##Calculate parameters#################################################
        Vfd=findIntersection(f1, f2, 0.0)[0]
        print("The depletion voltage is Vfd=",Vfd)
        ############################# Cmin #################################
        intercept3, position_voltage_horizontal_slope0_fit1, position_voltage_horizontal_slope0_fit2 = optimizefit_horizontal_curve_with_zero_slope(voltage,
                                                                                                                        capacitance,
                                                                                                                        Vfd,
                                                                                                                        500)
        print(position_voltage_horizontal_slope0_fit1, position_voltage_horizontal_slope0_fit2)
        ax1.plot(voltage[int(position_voltage_horizontal_slope0_fit1):int(position_voltage_horizontal_slope0_fit2)],
                 fit_func0(voltage[int(position_voltage_horizontal_slope0_fit1):int(position_voltage_horizontal_slope0_fit2)],intercept3),
                 linestyle='--', linewidth=2, color='black')

        Cmin=intercept3*Area

        ########################## Nsub ####################################
        Nsub = (2.0 / (q * esi *e0* slope1))
        #Nsub = 2.0 / ((q * esi * e0 * (math.pow(Area, 2))) * (slope1))

        print('%.2E' % Decimal(Nsub))
        ########################## Active thickness#################################
        active_thickness=((e0*esi*Area)/(Cmin))
        print("active_thickness=", '%.2E' % active_thickness)

        resistivity1=1/(Nsub*m_h*q*1E-12)   #1/((cm^-3*cm^2*pFV)/Vs)
        print("resistivity1=",resistivity1)

        resistivity2 = math.pow(0.0320,2) /(2*e0*1E-12*esi*m_h*Vfd)  #cm^2/((pF/cm)*(cm^2/Vs)*V)
        print("resistivity2=", resistivity2)

        label = label_contents[3] + '_' + label_contents[4] + '_' + label_contents[2] + '_' + \
                label_contents[1] + '_' + label_contents[6]

        file_exists = os.path.isfile('./CV_DIodes.csv')
        with open('CV_Diodes.csv', newline='', mode='a') as csv_file:
            fieldnames = ['Name', 'V_fd [V]', 'C_min [F]', 'N_sub [$cm^2$]', 'd [m]']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()  # file doesn't exist yet, write a header
                writer.writerow({'Name': label, 'V_fd [V]': '%.2E' % Vfd, 'C_min [F]': '%.2E' % Cmin,
                                 'N_sub [$cm^2$]': '%.2E' % Nsub, 'd [m]': '%.2E' %active_thickness})
            else:
                written_data = pd.read_csv("./CV_DIodes.csv", sep=",")
                if label not in written_data.values:
                    writer.writerow({'Name': label, 'V_fd [V]': '%.2E' % Vfd, 'C_min [F]': '%.2E' % Cmin,
                                     'N_sub [$cm^2$]': '%.2E' % Nsub, 'd [m]': '%.2E' %active_thickness})

        ax1.legend(loc="best", ncol=2, fontsize="xx-small")
        ax2.legend(loc="best", ncol=2, fontsize="xx-small")


width = root.winfo_screenwidth()
height = root.winfo_screenheight()
print("width=",width)

fig.set_size_inches(width/100,height/100)
fig.savefig('CV_Diodes.png',dpi=300)


plt.show()


