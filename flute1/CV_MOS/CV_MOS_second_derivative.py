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
from scipy.interpolate import CubicSpline
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from scipy import stats
import csv
import pathlib

# Denoting fitting function
def fit_func1(x,a, b):
    return a * x + b

def fit_func2(x,a, b):
    return a * x + b

def fit_func0(x,b):
    return  0*x + b

def myfunc(x):
  return slope*0 * x + intercept

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

def optimizefit_horizontal_curve(xdata,ydata,start_posfit1,start_posfit2):
    #optimize fit curve
    #first an initial position of the fit positions is determined to a fixed amount of data pionts
    #The number of data points is increased iteratively and for each interation a new fit is calculated
    #Those values with the maximym r-squared are kept to the final outpout
    position_voltage_horizontal_fit1 =int(np.where(xdata == start_posfit1)[0])
    position_voltage_horizontal_fit2 = int(np.where(xdata == (start_posfit1+2))[0])
    r_squared={}
    slopes={}
    intercepts={}
    for x in xdata:
        if (position_voltage_horizontal_fit2 < int(np.where(xdata == start_posfit2)[0])):
             slope, intercept, r, p, std_err = stats.linregress(xdata[int(position_voltage_horizontal_fit1):int(position_voltage_horizontal_fit2)],
                                                           ydata[int(position_voltage_horizontal_fit1):int(position_voltage_horizontal_fit2)])
             slope=slope*1.0
             r_squared[position_voltage_horizontal_fit2]=r**2
             slopes[position_voltage_horizontal_fit2]=slope
             intercepts[position_voltage_horizontal_fit2]=intercept*1.0
             position_voltage_horizontal_fit2+=1.0
             print("r=",r)
    position_at_maximum_rsquared=max(r_squared.items(), key=operator.itemgetter(1))[0]
    return slopes[position_at_maximum_rsquared],intercepts[position_at_maximum_rsquared],position_voltage_horizontal_fit1 , position_at_maximum_rsquared


def optimizefit_horizontal_curve_with_zero_slope(xdata,ydata,start_posfit1,start_posfit2):
    #optimize fit curve
    #first an initial position of the fit positions is determined to a fixed amount of data pionts
    #The number of data points is increased iteratively and for each interation a new fit is calculated
    #Those values with the maximym r-squared are kept to the final outpout
    position_voltage_horizontal_fit1 =int(np.where(xdata == start_posfit1)[0])
    position_voltage_horizontal_fit2 = int(np.where(xdata == (start_posfit1+2))[0])
    r_squared={}
    intercepts={}
    for x in xdata:
        if (position_voltage_horizontal_fit2 < int(np.where(xdata == start_posfit2)[0])):
            pars1, cov1 = optimize.curve_fit(fit_func0, xdata[int(position_voltage_horizontal_fit1):int(position_voltage_horizontal_fit2)],
                                             ydata[int(position_voltage_horizontal_fit1):int(position_voltage_horizontal_fit2)])
            r = rsquared(fit_func0, pars1, xdata[int(position_voltage_horizontal_fit1):int(position_voltage_horizontal_fit2)],
                                 ydata[int(position_voltage_horizontal_fit1):int(position_voltage_horizontal_fit2)])
            r_squared[int(position_voltage_horizontal_fit2)]=r
            intercepts[position_voltage_horizontal_fit2]=pars1[0]
            position_voltage_horizontal_fit2+=1.0
    position_at_maximum_rsquared=max(r_squared.items(), key=operator.itemgetter(1))[0]
    return intercepts[position_at_maximum_rsquared],position_voltage_horizontal_fit1 , position_at_maximum_rsquared



def optimizefit_incline_curve(xdata,ydata,start_posfit1,start_posfit2):
    #optimize fit curve
    #first an initial position of the fit positions is determined to a fixed amount of data pionts
    #The number of data points is increased iteratively and for each interation a new fit is calculated
    #Those values with the maximym r-squared are kept to the final outpout
    position_voltage_incline_fit1 =int(np.where(xdata == start_posfit1)[0])
    position_voltage_incline_fit2 =int(np.where(xdata == (start_posfit2))[0])
    r_squared={}
    r_squared2={}
    slopes={}
    intercepts={}
    for x in xdata:
        slope, intercept, r, p, std_err = stats.linregress(
            xdata[int(position_voltage_incline_fit1):int(position_voltage_incline_fit2)],
            ydata[int(position_voltage_incline_fit1):int(position_voltage_incline_fit2)])
        r_squared[position_voltage_incline_fit2] = r ** 2
        slopes[position_voltage_incline_fit2] = slope
        intercepts[position_voltage_incline_fit2] = intercept
        position_voltage_incline_fit2 += 1.0
    position_at_maximum_fit2=max(r_squared.items(), key=operator.itemgetter(1))[0]
    for x in xdata:
        if (position_voltage_incline_fit1 > 0):
            slope, intercept, r, p, std_err = stats.linregress(
                xdata[int(position_voltage_incline_fit1):int(position_at_maximum_fit2)],
                ydata[int(position_voltage_incline_fit1):int(position_at_maximum_fit2)])
            r_squared2[position_voltage_incline_fit1] = r ** 2
            slopes[position_voltage_incline_fit1] = slope
            intercepts[position_voltage_incline_fit1] = intercept
            position_voltage_incline_fit1 -= 1.0
    position_at_maximum_fit1 = max(r_squared2.items(), key=operator.itemgetter(1))[0]
    return slopes[position_at_maximum_fit1], intercepts[position_at_maximum_fit1] ,position_at_maximum_fit1,position_at_maximum_fit2

def second_derivative_method(xdata,ydata):
    cs = CubicSpline(xdata, ydata)
    ##plot CubicSpline
    diff_ydata = np.gradient(cs(xdata), xdata,edge_order=2)
    cs_first_derivative = CubicSpline(xdata, diff_ydata)
    diff_diff_ydata=np.gradient(cs_first_derivative(xdata),xdata,edge_order=2)
    cs_second_derivative = CubicSpline(xdata, diff_diff_ydata)
    #minimum_second_derivative = min(diff_diff_ydata)
    minimum_second_derivative = min(cs_second_derivative(xdata))
    print(minimum_second_derivative)
    #index_at_minimum = np.where(diff_diff_ydata == np.amin(diff_diff_ydata))[0]
    index_at_minimum = np.where(diff_diff_ydata == np.amin(cs_second_derivative(xdata)))[0]
    print(index_at_minimum)
    xdata_at_minimum = xdata[index_at_minimum[0]]
    ydata_at_minimum = ydata[index_at_minimum[0]]
    print(index_at_minimum)
    print(xdata_at_minimum)
    print(ydata_at_minimum)
    # maximum_second_derivative = min(cs_second_derivative(xdata)
    index_at_maximum = np.where(diff_diff_ydata == np.amax(cs_second_derivative(xdata)))[0]
    xdata_at_maximum = xdata[index_at_maximum[0]]
    ydata_at_maximum = ydata[index_at_maximum[0]]
    print(index_at_maximum)
    print(xdata_at_maximum)
    print(ydata_at_maximum)
    return xdata_at_minimum, ydata_at_minimum , xdata_at_maximum, ydata_at_maximum , cs_second_derivative





########Filechooser#####################################################
root = tk.Tk()
root.withdraw() # we don't want a full GUI, so keep the root window from appearing
path=pathlib.Path().absolute()
file_names = filedialog.askopenfilenames(initialdir=path,parent=root,title='Choose a file')

#directory=os.path.dirname(os.path.realpath(__file__))
#file_names=os.listdir(directory)
#print(file_names)

########Plot labels and title ####################################
fig1, (ax1, ax2) = plt.subplots(1,2)
fig1.suptitle('CV MOS')
ax1.set_xlabel('Voltage [V]')
ax1.set_ylabel('Capacitance [F]')
ax2.set_xlabel('$|Voltage|$ [V]')
ax2.set_ylabel('$1/C^{2} $ [F]')


fig2, (ax3) = plt.subplots()
fig2.suptitle('MOS : second derivative')
ax3.set_xlabel('Voltage [V]')
ax3.set_ylabel('diff^2(Capacitance) [F]')



#######Define constants ##########################################
e0=8.8541878128E-2 #pF/cm vacuum permitivity
esi=11.68            # relative permitivity of silicon
esi02=3.9         # relative permitivity of SiO2
q=1.602E-7 #pFV charge unit
Area=0.129*0.129 #cm^2 MOS Area of flute1
k=1.38064852*1E-23
T=273+24.7      #J/K
Q=k*T           #Thermal energy [FV^2]
Q=k*T*1E+12    #Thermal energy [pFV^2]
#   Ni=1.45*1e+10  # intristic concentration
#Ni=5.29*1E+19*(math.pow((T/300),2.54))*math.exp(-6726/T)
Ni=9.65*1E+9

print(e0)
print(esi)

##Loop in each file
for file_name in file_names:
    pos=file_name.find(".txt")
    if (pos!=-1):
        ###############Read data######################################
        label = file_name[0:pos].split('/')[-1]
        print(label)
        data=pd.read_csv(file_name, sep="\t",skiprows=4)
        data_sorted=data.sort_values(by=['Voltage (V)'],ascending=False)
        data_sorted.reset_index(drop=True,inplace=True)
        file_name = os.path.splitext(file_name)[0]
        voltage = -1*data_sorted.iloc[:,0]
        print(voltage)
        capacitance = data_sorted.iloc[:,1]*1E+12     #capacitance in pF
        inv_cap_squred=(1/(capacitance))**2    #1/C^2 in pF
        data_sorted['1/C2'] =inv_cap_squred/(Area**2)
        print(data_sorted)
        label_contents=label.split('_')
        name=label_contents[4]+"_"+label_contents[1]+"_"+label_contents[6]+"_"+label_contents[8]
        ###############Plot data######################################
        ax1.plot(voltage,capacitance,linestyle='solid',marker='o',label=name)
        ax2.plot(voltage,inv_cap_squred, linestyle='solid', marker='o', label=name)

        Vmin, Cmin , Vmax , Cmax , second_der=second_derivative_method(voltage[1:], capacitance[1:]) # finds the mimum and the maximum of the second derivative
        ax3.plot(voltage, second_der(voltage), label=name)
        #cs=CubicSpline(voltage,capacitance)
        print("The minimum second derivative of the capacitance is {} pF at {} V ".format('%.2E' % Cmin,'%.2E' % Vmin))
        print("The maximum second derivative of the capacitance is {} pF at {} V ".format('%.2E' % Cmax,'%.2E' % Vmax))
        slope1,intercept1,position_voltage_horizontal_fit1,position_voltage_horizontal_fit2=optimizefit_horizontal_curve(voltage,
                                                                                                                         capacitance,
                                                                                                                         voltage[1],
                                                                                                                         Vmin)
        ax1.plot(voltage[int(position_voltage_horizontal_fit1):int(position_voltage_horizontal_fit2)],
                 fit_func1(voltage[int(position_voltage_horizontal_fit1):int(position_voltage_horizontal_fit2)],
                           slope1,intercept1),
                 linestyle='--', linewidth=2, color='black')

        slope2,intercept2,position_voltage_incline_fit1,position_voltage_incline_fit2=optimizefit_incline_curve(voltage,
                                                                                                                capacitance,
                                                                                                                Vmin,
                                                                                                                Vmax)
        ax1.plot(voltage[int(position_voltage_incline_fit1):int(position_voltage_incline_fit2)],
                 fit_func2(voltage[int(position_voltage_incline_fit1):int(position_voltage_incline_fit2)],
                           slope2,intercept2),
                 linestyle='--', linewidth=2, color='black')
        f1=lambda x:slope1*x+intercept1
        f2=lambda x:slope2*x+intercept2
        Vfb=findIntersection(f1,f2,0)[0]
        print("The flat band voltage is Vfb=",Vfb)
        ######################## Cox ################################################################
        intercept1, position_voltage_horizontal_slope0_fit1, position_voltage_horizontal_slope0_fit2 = optimizefit_horizontal_curve_with_zero_slope(voltage,
                                                                                                                        capacitance,
                                                                                                                        voltage[0],
                                                                                                                        Vmin)
        ax1.plot(voltage[int(position_voltage_horizontal_slope0_fit1):int(position_voltage_horizontal_slope0_fit2)],
                 fit_func0(voltage[int(position_voltage_horizontal_slope0_fit1):int(position_voltage_horizontal_slope0_fit2)],intercept1),
                 linestyle='--', linewidth=2, color='green')

        Cox=intercept1
        print("The oxide capacitance is Cox=", Cox)
        ###################### Nsub ####################################################################
        slope3, intercept3, position_at_maximum_fit1, position_at_maximum_fit2 = optimizefit_incline_curve(voltage,
                                                                                                           inv_cap_squred,
                                                                                                           Vmin, Vmax)
        ax2.plot(voltage[int(position_at_maximum_fit1):int(position_at_maximum_fit2)],
                 fit_func1(voltage[int(position_at_maximum_fit1):int(position_at_maximum_fit2)],slope3, intercept3),
                 linestyle='--', linewidth=2, color='black')
        #Nsub = 2.0/((q* esi*e0 * (math.pow(Area, 2)))*(slope3))
        Nsub = 2.0 / ((q * esi * e0) * (slope3))
        print("Bulk doping concentration Nsub=", '%.2E' % Nsub)


        ###################### Nox ####################################################################
        fb = -1*(Q/q)*math.log(Nsub/Ni)
        WS = -0.61 + fb
        print("WS=",WS)
        Nox = Cox * (WS - (Vfb)) / (q * Area)
        Qox=Nox*Area*q # pFV
        Qox=Qox*1E-12 #FV
        print("Fixed oxide charge concentration Nox=", '%.2E' % Nox)
        print("Oxide charge concentration Qox=", '%.2E' % Qox)
        ###################### tox ####################################################################
        tox=(e0*esi02*Area)*1E-2/Cox
        print("Oxide thickness tox=", '%.2E' % tox)

        label = label_contents[3] + '_' + label_contents[4] + '_' + label_contents[2] + '_' + \
                label_contents[1] + '_' + label_contents[6]

        file_exists = os.path.isfile('./MOS_second_derivative.csv')
        with open('MOS_second_derivative.csv', newline='',mode='a') as csv_file:
            fieldnames = ['Name', 'V_fb [V]', 'C_acc [F]','N_ox [$cm^2$]' , 't_ox [m]']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()  # file doesn't exist yet, write a header
                writer.writerow({'Name': label, 'V_fb [V]': round(Vfb, 3), 'C_acc [F]': round(Cox, 3),
                                 'N_ox [$cm^2$]': '%.2E' % Nox,'t_ox [m]': '%.2E' % tox})
            else:
                written_data=pd.read_csv("./MOS_second_derivative.csv",sep=",")
                if label not in written_data.values:
                     writer.writerow({'Name': label, 'V_fb [V]': round(Vfb, 3), 'C_acc [F]': round(Cox, 3) ,
                                      'N_ox [$cm^2$]' : '%.2E' % Nox, 't_ox [m]': '%.2E' % tox})

        #label="Vfd={} V, Cmin={} pF\n".format(round(Vfb, 2),round(Cox, 2))

        ax1.legend(loc="best", ncol=2, fontsize="xx-small")
        ax2.legend(loc="best", ncol=2, fontsize="xx-small")
        ax3.legend(loc="best", ncol=2, fontsize="small")

width = root.winfo_screenwidth()
height = root.winfo_screenheight()
print("width=",width)

fig1.set_size_inches(width/100,height/100)
fig1.savefig('MOS_second_derivative_method.png',dpi=300)
fig2.set_size_inches(width/100,height/100)
fig2.savefig('MOS_second_derivative2.png',dpi=300)

plt.show()
