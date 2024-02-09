import math
import os
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfilenames
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import operator
from scipy import optimize
from scipy.optimize import fsolve
from pathlib import PurePath
from scipy import stats
from decimal import Decimal
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
    position_voltage_incline_fit1 = int(np.where(xdata == start_posfit1)[0])
    print(np.where(xdata == start_posfit2)[0])
    position_voltage_incline_fit2 = int(np.where(xdata == start_posfit2)[0])
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
    position_voltage_horizontal_fit1 = int(np.where(xdata == start_posfit1)[0])
    position_voltage_horizontal_fit2 = int(np.where(xdata == start_posfit2)[0])
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

##Filechooser
root = tk.Tk()
root.withdraw() # we don't want a full GUI, so keep the root window from appearing
path=pathlib.Path().absolute()
path_par=PurePath(path.parents[1], 'flute1/CV_Diodes')
file_names_flute3= filedialog.askopenfilenames(initialdir=path,parent=root,title='Choose a file')
file_names_flute1= filedialog.askopenfilenames(initialdir=path_par,parent=root,title='Choose a file')


########Plot labels and title ####################################

fig1=plt.figure()
# canvas1=FigureCanvas(fig1)
#fig1.suptitle('LinewidthStrip')
ax1= fig1.add_subplot()
ax1.set_xlabel('Voltage [V]')
ax1.set_ylabel('Capacitance [F]')
#ax1.set_ylim(-0.01,0.01)
#ax1.set_xlim(0,10)

fig2=plt.figure()
#FigureCanvas(fig2)
#fig2.suptitle('LinewidthStrip')
ax2= fig2.add_subplot()
ax2.set_xlabel('Voltage [V]')
ax2.set_ylabel('1/C^{2} [1/F^{-2}]')
#ax2.set_ylim(-0.025,0.025)
#ax2.set_xlim(0,10)
#print(plt.get_backend())


##Loop in each file
asmall=0.125
alarge=0.250

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


for file_name_flute3 in file_names_flute3:
    pos1 = file_name_flute3.find(".txt")
    if (pos1 != -1):
        label_flute3 = file_name_flute3[0:pos1].split('/')[-1]

        label_contents_flute3 = label_flute3.split('_')
        #        print(key_list[val_list.index(label)])
        label_check_flute3 = label_contents_flute3[3] + "_" + label_contents_flute3[4] \
                             + "_" + label_contents_flute3[2] + "_" + label_contents_flute3[1] \
                             + "_" + label_contents_flute3[6]

        for file_name_flute1 in file_names_flute1:
            pos2 = file_name_flute1.find(".txt")
            if (pos2 != -1):
                label_flute1 = file_name_flute1[0:pos2].split('/')[-1]

                label_contents_flute1 = label_flute1.split('_')
                #        print(key_list[val_list.index(label)])
                label_check_flute1 = label_contents_flute1[3] + "_" + label_contents_flute1[4] \
                                     + "_" + label_contents_flute1[2] + "_" + label_contents_flute1[1] \
                                     + "_" + label_contents_flute1[6]

                if (label_check_flute1 == label_check_flute3):
                    print(label_check_flute1, label_check_flute3)
                    data_flute1 = pd.read_csv(file_name_flute1, sep="\t", skiprows=4)
                    data_flute3 = pd.read_csv(file_name_flute3, sep="\t", skiprows=4)
                    data = pd.concat([data_flute1, data_flute3], axis=1, sort=False, keys=['flute_1', 'flute_3'])
                    #print(data)
                    voltage_flute1=abs(data['flute_1'].iloc[:,0])
                    capacitance_flute1 = data['flute_1'].iloc[:,1]
                    inv_cap_flute1=(1/capacitance_flute1)**2
                    voltage_flute3 =abs(data['flute_3'].iloc[:, 0])
                    capacitance_flute3 = data['flute_3'].iloc[:, 1]
                    inv_cap_flute3 = (1 / capacitance_flute3) ** 2
                    ax1.plot(abs(voltage_flute1),capacitance_flute1, linestyle='solid', marker='o')
                    ax1.plot(abs(voltage_flute3),capacitance_flute3, linestyle='solid', marker='+')
                    #ax2.plot(abs(voltage_flute1),inv_cap_flute1, linestyle='solid', marker='o')
                    #ax2.plot(abs(voltage_flute3),inv_cap_flute3,linestyle='solid', marker='+')
                    #ax2.plot(abs(voltage_flute1),inv_cap_flute1, linestyle='solid', marker='o')
                    #ax2.plot(abs(voltage_flute3),inv_cap_flute3, linestyle='solid', marker='+')
                    #asmall=0.125
                    #alarge=0.250
                    Cplanar=((capacitance_flute1/asmall)-(capacitance_flute3/alarge))/(asmall-alarge)
                    Cedge = (abs(alarge*(capacitance_flute1 / asmall) - asmall*(capacitance_flute3 / alarge))) / abs(
                        4*(asmall - alarge))

                    inv_cplanar = (1/Cplanar)** 2
                    inv_cedge = (1 / Cedge) ** 2
                    ax1.plot(abs(voltage_flute3), abs(Cplanar*asmall*asmall), linestyle='solid', marker='x',label="flute1")
                    ax2.plot(abs(voltage_flute3),abs(inv_cplanar), linestyle='solid', marker='x',label="Planar")
                    ax2.plot(abs(voltage_flute3),abs(inv_cplanar), linestyle='solid', marker='x',label="Planar")
                    ax1.plot(abs(voltage_flute3), abs(Cedge*asmall*asmall), linestyle='solid', marker='x',label="flute3")
                   # ax2.plot(abs(voltage_flute3), abs(inv_cedge), linestyle='solid', marker='+',label="Edge")
                    ax2.legend(loc="best", ncol=3, fontsize="small")
                    data['Cplanar']=Cplanar
                    data['Cedge']=Cedge
                    print(data)

                    ############Fit in the incline region##########################
                    max_rsquared, position_at_maximum = optimizefit_incline_curve(abs(voltage_flute3),inv_cplanar, 0.0,
                                                                                  100.0)  # Firt fit at position 0.0 and 100.0
                    position_voltage_incline_fit1 = int(np.where(voltage_flute3== 0.0)[0])
                    position_voltage_incline_fit2 = position_at_maximum
                    print("r_squared=", max_rsquared)
                    print(max_rsquared, position_at_maximum, voltage_flute3[position_at_maximum])
                    slope1, intercept1, r, p, std_err = stats.linregress(
                        voltage_flute3[position_voltage_incline_fit1:position_voltage_incline_fit2],
                        inv_cplanar[position_voltage_incline_fit1:position_voltage_incline_fit2])
                    ax2.plot(abs(voltage_flute3[position_voltage_incline_fit1:position_voltage_incline_fit2]),
                             fit_func1(abs(voltage_flute3[position_voltage_incline_fit1:position_voltage_incline_fit2]),
                                       slope1, intercept1),
                             linestyle='--', linewidth=2, color='black')
                    f1 = lambda x: slope1 * x + intercept1
                    ############Fit in the horizontal region##########################
                    max_rsquared_horizontal, position_at_maximum_rsquared_horizontal = optimizefit_horizontal_curve(
                        abs(voltage_flute3),inv_cplanar, 300.0, 500.0)  # Firt fit at position 500.0 and 1000.0
                    print(max_rsquared_horizontal, position_at_maximum_rsquared_horizontal,
                          voltage_flute3[position_at_maximum_rsquared_horizontal])
                    position_voltage_horizontal_fit1 = position_at_maximum_rsquared_horizontal
                    position_voltage_horizontal_fit2 = int(np.where(abs(voltage_flute3)== 500.0)[0])
                    slope2, intercept2, r, p, std_err = stats.linregress(
                        voltage_flute3[position_voltage_horizontal_fit1:position_voltage_horizontal_fit2],
                        inv_cplanar[position_voltage_horizontal_fit1:position_voltage_horizontal_fit2])
                    # pars2, cov2 = optimize.curve_fit(fit_func1,
                    #                               voltage_flute3[position_voltage_horizontal_fit1:position_voltage_horizontal_fit2],
                    #                               inv_cap_squred[position_voltage_horizontal_fit1:position_voltage_horizontal_fit2])
                    ax2.plot(abs(voltage_flute3[position_voltage_horizontal_fit1:position_voltage_horizontal_fit2]),
                             fit_func1(abs(voltage_flute3[position_voltage_horizontal_fit1:position_voltage_horizontal_fit2]),
                                       slope2, intercept2),
                             linestyle='--', linewidth=2, color='black')
                    f2 = lambda x: slope2 * x + intercept2
                    print("f1=", slope1, " ", intercept1)
                    print("f2=", slope2, " ", intercept2)
                    ##Calculate parameters#################################################
                    Vfd = findIntersection(f1, f2, 0.0)[0]
                    print("The depletion voltage is Vfd=", Vfd)
                    ############################# Cmin #################################
                    intercept3, position_voltage_horizontal_slope0_fit1, position_voltage_horizontal_slope0_fit2 = optimizefit_horizontal_curve_with_zero_slope(
                        voltage_flute3,
                        Cplanar,
                        Vfd,
                        500)
                    print(position_voltage_horizontal_slope0_fit1, position_voltage_horizontal_slope0_fit2)
                    ax1.plot(voltage_flute3[
                             int(position_voltage_horizontal_slope0_fit1):int(position_voltage_horizontal_slope0_fit2)],
                             fit_func0(voltage_flute3[int(position_voltage_horizontal_slope0_fit1):int(
                                 position_voltage_horizontal_slope0_fit2)], intercept3),
                             linestyle='--', linewidth=2, color='black')

                    Cmin = intercept3
                    print("Cmin=",Cmin)

                    ########################## Nsub ####################################
                    Nsub = (2.0 / (q * esi * e0 * slope1))
                    # Nsub = 2.0 / ((q * esi * e0 * (math.pow(Area, 2))) * (slope1))

                    print('%.2E' % Decimal(Nsub))
                    ########################## Active thickness#################################
                    active_thickness = ((e0 * esi * Area) / (Cmin))
                    print("active_thickness=", '%.2E' % active_thickness)

                    resistivity1 = 1 / (Nsub * m_h * q * 1E-12)  # 1/((cm^-3*cm^2*pFV)/Vs)
                    print("resistivity1=", resistivity1)

                    resistivity2 = math.pow(0.0320, 2) / (
                                2 * e0 * 1E-12 * esi * m_h * Vfd)  # cm^2/((pF/cm)*(cm^2/Vs)*V)
                    print("resistivity2=", resistivity2)



'''
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
        #label=file_name[0:pos].split('/')[-1]

        #label_contents = label.split('_')
#        print(key_list[val_list.index(label)])
        #label_check=label_contents[3]+"_"+label_contents[4]\
         #           +"_"+label_contents[2]+"_"+label_contents[1]\
         #           +"_"+label_contents[6]
        #print("label_check=", label_check)
        label = file_name[0:pos].split('/')[-1]

        label_contents = label.split('_')
        #        print(key_list[val_list.index(label)])
        label_check = label_contents[3] + "_" + label_contents[4] \
                      + "_" + label_contents[2] + "_" + label_contents[1] \
                      + "_" + label_contents[6]
        #label_list.append(label)


        for measurement in Rsh_measurment_name:
            #print(measurement)
            #print(label_check)

           # measurement_check = measurement[3] + '_' + measurement[4] + '_' + measurement[2] + '_' + \
           #         measurement[1] + '_' + measurement[6] + '_' + measurement[8]

            if(label_check in measurement):

                #label_list.append(label)
                #print("checked")
                position=Rsh_measurment_name[Rsh_measurment_name == measurement].index[0]
                #print("position=",position)
                #print(Rsh[int(position)])
                if "_s" in measurement:
                    if label not in label_list:
                        label_list.append(label)
                    #print(measurement)
                    name = label_contents[4] + "_" + label_contents[1] + "_" \
                           + label_contents[6] + "_" + label_contents[8] + "_"+"s"
                    ax1.plot(current, voltage, linestyle='solid', marker='o', label=name)
                    ax1.title.set_text(
                        "LinewidthStrip" + " : " + label_contents[2] + "_" + label_contents[3] + "_" + "standard")
                    ##fit the data
                   # position in which the fit starts
                    pars1, cov = optimize.curve_fit(fit_func, current[3:], voltage_flute3[3:])  # pars1 is the parameters of the fit and cov is the convolution
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
                    if label not in label_list:
                        label_list.append(label)
                   # label = file_name[0:pos].split('/')[-1]
                    #label_list.append(label)
                    name = label_contents[4] + "_" + label_contents[1] + "_" + \
                           label_contents[6] + "_" + label_contents[8] + "_"+"r"
                    #print(name)
                    ax2.plot(current, voltage, linestyle='solid', marker='o', label=name)
                    ax2.title.set_text(
                        "LinewidthStrip" + " : " + label_contents[2] + "_" + label_contents[3] + "_" + "rotated")
                    # position in which the fit starts
                    pars1, cov = optimize.curve_fit(fit_func, current[3:], voltage_flute3[3:])  # pars1 is the parameters of the fit and cov is the convolution
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


file_exists = os.path.isfile('./LinewidthStrip.csv')
print("file_exists=", file_exists)
with open('LinewidthStrip.csv', newline='', mode='a') as csv_file:
    fieldnames = ['Name', 'Linewidth_Strip-standard [m]', 'Linewidth_Strip-rotated [m]']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not file_exists:
        print("Write header")
        writer.writeheader()  # file doesn't exist yet, write a header
    writer.writerow({'Name': label_list[0], 'Linewidth_Strip-standard [m]': t_s_list[0],
                    'Linewidth_Strip-rotated [m]': t_r_list[0]})
    #written_data = pd.read_csv("./LinewidthStrip.csv", sep=",")
    print(len(label_list))
    for i in range(1,(len(label_list))):
        #if label_list[i] not in written_data.values:
        print(label_list[i])
        writer.writerow({'Name': label_list[i], 'Linewidth_Strip-standard [m]': t_s_list[i],
                        'Linewidth_Strip-rotated [m]': t_r_list[i]})


width = root.winfo_screenwidth()
height = root.winfo_screenheight()
print("width=",width)

fig1.set_size_inches(width/100,height/100)
fig2.set_size_inches(width/100,height/100)

fig1.savefig('LinewidthStrip-standard.png',dpi=300)
fig2.savefig('LinewidthStrip-rotated.png',dpi=300)


'''
plt.show()