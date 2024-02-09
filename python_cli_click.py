import VDP_Poly
import click
import importlib.util
import os
import python_scripts
from pyfiglet import Figlet
import pathlib
import tkinter as tk
import pandas as pd
from tkinter import filedialog



class main:
    def __init__(self):
        self.data={}
        self.filenames={

        }

    def choose_measurment_type(m):
        measurment_name = m
        return m

    def choose_measurment_files():
        """PQC Analysis programm"""

        ##Filechooser
        root = tk.Tk()
        root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
        path = pathlib.Path().absolute()
        file_names = filedialog.askopenfilenames(initialdir=path, parent=root, title='Choose measurment files')
        # for x in range(count):
        #    click.echo('{} send kisses to Ioanna !'.format(m))
        # print(file_names)
        return file_names

    def build_dataframe_from_filnames(filenames):
        header_number = 4
        data = pd.read_csv(file_name, sep="\t", skiprows=header_number)
        return data


if __name__ == '__main__':
    f = Figlet(font='slant')
    print(f.renderText('PQC ANALYSIS'))
    filenames=choose_measurment_files()
    print(filenames)
    click.echo('{} send kisses to Ioanna !'.format(filenames))
    #execfile('./VDP_Polaady.py')
    #spec = importlib.util.spec_from_file_location("VDP_Poly.py", "./")
    #    foo = importlib.util.module_from_spec(spec)
    #spec.loader.exec_module(foo)
    #foo.MyClass()
    m=choose_measurment_type()
    click.echo('{}!'.format(m))
    for filename in filenames:
        data = build_dataframe_from_filnames(filename)
        click.echo("data")

    vdppoly = VDPPoly(data)

    #if m=='VDPPoly':

