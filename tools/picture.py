#!/bin/bash python

import csv
import numpy as np
import xlrd
from matplotlib import pyplot as plt
import os
import pandas as pd


def exact_data(f):
    col_types = [float]
    f_csv = csv.reader(f)
    data = []
    index = 1
    for col in f_csv:
        col = tuple(convert(value) for convert, value in zip(col_types, col))
        data.append(col[0])
        index += 1

    return np.array(data)


def plot_data(rec_num, name_1="ego_info", name_2="ego_s", name_3="s_ahead"):
    if name_1 == "ego_info":
        if isinstance(rec_num, int):

            record_num = 'data_record_no_' + str(rec_num)
            d1 = pd.read_csv(os.path.abspath('..') + "/datas/plot_data_record/" + record_num + '/ego_info_record.csv')
            d01 = pd.read_csv(os.path.abspath('..') + "/datas/plot_data_record/" + record_num + '/vehicle_info_1_record.csv')
            d02 = pd.read_csv(os.path.abspath('..') + "/datas/plot_data_record/" + record_num + '/vehicle_info_2_record.csv')
            d03 = pd.read_csv(os.path.abspath('..') + "/datas/plot_data_record/" + record_num + '/vehicle_info_3_record.csv')
            d04 = pd.read_csv(os.path.abspath('..') + "/datas/plot_data_record/" + record_num + '/vehicle_info_4_record.csv')
            d05 = pd.read_csv(os.path.abspath('..') + "/datas/plot_data_record/" + record_num + '/vehicle_info_5_record.csv')

            ego_s = d1.ego_s.values
            ego_d = d1.ego_d.values

            vehicle_1_s = d01.info_vehicle_1_s.values
            vehicle_1_d = d01.info_vehicle_1_d.values

            vehicle_2_s = d02.info_vehicle_2_s.values
            vehicle_2_d = d02.info_vehicle_2_d.values

            vehicle_3_s = d03.info_vehicle_3_s.values
            vehicle_3_d = d03.info_vehicle_3_d.values

            vehicle_4_s = d04.info_vehicle_4_s.values
            vehicle_4_d = d04.info_vehicle_4_d.values

            vehicle_5_s = d05.info_vehicle_5_s.values
            vehicle_5_d = d05.info_vehicle_5_d.values

            t = np.arange(0, ego_d.shape[0])
            plt.figure()
            plt.plot(ego_s[0:-1], ego_d[0:-1], color="blue", label='ego_s_d')
            plt.plot(vehicle_1_s[0:-1], vehicle_1_d[0:-1], color="black", label='vehhicle_1_s_d')
            plt.plot(vehicle_2_s[0:-1], vehicle_2_d[0:-1], color="black", label='vehhicle_2_s_d')
            plt.plot(vehicle_3_s[0:-1], vehicle_3_d[0:-1], color="black", label='vehhicle_3_s_d')
            plt.plot(vehicle_4_s[0:-1], vehicle_4_d[0:-1], color="black", label='vehhicle_4_s_d')
            plt.plot(vehicle_5_s[0:-1], vehicle_5_d[0:-1], color="black", label='vehhicle_5_s_d')

            plt.legend()
            plt.show()

        else:

            record_num_1 = 'data_record_no_' + str(rec_num[0])
            record_num_2 = 'data_record_no_' + str(rec_num[1])
            d1 = pd.read_csv(os.path.abspath('..') + "/datas/plot_data_record/" + record_num_1 + '/ego_info_record.csv')
            d2 = pd.read_csv(os.path.abspath('..') + "/datas/plot_data_record/" + record_num_2 + '/ego_info_record.csv')

            d01 = pd.read_csv(os.path.abspath('..') + "/datas/plot_data_record/" + record_num_1 + '/vehicle_info_1_record.csv')
            d02 = pd.read_csv(os.path.abspath('..') + "/datas/plot_data_record/" + record_num_1 + '/vehicle_info_2_record.csv')
            d03 = pd.read_csv(os.path.abspath('..') + "/datas/plot_data_record/" + record_num_1 + '/vehicle_info_3_record.csv')
            d04 = pd.read_csv(os.path.abspath('..') + "/datas/plot_data_record/" + record_num_1 + '/vehicle_info_4_record.csv')
            d05 = pd.read_csv(os.path.abspath('..') + "/datas/plot_data_record/" + record_num_1 + '/vehicle_info_5_record.csv')

            vehicle_1_s = d01.info_vehicle_1_s.values
            vehicle_1_d = d01.info_vehicle_1_d.values

            vehicle_2_s = d02.info_vehicle_2_s.values
            vehicle_2_d = d02.info_vehicle_2_d.values

            vehicle_3_s = d03.info_vehicle_3_s.values
            vehicle_3_d = d03.info_vehicle_3_d.values

            vehicle_4_s = d04.info_vehicle_4_s.values
            vehicle_4_d = d04.info_vehicle_4_d.values

            vehicle_5_s = d05.info_vehicle_5_s.values
            vehicle_5_d = d05.info_vehicle_5_d.values
            ego_s_1 = d1.ego_s.values
            ego_s_2 = d2.ego_s.values
            ego_d_1 = d1.ego_d.values
            ego_d_1 = d2.ego_d.values

            t1 = np.arange(0, ego_s_1.shape[0])
            t2 = np.arange(0, ego_s_2.shape[0])
            plt.figure()
            plt.plot(ego_s_1[0:-1], ego_d_1[0:-1], color="blue", label=str(record_num_1) + "_ego_s_d")
            plt.plot(ego_s_2[0:-1], ego_d_1[0:-1], color="red", label=str(record_num_2) + "_ego_s_d")

            plt.plot(vehicle_1_s[0:-1], vehicle_1_d[0:-1], color="black", label='vehhicle_1_s_d')
            plt.plot(vehicle_2_s[0:-1], vehicle_2_d[0:-1], color="black", label='vehhicle_2_s_d')
            plt.plot(vehicle_3_s[0:-1], vehicle_3_d[0:-1], color="black", label='vehhicle_3_s_d')
            plt.plot(vehicle_4_s[0:-1], vehicle_4_d[0:-1], color="black", label='vehhicle_4_s_d')
            plt.plot(vehicle_5_s[0:-1], vehicle_5_d[0:-1], color="black", label='vehhicle_5_s_d')

            plt.legend()
            plt.show()


if __name__ == '__main__':
    '''
    rec_num = [1, 2]

    '''
    rec_num = [1,2]
    # rec_num = 1
    '''
    ego_s,  ego_d
    '''
    plot_data(rec_num, 'ego_info', 'ego_s','ego_d')


# t = np.arange(0, 1000)
# ob = xlrd.open_workbook('/home/wangliwen/Carla_RL_MPC/tools/ob_v_data.xlsx_1')
# ob = ob.sheets()[0]#第几个sheet
# plt.plot(t,np.array(ob.row_values(0)),'g',label='ob_v_data')
# plt.show()
