import os
import pandas as pd
import csv
import numpy as np
import re


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


class data_collection:

    def __init__(self):

        self.ego_info_record = pd.DataFrame(columns=('ego_s', 'ego_d'))
        self.vehicle_info_1_record = pd.DataFrame(columns=('vehicle_1_s', 'vehicle_1_d'))
        self.vehicle_info_2_record = pd.DataFrame(columns=('vehicle_2_s', 'vehicle_2_d'))
        self.vehicle_info_3_record = pd.DataFrame(columns=('vehicle_3_s', 'vehicle_3_d'))
        self.vehicle_info_4_record = pd.DataFrame(columns=('vehicle_4_s', 'vehicle_4_d'))
        self.vehicle_info_5_record = pd.DataFrame(columns=('vehicle_5_s', 'vehicle_5_d'))

        self.directory_path = self.mkdir(os.path.abspath('.') + '/datas/plot_data_record/data_record_no_1')
        self.file_name_0 = 'ego_info_record.csv'
        self.file_name_1 = 'vehicle_info_1_record.csv'
        self.file_name_2 = 'vehicle_info_2_record.csv'
        self.file_name_3 = 'vehicle_info_3_record.csv'
        self.file_name_4 = 'vehicle_info_4_record.csv'
        self.file_name_5 = 'vehicle_info_5_record.csv'

    def data_record(self, info, name):
        if name == "ego_info":
            self.ego_info_record = self.ego_info_record.append(
                pd.DataFrame({info[0]: [info[1]], info[2]: [info[3]]}),
                ignore_index=True)
            self.ego_info_record.to_csv(self.directory_path + '/' + self.file_name_0,
                                   index=False)
        elif name == "vehicle_info_1":
            self.vehicle_info_1_record = self.vehicle_info_1_record.append(
                pd.DataFrame({info[0]: [info[1]], info[2]: [info[3]]}), ignore_index=True)
            self.vehicle_info_1_record.to_csv(self.directory_path + '/' + self.file_name_1,
                                            index=False)
        elif name == "vehicle_info_2":
            self.vehicle_info_2_record = self.vehicle_info_2_record.append(
                pd.DataFrame({info[0]: [info[1]], info[2]: [info[3]]}), ignore_index=True)
            self.vehicle_info_2_record.to_csv(self.directory_path + '/' + self.file_name_2,
                                              index=False)
        elif name == "vehicle_info_3":
            self.vehicle_info_3_record = self.vehicle_info_3_record.append(
                pd.DataFrame({info[0]: [info[1]], info[2]: [info[3]]}), ignore_index=True)
            self.vehicle_info_3_record.to_csv(self.directory_path + '/' + self.file_name_3,
                                            index=False)
        elif name == "vehicle_info_4":
            self.vehicle_info_4_record = self.vehicle_info_4_record.append(
                pd.DataFrame({info[0]: [info[1]], info[2]: [info[3]]}), ignore_index=True)
            self.vehicle_info_4_record.to_csv(self.directory_path + '/' + self.file_name_4,
                                            index=False)
        elif name == "vehicle_info_5":
            self.vehicle_info_5_record = self.vehicle_info_5_record.append(
                pd.DataFrame({info[0]: [info[1]], info[2]: [info[3]]}), ignore_index=True)
            self.vehicle_info_5_record.to_csv(self.directory_path + '/' + self.file_name_5,
                                            index=False)


    def mkdir(self, path):
        """
        创建指定的文件夹
        :param path: 文件夹路径，字符串格式
        :return: True(新建成功) or False(文件夹已存在，新建失败)
        """

        path = path.strip()
        path = path.rstrip("\\")
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)
            return path
        else:
            path = self.directory_check(path)
            os.makedirs(path)
            return path

    @staticmethod
    def directory_check(directory_check):
        temp_directory_check = directory_check
        i = 1
        while i:

            if os.path.exists(temp_directory_check):
                search = '_'
                numList = [m.start() for m in re.finditer(search, temp_directory_check)]
                numList[-1]
                temp_directory_check = temp_directory_check[0:numList[-1] + 1] + str(i)
                i = i + 1
            else:
                return temp_directory_check
