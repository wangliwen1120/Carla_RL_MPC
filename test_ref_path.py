self.ref_path = xlrd.open_workbook(os.path.abspath('.') + '/tools/ref_global_path.xlsx')
        self.ref_path = self.ref_path.sheets()[0]
        self.ref_path_x = self.ref_path.col_values(0)
        self.ref_path_y = self.ref_path.col_values(1)