import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import xlrd
import pickle
import torch
import torch.nn as nn
from datetime import date
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    
    #choose PM2.5 or PM10 data
    #filename = 'pm2.5.xlsx' 
    filename = 'pm10.xlsx'

    wb = xlrd.open_workbook(filename)
    sheet = wb.sheet_by_index(0)
    f = open('harvest_landmarks.csv', "r")
    reader = csv.reader(f)

    years, dates, corn, soybeans = csv_date_conversion(reader)
    write_corn_days(years, dates, corn)
    years_xl, dates_xl, pm10 = xl_date_conversion(sheet)
    final_corn = []
    final_soybeans = []
    final_harvest = []

    for i in range(len(years_xl)):
        day_check = dates_xl[i]
        indices = [j for j, obj in enumerate(dates) if obj == day_check]
        found = False
        for k in range(len(indices)):
            if(years[indices[k]] == years_xl[i]):

            	# handling representation of harvest period, either 0/1, values, or derivatives of values

                final_corn.append(corn[indices[k]])
                final_soybeans.append(soybeans[indices[k]])
                
                '''
                if(corn[indices[k]] > 0):
                    final_corn.append(1)
                else:
                    final_corn.append(0)

                if(soybeans[indices[k]] > 0):
                    final_soybeans.append(1)
                else:
                    final_soybeans.append(0)
                '''
                
                found = True
                break
        if(not found):
            final_corn.append(0)
            final_soybeans.append(0)

    
    for m in range(len(final_corn)):
        if(final_corn[m] > 0 or final_soybeans[m] > 0):
            val = 0.4
            to_add = val*final_corn[m] + (1-val)*final_soybeans[m]
            final_harvest.append(to_add)
            
            #final_harvest.append(1)
        else:
            final_harvest.append(0.0)


    precip, rel_hum, wind = get_precip_rel_hum("weather_raw.txt")
    pm10_deriv = generate_derivatives(years_xl, pm10)
    final_harvest_deriv = generate_derivatives(years_xl, final_harvest)
    soybeans_deriv = generate_derivatives(years_xl, final_soybeans) 
    corn_deriv = generate_derivatives(years_xl, final_corn)

    linear_model(years_xl, dates_xl, pm10, final_corn, final_soybeans, rel_hum, precip, final_harvest, wind, 
        pm10_deriv, final_harvest_deriv, soybeans_deriv, corn_deriv)
    #nonlinear_model_neural(years_xl, dates_xl, pm10, final_corn, final_soybeans, rel_hum, precip, final_harvest, wind)
    #nonlinear_model_svm(years_xl, dates_xl, pm10, final_corn,final_soybeans, rel_hum, precip, final_harvest, wind)
    #nonlinear_model_log_reg(years_xl, dates_xl, pm10, final_corn, final_soybeans, rel_hum, precip, final_harvest, wind)
    nonlinear_model_svr(years_xl, dates_xl, pm10, final_corn,final_soybeans, rel_hum, precip, final_harvest, wind, 
        pm10_deriv, final_harvest_deriv, soybeans_deriv, corn_deriv)
    #plot_generation(years_xl, pm10_deriv, final_harvest_deriv, corn_deriv, soybeans_deriv, final_harvest, final_corn, final_soybeans)

    f.close()


def linear_model(years_xl, dates_xl, pm10, final_corn, final_soybeans, rel_hum, precip, final_harvest, wind,
    pm10_deriv, final_harvest_deriv, soybeans_deriv, corn_deriv):
    df = pd.DataFrame({'Year': years_xl, 'Day': dates_xl, 'PM10': pm10, 'Corn': final_corn, 'Soybean': final_soybeans,
                       'Relative_Humidity': rel_hum, 'Precipitation': precip, 'Winds': wind, 'Harvest': final_harvest, 
                       'Harvest_Derivative': final_harvest_deriv, 'Corn_Derivative': corn_deriv, 'Soybeans_Derivative': soybeans_deriv,
                       'PM10_Derivative': pm10_deriv})

    mod = smf.ols(formula='PM10_Derivative ~ C(Year) + C(Day) + Soybeans_Derivative', data=df)
    res = mod.fit()
    print(res.summary())

def plot_generation(years_xl, pm10_deriv, final_harvest_deriv, corn_deriv, soybeans_deriv, final_harvest, final_corn, final_soybeans):
    plt.plot(years_xl, pm10_deriv)
    plt.show()

    

class Pm10Dataset(Dataset):
    def __init__(self, data, labels):
        self.labels = (torch.from_numpy(labels)).float()
        self.data = (torch.from_numpy(data)).float()

    def __len__(self):
        return self.data.size()[0]

    def __getitem__(self, idx):
        return (self.data[idx], self.labels[idx])


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        '''
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 1, 2),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.MaxPool1d(1, 2))
        '''
        self.fc1 = nn.Linear(1, 800)
        self.fc2 = nn.Linear(800, num_classes)

    def forward(self, x):
        if(len(x.shape) < 2 or x.shape[0] < x.shape[1]):
            x = x.view((list(x.size()))[0], 1)
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        return x


def nonlinear_model_neural(years_xl, dates_xl, pm10, final_corn, final_soybeans, rel_hum, precip, final_harvest, wind):
    pm10_deriv = generate_derivatives(years_xl, pm10)
    #final_harvest = generate_derivatives(years_xl, final_harvest)
    train_data, test_data, train_labels, test_labels = divide_data(
        years_xl, pm10_deriv, final_harvest)

    device = torch.device('cpu')
    num_epochs = 15
    batch_size = 283
    learning_rate = 0.01

    train_dataset = Pm10Dataset(train_data, train_labels)
    test_dataset = Pm10Dataset(test_data, test_labels)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            data = data.view(batch_size, 1)
            labels = labels.to(device)

            
            outputs = model(data)
            labels = (labels.type(torch.LongTensor))

            print(outputs.shape)
            print(data.shape)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the test data: {} %'.format(
            100 * correct / total))

    torch.save(model.state_dict(), 'model.ckpt')


def nonlinear_model_svm(years_xl, dates_xl, pm10, final_corn, final_soybeans, rel_hum, precip, final_harvest, wind):
    pm10_deriv = generate_derivatives(years_xl, pm10)
    #final_harvest = generate_derivatives(years_xl, final_harvest)
    train_data, test_data, train_labels, test_labels = divide_data(years_xl, pm10_deriv, final_harvest)
    train_data = train_data.reshape(-1, 1)
    test_data = test_data.reshape(-1, 1)

    model = svm.SVC()
    model.fit(train_data, train_labels)
    #print(model.support_vectors_)
    predicted = create_numpy_array()
    for val in test_data:
        prediction = (model.predict([val]))[0]
        predicted = np.append(predicted, prediction)

    correct = 0
    for i in range(len(test_labels)):
        if(predicted[i] == test_labels[i]):
            correct = correct + 1

    accuracy = correct/len(test_labels) * 100
    print('SVM Accuracy: ' + str(accuracy) + '%')


def nonlinear_model_log_reg(years_xl, dates_xl, pm10, final_corn, final_soybeans, rel_hum, precip, final_harvest, wind):
    pm10_deriv = generate_derivatives(years_xl, pm10)
    #final_harvest = generate_derivatives(years_xl, final_harvest)
    train_data, test_data, train_labels, test_labels = divide_data(
        years_xl, pm10_deriv, final_harvest)
    train_data = train_data.reshape(-1, 1)
    test_data = test_data.reshape(-1, 1)

    model = LogisticRegression()
    model.fit(train_data, train_labels)
    #print(train_data)
    #print(train_labels)
    #for i in range(len(train_data)):
    #	print('data: ' + str(train_data[i]) + 'label: ' + str(train_labels[i]))
    accuracy = model.score(test_data, test_labels) * 100
    print('Logistic Regression Accuracy: ' + str(accuracy) + '%')

def nonlinear_model_svr(years_xl, dates_xl, pm10, final_corn, final_soybeans, rel_hum, precip, final_harvest, wind,
    pm10_deriv, final_harvest_deriv, soybeans_deriv, corn_deriv):
    pm10_deriv = generate_derivatives(years_xl, pm10)
    pm10_deriv = pm10_deriv.reshape(-1,1)
    final_harvest = np.array(final_harvest)
    #final_harvest = generate_derivatives(years_xl, final_harvest)
    final_harvest = final_harvest.reshape(-1,1)
    svp = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.01, degree = 3, kernel='linear'))
    svp.fit(final_harvest, pm10_deriv)

    r2 = svp.score(final_harvest, pm10_deriv)
    parameters = svp.get_params()
    print('R^2 value for SVR is: ' + str(r2))
    print('parameters for SVR are: ' + str(parameters))


def divide_data(years_xl, pm10_deriv, final_harvest):
    train_data = create_numpy_array()
    test_data = create_numpy_array()
    train_labels = create_numpy_array()
    test_labels = create_numpy_array()
    for i in range(len(years_xl)):
        if(years_xl[i] % 2 == 1):
            train_data = np.append(train_data, pm10_deriv[i])
            train_labels = np.append(train_labels, final_harvest[i])
        else:
            test_data = np.append(train_data, pm10_deriv[i])
            test_labels = np.append(train_labels, final_harvest[i])

    return train_data, test_data, train_labels, test_labels


def generate_derivatives(years_xl, pm10):
    border_indices = get_border_indices(years_xl)
    date_deriv = create_numpy_array()
    cur_array = create_numpy_array()
    for i in range(len(pm10)):
        if((i+1) in border_indices):
            cur_array = np.append(cur_array, pm10[i])
            deriv = np.gradient(cur_array)
            for val in deriv:
                date_deriv = np.append(date_deriv, val)
            cur_array = create_numpy_array()
        else:
            cur_array = np.append(cur_array, pm10[i])
    return date_deriv


def get_border_indices(years_xl):
    border_indices = []
    cur_year = years_xl[0]
    last_year = years_xl[0]
    for i in range(len(years_xl)):
        cur_year = years_xl[i]
        if(cur_year != last_year):
            border_indices.append(i)
        last_year = cur_year
    border_indices.append(len(years_xl))
    return border_indices


def xl_date_conversion(sheet):
    years = []
    days = []
    pm10 = []
    for i in range(1, sheet.nrows):
        if i == 0:
            continue
        raw_date = sheet.cell_value(i, 3)
        raw_data = float(sheet.cell_value(i, 4))
        raw_date_spl = raw_date.split('/')
        month = int(raw_date_spl[0])
        day = int(raw_date_spl[1])
        year = int(raw_date_spl[2])
        date_int = date_to_int(year, month, day)

        years.append(year)
        days.append(date_int)
        pm10.append(raw_data)

    return years, days, pm10


def csv_date_conversion(reader):
    is_start = True
    dim = dict()
    years = []
    dates = []
    corn_l = []
    soybeans_l = []
    for row in reader:
        if(is_start):
            is_start = False
            continue
        date = row[0]
        corn = int(row[1])
        soybean = int(row[2])

        date_split = date.split('-')
        # print(date_split)
        year = int(date_split[0])
        if year not in dim:
            dim[year] = dict()
        month = int(date_split[1])
        day = int(date_split[2])
        date_int = date_to_int(year, month, day)
        dim[year][date_int] = (corn, soybean)

    for year in dim:
        x = (list(dim[year].keys()))[::-1]
        y1 = []
        y2 = []
        for day in dim[year]:
            y1.append(dim[year][day][0])
            y2.append(dim[year][day][1])
        y1 = y1[::-1]
        y2 = y2[::-1]
        vals = np.arange(x[0], x[len(x)-1]+1)
        new_corn = np.interp(vals, x, y1)
        new_soybeans = np.interp(vals, x, y2)

        for i in range(len(vals)):
            years.append(year)
            dates.append(vals[i])
            corn_l.append(new_corn[i])
            soybeans_l.append(new_soybeans[i])
    return years, dates, corn_l, soybeans_l


def date_to_int(year, month, day):
    year_to_leap = {2001: False, 2002: False, 2003: False, 2004: True, 2005: False, 2006: False, 2007: False, 2008: True, 2009: False, 2010: False,
                    2011: False, 2012: True, 2013: False, 2014: False, 2015: False, 2016: True, 2017: False, 2018: False, 2019: False}
    month_to_days = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31,
                     6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    if(year_to_leap[year]):
        month_to_days[2] = 29
    day_ct = 0
    for i in range(0, month - 1):
        day_ct = day_ct + month_to_days[i+1]
    day_ct = day_ct + day
    return day_ct


def get_precip_rel_hum(filename):
    f = open(filename, "r")
    rel_hum = []
    precip = []
    wind = []
    sum_rh = 0
    len_rh = 0
    sum_p = 0
    len_p = 0
    sum_w = 0
    len_w = 0
    for line in f:
        b = line.split()
        if(is_float(b[12]) and (float(b[12]) < 100 and float(b[12]) > 0)):
            rel_hum.append(float(b[12]))
            sum_rh = sum_rh + float(b[12])
            len_rh = len_rh + 1
        else:
            rel_hum.append(-1)
        if(is_float(b[14]) and float(b[14]) < 8):
            precip.append(float(b[14]))
            sum_p = sum_p + float(b[14])
            len_p = len_p + 1
        else:
            precip.append(-1)
        if(is_float(b[4]) and float(b[4]) < 50):
            wind.append(float(b[4]))
            sum_w = sum_w + float(b[4])
            len_w = len_w + 1
        else:
            wind.append(-1)
    avg_rh = sum_rh/len_rh
    avg_p = sum_p/len_p
    avg_w = sum_w/len_w
    for j in range(len(rel_hum)):
        if(rel_hum[j] == -1):
            rel_hum[j] = avg_rh
        if(precip[j] == -1):
            precip[j] = avg_p
        if(wind[j] == -1):
        	wind[j] = avg_w
    f.close()

    days = [67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127, 130, 133, 136, 139, 142, 145, 148, 151, 154, 157, 160, 163, 166, 169, 172, 175, 178, 181, 184, 187, 190, 193, 196, 199, 202, 205, 208, 211, 214, 217, 220, 223, 226, 229, 232, 235, 238, 241, 244, 247, 250, 253, 256, 259, 262, 265, 268, 271, 274, 277, 280, 283, 286, 289, 292, 295, 298, 301, 304, 307, 310, 313, 316, 319, 322, 325, 328, 331, 334, 337, 340, 343, 346, 349, 352, 355, 358, 361, 364, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92, 95, 98, 101, 104, 107, 110, 113, 116, 119, 122, 125, 128, 131, 134, 137, 140, 143, 146, 149, 152, 155, 158, 161, 164, 167, 170, 173, 176, 179, 182, 185, 188, 191, 194, 197, 200, 203, 206, 209, 212, 215, 218, 221, 224, 227, 230, 233, 236, 239, 242, 245, 248, 251, 254, 257, 260, 263, 266, 269, 272, 275, 278, 281, 284, 287, 290, 293, 296, 299, 302, 305, 308, 311, 314, 317, 320, 323, 326, 329, 332, 335, 338, 341, 344, 347, 350, 353, 356, 359, 362, 365, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117, 120, 123, 126, 129, 132, 135, 138, 141, 144, 147, 150, 153, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 186, 189, 192, 195, 198, 201, 204, 207, 210, 213, 216, 219, 222, 225, 228, 231, 234, 237, 240, 243, 246, 249, 252, 255, 258, 261, 264, 267, 270, 273, 276, 279, 282, 285, 288, 291, 294, 297, 300, 303, 306, 309, 312, 315, 318, 321, 324, 327, 330, 333, 336, 339, 342, 345, 348, 351, 354, 357, 360, 363, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127, 130, 133, 136, 139, 142, 145, 148, 151, 154, 157, 160, 163, 166, 169, 172, 175, 178, 181, 184, 187, 190, 193, 196, 199, 202, 205, 208, 211, 214, 217, 220, 223, 226, 229, 232, 235, 238, 241, 244, 247, 250, 253, 256, 259, 262, 265, 268, 271, 274, 277, 280, 283, 286, 289, 292, 295, 298, 301, 304, 307, 310, 313, 316, 319, 322, 325, 328, 331, 334, 337, 340, 343, 346, 349, 352, 355, 358, 361, 364, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127, 130, 133, 136, 139, 142, 145, 148, 151, 154, 157, 160, 163, 166, 169, 172, 175, 178, 181, 184, 187, 190, 193, 196, 199, 202, 205, 208, 211, 214, 217, 220, 223, 226, 229, 232, 235, 238, 241, 244, 247, 250, 253, 256, 259, 262, 265, 268, 271, 274, 277, 280, 283, 286, 289, 292, 295, 298, 301, 304, 307, 310, 313, 316, 319, 322, 325, 328, 331, 334, 337, 340, 343, 346, 349, 352, 355, 358, 361, 364, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92, 95, 98, 101, 104, 107, 110, 113, 116, 119, 122, 125, 128, 131, 134, 137, 140, 143, 146, 149, 152, 155, 158, 161, 164, 167, 170, 173, 176, 179, 182, 185, 188, 191, 194, 197, 200, 203, 206, 209, 212, 215, 218, 221, 224, 227, 230, 233, 236, 239, 242, 245, 248, 251, 254, 257, 260, 263, 266, 269, 272, 275, 278, 281, 284, 287, 290, 293, 296, 299, 302, 305, 308, 311, 314, 317, 320, 323, 326, 329, 332, 335, 338, 341, 344, 347, 350, 353, 356, 359, 362, 365, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117, 120, 123, 126, 129, 132, 135, 138, 141, 144, 147, 150, 153, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 186, 189, 192, 195, 198, 201, 204, 207, 210, 213, 216, 219, 222, 225, 228, 231, 234, 237, 240, 243, 246, 249, 252, 255, 258, 261, 264, 267, 270, 273, 276, 279, 282, 285, 288, 291, 294, 297, 300, 303, 306, 309, 312, 315, 318, 321, 324, 327, 330, 333, 336, 339, 342, 345, 348, 351, 354, 357, 360, 363, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127, 130, 133, 136, 139, 142, 145, 148, 151, 154, 157, 160, 163, 166, 169, 172, 175, 178, 181, 184, 187, 190, 193, 196, 199, 202, 205, 208, 211, 214, 217, 220, 223, 226, 229, 232, 235, 238, 241, 244, 247, 250, 253, 256, 259, 262, 265, 268, 271, 274, 277, 280, 283, 286, 289, 292, 295, 298, 301, 304, 307, 310, 313, 316, 319, 322, 325, 328, 331, 334, 337, 340, 343, 346, 349, 352, 355, 358, 361, 364, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127, 130, 133, 136, 139, 142, 145, 148, 151, 154, 157, 160, 163, 166, 169, 172, 175, 178, 181, 184, 187, 190, 193, 196, 199, 202, 205, 208, 211, 214, 217, 220, 223, 226, 229, 232, 235, 238, 241, 244, 247, 250, 253, 256, 259, 262, 265, 268, 271, 274, 277, 280, 283, 286, 289, 292, 295, 298, 301, 304, 307, 310, 313, 316, 319, 322, 325, 328, 331, 334, 337, 340, 343, 346, 349, 352, 355, 358, 361, 364, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92, 95, 98, 101, 104, 107, 110, 113, 116, 119, 122,
            125, 128, 131, 134, 137, 140, 143, 146, 149, 152, 155, 158, 161, 164, 167, 170, 173, 176, 179, 182, 185, 188, 191, 194, 197, 200, 203, 206, 209, 212, 215, 218, 221, 224, 227, 230, 233, 236, 239, 242, 245, 248, 251, 254, 257, 260, 263, 266, 269, 272, 275, 278, 281, 284, 287, 290, 293, 296, 299, 302, 305, 308, 311, 314, 317, 320, 323, 326, 329, 332, 335, 338, 341, 344, 347, 350, 353, 356, 359, 362, 365, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117, 120, 123, 126, 129, 132, 135, 138, 141, 144, 147, 150, 153, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 186, 189, 192, 195, 198, 201, 204, 207, 210, 213, 216, 219, 222, 225, 228, 231, 234, 237, 240, 243, 246, 249, 252, 255, 258, 261, 264, 267, 270, 273, 276, 279, 282, 285, 288, 291, 294, 297, 300, 303, 306, 309, 312, 315, 318, 321, 324, 327, 330, 333, 336, 339, 342, 345, 348, 351, 354, 357, 360, 363, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127, 130, 133, 136, 139, 142, 145, 148, 151, 154, 157, 160, 163, 166, 169, 172, 175, 178, 181, 184, 187, 190, 193, 196, 199, 202, 205, 208, 211, 214, 217, 220, 223, 226, 229, 232, 235, 238, 241, 244, 247, 250, 253, 256, 259, 262, 265, 268, 271, 274, 277, 280, 283, 286, 289, 292, 295, 298, 301, 304, 307, 310, 313, 316, 319, 322, 325, 328, 331, 334, 337, 340, 343, 346, 349, 352, 355, 358, 361, 364, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127, 130, 133, 136, 139, 142, 145, 148, 151, 154, 157, 160, 163, 166, 169, 172, 175, 178, 181, 184, 187, 190, 193, 196, 199, 202, 205, 208, 211, 214, 217, 220, 223, 226, 229, 232, 235, 238, 241, 244, 247, 250, 253, 256, 259, 262, 265, 268, 271, 274, 277, 280, 283, 286, 289, 292, 295, 298, 301, 304, 307, 310, 313, 316, 319, 322, 325, 328, 331, 334, 337, 340, 343, 346, 349, 352, 355, 358, 361, 364, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92, 95, 98, 101, 104, 107, 110, 113, 116, 119, 122, 125, 128, 131, 134, 137, 140, 143, 146, 149, 152, 155, 158, 161, 164, 167, 170, 173, 176, 179, 182, 185, 188, 191, 194, 197, 200, 203, 206, 209, 212, 215, 218, 221, 224, 227, 230, 233, 236, 239, 242, 245, 248, 251, 254, 257, 260, 263, 266, 269, 272, 275, 278, 281, 284, 287, 290, 293, 296, 299, 302, 305, 308, 311, 314, 317, 320, 323, 326, 329, 332, 335, 338, 341, 344, 347, 350, 353, 356, 359, 362, 365, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117, 120, 123, 126, 129, 132, 135, 138, 141, 144, 147, 150, 153, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 186, 189, 192, 195, 198, 201, 204, 207, 210, 213, 216, 219, 222, 225, 228, 231, 234, 237, 240, 243, 246, 249, 252, 255, 258, 261, 264, 267, 270, 273, 276, 279, 282, 285, 288, 291, 294, 297, 300, 303, 306, 309, 312, 315, 318, 321, 324, 327, 330, 333, 336, 339, 342, 345, 348, 351, 354, 357, 360, 363, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127, 130, 133, 136, 139, 142, 145, 148, 151, 154, 157, 160, 163, 166, 169, 172, 175, 178, 181, 184, 187, 190, 193, 196, 199, 202, 205, 208, 211, 214, 217, 220, 223, 226, 229, 232, 235, 238, 241, 244, 247, 250, 253, 256, 259, 262, 265, 268, 271, 274, 277, 280, 283, 286, 289, 292, 295, 298, 301, 304, 307, 310, 313, 316, 319, 322, 325, 328, 331, 334, 337, 340, 343, 346, 349, 352, 355, 358, 361, 364, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127, 130, 133, 136, 139, 142, 145, 148, 151, 154, 157, 160, 163, 166, 169, 172, 175, 178, 181, 184, 187, 190, 193, 196, 199, 202, 205, 208, 211, 214, 217, 220, 223, 226, 229, 232, 235, 238, 241, 244, 247, 250, 253, 256, 259, 262, 265, 268, 271, 274, 277, 280, 283, 286, 289, 292, 295, 298, 301, 304, 307, 310, 313, 316, 319, 322, 325, 328, 331, 334, 337, 340, 343, 346, 349, 352, 355, 358, 361, 364, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92, 95, 98, 101, 104, 107, 110, 113, 116, 119, 122, 125, 128, 131, 134, 137, 140, 143, 146, 149, 152, 155, 158, 161, 164, 167, 170, 173, 176, 179, 182, 185, 188, 191, 194, 197, 200, 203, 206, 209, 212, 215, 218, 221, 224, 227, 230, 233, 236, 239, 242, 245, 248, 251, 254, 257, 260, 263, 266, 269, 272, 275, 278, 281, 284, 287, 290, 293, 296, 299, 302, 305, 308, 311, 314, 317, 320, 323, 326, 329, 332, 335, 338, 341, 344, 347, 350, 353, 356, 359, 362, 365, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117, 120, 123, 126, 129, 132, 135, 138, 141, 144, 147, 150, 153, 156, 159, 162, 165, 168, 171, 174, 177, 180]
    new_days = convert(days)
    final_rel_hum = []
    final_precip = []
    final_wind = []
    for i in range(len(new_days)):
        final_rel_hum.append(rel_hum[new_days[i]])
        final_precip.append(precip[new_days[i]])
        final_wind.append(wind[new_days[i]])

    very_final_precip = []
    for j in range(len(final_precip)):
        if(j > 6):
            val = final_precip[j-1] + final_precip[j-2] + final_precip[j-3] + final_precip[j-4] +final_precip[j-5] + final_precip[j-6] + final_precip[j-7]
            very_final_precip.append(val)
        else:
            very_final_precip.append(final_precip[j] * 7)

    final_precip = very_final_precip

    return final_precip, final_rel_hum, final_wind


def convert(days):
    converted = []
    start = date(2001, 3, 8)
    year_to_leap = {2001: False, 2002: False, 2003: False, 2004: True, 2005: False, 2006: False, 2007: False, 2008: True, 2009: False, 2010: False,
                    2011: False, 2012: True, 2013: False, 2014: False, 2015: False, 2016: True, 2017: False, 2018: False, 2019: False}
    month_to_days = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31,
                     6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

    year = 2001
    for i in range(len(days)):
        num = days[i]
        if(i != 0 and num < days[i-1]):
            year = year + 1
        if(year_to_leap[year]):
            month_to_days[2] = 29
        else:
            month_to_days[2] = 28

        month = 1
        while(num > month_to_days[month]):
            num = num - month_to_days[month]
            month = month + 1
        new_date = date(year, month, num)
        diff = (new_date - start).days
        converted.append(diff)
    return converted


def write_corn_days(years, dates, corn):
    year_to_leap = {2001: False, 2002: False, 2003: False, 2004: True, 2005: False, 2006: False, 2007: False, 2008: True, 2009: False, 2010: False,
                    2011: False, 2012: True, 2013: False, 2014: False, 2015: False, 2016: True, 2017: False, 2018: False, 2019: False}
    vals = dict()
    for i in range(2001, 2020):
        vals[i] = dict()
        if(year_to_leap[i]):
            for j in range(0, 367):
                vals[i][j] = 0
        else:
            for j in range(0, 366):
                vals[i][j] = 0
    for k in range(len(years)):
        vals[years[k]][dates[k]] = 1

    f = open('bar_data.p', "wb")
    pickle.dump(vals, f)
    f.close()


def create_numpy_array():
    array = np.array([0.0])
    array = np.delete(array, 0)
    return array


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


if __name__ == '__main__':
    main()
