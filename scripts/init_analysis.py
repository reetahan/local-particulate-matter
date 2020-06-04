import numpy as np
import xlrd
from datetime import date
import matplotlib.pyplot as plt

def main():
	file = 'init_data_1_2.xlsx'
	dates,values = preprocess_gen(file)
	analyze_gen(dates,values)
	months,values = preprocess_avgs(file)
	analyze_avgs(months,values)


def preprocess_gen(filename): 
	dates = np.array([0])
	dates = np.delete(dates,0)
	values = np.array([0.0])
	values = np.delete(values,0)

	MISSING_VALUE = -999
	wb = xlrd.open_workbook(filename) 
	sheet = wb.sheet_by_index(0) 
	
	for i in range(1,sheet.nrows):
		raw_date = sheet.cell_value(i,3)

		processed_date = date_to_int(raw_date)
		processed_value = sheet.cell_value(i,4)
		if(processed_value == MISSING_VALUE):
			continue

		dates = np.append(dates, processed_date)
		values = np.append(values, processed_value)
		
		''' restrict to the first year
		if(processed_date > 365):
			break
		'''
	
	return dates,values

def date_to_int(date_string):
	START_DATE = date(2001,3,8)

	month_str = date_string[0:2]
	day_str = date_string[3:5]
	year_str = date_string[6:]
	
	month = int(month_str)
	day = int(day_str)
	year = int(year_str)

	cur_date = date(year,month,day)
	days_since_start = (cur_date - START_DATE).days 
	return days_since_start

def analyze_gen(dates,values):
	plt.plot(dates,values)
	plt.title('PM10 Density in Bondville Over Time')
	plt.xlabel('Days (since 3/8/2001)')
	plt.ylabel('PM10 Density (μg/m^3)')
	plt.show()

def preprocess_avgs(filename): 
	months = np.array(['January','February','March','April','May','June','July','August','September','October','November','December'])
	averages = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
	counts = np.array([0,0,0,0,0,0,0,0,0,0,0,0])

	MISSING_VALUE = -999
	wb = xlrd.open_workbook(filename) 
	sheet = wb.sheet_by_index(0) 
	
	for i in range(1,sheet.nrows):
		raw_date = sheet.cell_value(i,3)
		month = int(raw_date[0:2]) - 1 
		value = sheet.cell_value(i,4)
		
		if(value == MISSING_VALUE):
			continue

		averages[month] = averages[month] + value
		counts[month] = counts[month] + 1

	for j in range(len(averages)):
		averages[j] = averages[j]/counts[j]
	
	return months,averages


def analyze_avgs(dates,values):
	plt.bar(dates,values)
	plt.title('Average PM10 Density in Bondville Per Month')
	plt.xlabel('Month')
	plt.ylabel('Average PM10 Density (μg/m^3)')
	plt.show()

if __name__ == '__main__':
	main()