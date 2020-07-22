import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import csv
import xlrd


def main():
	filename = 'init_data_1.xlsx' #'init_data_1_2.xlsx'
	wb = xlrd.open_workbook(filename)
	sheet = wb.sheet_by_index(0)
	f = open('harvest_landmarks.csv', "r")
	reader = csv.reader(f)

	years, dates, corn, soybeans = csv_date_conversion(reader)
	years_xl, dates_xl, pm10 = xl_date_conversion(sheet)
	final_corn = []
	final_soybeans = []

	for i in range(len(years_xl)):
		day_check = dates_xl[i]
		indices = [j for j, obj in enumerate(dates) if obj == day_check]
		found = False
		for k in range(len(indices)):
			if(years[indices[k]] == years_xl[i]):
				#final_corn.append(corn[indices[k]])
				#final_soybeans.append(soybeans[indices[k]])
				if(corn[indices[k]] == 0):
					final_corn.append(0)
				else:
					final_corn.append(1)

				if(soybeans[indices[k]] == 0):
					final_soybeans.append(0)
				else:
					final_soybeans.append(1)
				found = True
				break
		if(not found):
			final_corn.append(0)
			final_soybeans.append(0)

	df = pd.DataFrame({'Year': years_xl,'Day': dates_xl,'PM10': pm10,'Corn': final_corn, 'Soybean': final_soybeans})

	mod = smf.ols(formula='PM10 ~ C(Year) + C(Day) + Corn + Soybean', data=df)
	res = mod.fit()
	print(res.summary())
	f.close()

def xl_date_conversion(sheet):
	years = []
	days = []
	pm10 = []
	for i in range(1,sheet.nrows):
		if i == 0:
			continue
		raw_date = sheet.cell_value(i,3)
		raw_data = float(sheet.cell_value(i,4))
		raw_date_spl = raw_date.split('/')
		month = int(raw_date_spl[0])
		day = int(raw_date_spl[1])
		year = int(raw_date_spl[2])
		date_int = date_to_int(year,month,day)

		years.append(year)
		days.append(date_int)
		pm10.append(raw_data)

	return years,days,pm10


def csv_date_conversion(reader):
	is_start = True
	dim = dict()
	years = []
	dates = []
	corn_l= []
	soybeans_l = []
	for row in reader:
		if(is_start):
			is_start = False
			continue
		date = row[0]
		corn = int(row[1])
		soybean = int(row[2])

		date_split = date.split('-')
		#print(date_split)
		year = int(date_split[0])
		if year not in dim:
			dim[year] = dict()
		month = int(date_split[1])
		day = int(date_split[2])
		date_int = date_to_int(year,month,day)
		dim[year][date_int] = (corn,soybean)

	for year in dim:
		x = (list(dim[year].keys()))[::-1]
		y1 = []
		y2 = []
		for day in dim[year]:
			y1.append(dim[year][day][0])
			y2.append(dim[year][day][1])
		y1 = y1[::-1]
		y2 = y2[::-1]
		vals = np.arange(x[0],x[len(x)-1]+1)
		new_corn = np.interp(vals,x,y1)
		new_soybeans = np.interp(vals,x,y2)

		for i in range(len(vals)):
			years.append(year)
			dates.append(vals[i])
			corn_l.append(new_corn[i])
			soybeans_l.append(new_soybeans[i])
	return years,dates,corn_l,soybeans_l

def date_to_int(year,month,day):
	year_to_leap = {2001:False,2002:False,2003:False,2004:True,2005:False,2006:False,2007:False,2008:True,2009:False,2010:False,
	2011:False,2012:True,2013:False,2014:False,2015:False,2016:True,2017:False,2018:False,2019:False}
	month_to_days = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
	if(year_to_leap[year]):
		month_to_days = {1:31,2:29,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
	day_ct = 0
	for i in range(0, month - 1):
		day_ct = day_ct + month_to_days[i+1]
	day_ct = day_ct + day
	return day_ct
	

if __name__ == '__main__':
	main()