import numpy 
filename = 'harsh.csv'
raw_data = open (filename, 'rt')
data = numpy.loadtxt(raw_data, delimiter = ",")
for i in range (0,20):
	print(data[i][4])
	i = i+1