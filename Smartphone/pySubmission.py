import csv
with open('submission.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='|')
	with open('submission2.csv', 'wb') as csvfile2:	
		writer = csv.writer(csvfile2, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for row in reader:
			writer.writerow(row)