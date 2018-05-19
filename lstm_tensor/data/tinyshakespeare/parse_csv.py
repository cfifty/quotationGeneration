import csv

f = open('ciceroquotes.csv','rb')
reader = csv.reader(f)   
cicero_quotes = [x[0] for x in reader]


with open('cicero.txt','wb') as f:
	for quote in cicero_quotes:
		f.write("Cicero: \n")
		f.write(quote + '\n')
		f.write('\n')
