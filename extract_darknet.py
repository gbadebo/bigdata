import os.path

tags = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB","class"]

domain = ["nulled", "blackhat"]	
	
def extractProduct(filename, start, end):
	with open(filename, 'r') as content_file:
    		content = content_file.read()	
		product = content[start:end]
		return product


def checkdarkfile(filename):
	datafile  = d+ "/tokenized/buy/" + "0-initiator{0}.txt.tok".format(filename)
	if os.path.isfile(datafile):
		return datafile
	datafile  = d+ "/tokenized/sell_verified/" + "0-initiator{0}.txt.tok".format(filename)
	if os.path.isfile(datafile):
		return datafile

	datafile  = d+ "/tokenized/other/" + "0-initiator{0}.txt.tok".format(filename)
	if os.path.isfile(datafile):
		return datafile
        datafile  = d+ "/tokenized/sell_unverified/" + "0-initiator{0}.txt.tok".format(filename)
	if os.path.isfile(datafile):
		return datafile
	return ""

def checkdarkfile_parsed(filename):
	datafile  = d+ "/parsed/buy/" + "0-initiator{0}.txt.tok".format(filename)
	if os.path.isfile(datafile):
		return datafile
	datafile  = d+ "/parsed/sell_verified/" + "0-initiator{0}.txt.tok".format(filename)
	if os.path.isfile(datafile):
		return datafile

	datafile  = d+ "/parsed/other/" + "0-initiator{0}.txt.tok".format(filename)
	if os.path.isfile(datafile):
		return datafile
        datafile  = d+ "/parsed/sell_unverified/" + "0-initiator{0}.txt.tok".format(filename)
	if os.path.isfile(datafile):
		return datafile
	return ""

def checkhackforumfile(filename):
	datafile  = d+ "/tokenized/all_nocurr_noprem/" + "{0}".format(filename)
	if os.path.isfile(datafile):
		return datafile
	datafile  = d+ "/tokenized/premium_sellers_section/" + "{0}".format(filename)
	if os.path.isfile(datafile):
		return datafile

	return ""
def checkhackforumfile_parsed(filename):
	datafile  = d+ "/parsed/all_nocurr_noprem/" + "{0}".format(filename)
	if os.path.isfile(datafile):
		return datafile
	datafile  = d+ "/parsed/premium_sellers_section/" + "{0}".format(filename)
	if os.path.isfile(datafile):
		return datafile

	return ""



annotationfiles = ["/annotations/test-annotations.txt"]
products = set()
for d in domain:
	f = open(d + annotationfiles[0])
	lines = f.readlines()

	for l in lines:
                p = l.replace("{","").replace("}","").replace("[","").replace("]","")
		data = p.split()
		filename = data[0] 
		start = int(data[2])

		end = int(data[3])
                datafile  = d+ "/tokenized/" + "0-initiator{0}.txt.tok".format(filename)
                if d == "darkode":
			datafile  = checkdarkfile(filename)
		if d == "hackforums":
			datafile  = checkhackforumfile(filename)
		if os.path.isfile(datafile):
			product = extractProduct(datafile.format(filename), start,end)
			products.add(product.strip())

print products

import nltk
from nltk.corpus import stopwords
mystopword = set(stopwords.words('english'))
def extract_features(datafile):
	f = open(datafile)
	lines = f.readlines()
	
	allfeatures = []
	for l in lines:
                data = l.split()
		if len(data) > 4:
			word = data[1]
			tagtype = data[4]
			pos = data[6] #
			mylen = len(word)
                        if word in mystopword:#this was bad very bad for results
				continue
			ptype = data[7]
			features = listofzeros = [0] * (len(tags))
			if word in products or tagtype == "NN" or tagtype == "NNP" or tagtype == "NNS"or tagtype =="NNPS": #or tagtype == "NNS" or tagtype == "NNP":
				features[len(features)-1] = 1
				
			else:
				features[len(features)-1] = 0
			if tagtype in tags:
				index = tags.index(tagtype)
				features[index] = 1
			if ptype in tags:
				index = tags.index(ptype)
				features[index] = 1
			features[len(features)-2] = pos
			features[len(features)-3] = mylen
			allfeatures.append(features)
			
	return allfeatures



import pandas as pd
for d in domain:
	f = open(d + annotationfiles[0])
	lines = f.readlines()
        dfeatures = []
	for l in lines:
                p = l.replace("{","").replace("}","").replace("[","").replace("]","")
		data = p.split()
		filename = data[0] 
		start = int(data[2])

		end = int(data[3])
                datafile  = d+ "/parsed/" + "0-initiator{0}.txt.tok".format(filename)
                if d == "darkode":
			datafile  = checkdarkfile_parsed(filename)
		if d == "hackforums":
			datafile  = checkhackforumfile_parsed(filename)
		if os.path.isfile(datafile):
			filefeatures = extract_features(datafile)
			dfeatures.extend(filefeatures)
	
	df = pd.DataFrame(dfeatures,columns=tags)
	
	df.to_csv(d + "features_yi.csv", sep=',')
	



	
