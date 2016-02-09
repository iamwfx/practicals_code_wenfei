import pandas as pd
import math as m
import numpy as np
import scipy.stats as stats
import time

from rdkit import Chem
from rdkit.Chem import Descriptors

def main():
	start_time = time.clock()

	# Read in train and test as Pandas DataFrames
	df_train_all = pd.read_csv("train_small.csv")

	testMol = Chem.MolFromSmiles(df_train_all.sort('gap').head().smiles.values[0])

	## Leaving out magic functions and functions from the fr_ category that seem to produce all zeros
	functions = [i for i in dir(Descriptors) if not (i.startswith("__") or i.startswith("_") or i.startswith("fr_"))] 
	for each in functions:
		df_train_all[each] =0
	
	## Get array of converted smiles molecules
	smiles =[Chem.MolFromSmiles(x) for x in df_train_all.smiles.values]
	
	## Iterate over all the functions from our list
	for each in functions:
		print each
		
		## Call the function on each of our molecules, unless an exception is raised
		item = getattr(Descriptors,each)
		if callable(item):
			try: 
				feature = [item(x) for x in smiles]
				# print feature
				df_train_all.ix[:,each] = feature

			except AttributeError: 
				print "AttributeError"
			except ValueError: 
				print "ValueError"

	
	print "Runtime is ", time.clock() - start_time, "seconds"
	print df_train_all.TPSA
	# df_train_subset.to_csv('test.csv')



if __name__ == "__main__":
    # execute only if run as a script
    main()
            