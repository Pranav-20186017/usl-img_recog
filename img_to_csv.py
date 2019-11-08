import pandas as pd
from PIL import Image
import os
import numpy as np

columns = [i for i in range(10304)]
columns.append('Label')

df = pd.DataFrame(columns=columns)
rows = 0

for direc in os.listdir():
	if direc.startswith('s'):
		print('Adding ' + direc)
		print('--------------')

		for file in os.listdir('./' + direc):
			im = Image.open('./' + direc + '/' + file)
			x = np.array(im.getdata())
			x = x.tolist()
			x.append(int(direc.replace('s', '')))
			df.loc[rows] = x
			rows += 1  

df.to_csv('Dataset.csv')
