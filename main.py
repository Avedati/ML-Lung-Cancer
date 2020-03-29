import os
import warnings
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings('ignore')

# Models tested: Decision Tree, Random Forest, KNN, Linear Reg., Log. Reg., SVM

def read_data():
	entries = []
	with open(os.path.join('data', 'lung-cancer.txt')) as fp:
		lines = fp.read().split('\n')
		entries = [line.split(',') for line in lines]
		for y in range(len(entries)):
			for x in range(len(entries[y])):
				if entries[y][x] == '?':
					mean = 0
					n = 0
					for i in range(len(entries)):
						if x < len(entries[i]) and (type(entries[i][x]) in [int, float] or entries[i][x].isdigit()):
							mean += float(entries[i][x])
							n += 1
					if n > 0:
						mean /= n
					entries[y][x] = mean
				elif type(entries[y][x]) == str:
					if entries[y][x].isdigit():
						entries[y][x] = float(entries[y][x])
	i = 0
	while i < len(entries):
		if len(entries[i]) != len(entries[1]):
			entries.pop(i)
			continue
		i += 1
	return entries

entries = read_data()
X = np.array([entry[1:] for entry in entries])
y = np.asarray([float(entry[0]) for entry in entries], dtype='float64')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

models = [DecisionTreeClassifier(), RandomForestClassifier(), KNeighborsClassifier(), LinearRegression(), LogisticRegression(), SVC()]
scores = []

for model in models:
	model.fit(X_train, y_train)
	scores.append(model.score(X_test, y_test))

notOverfittedScores = filter(lambda k: k != 1.0, scores)
maxIndex = scores.index(max(notOverfittedScores))
print('------------- Done -------------')
print('Best Model (based on R^2 score):')
print('  ' + str(models[maxIndex].__class__.__name__))
print('  -> score: ' + str(scores[maxIndex]))
