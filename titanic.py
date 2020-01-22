# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# Pandas options
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Acquiring data

train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')
combine = [train_df, test_df]

##print(train_df.columns.values)
##print(train_df.head())
##train_df.info()
##print('_'*40)
##test_df.info()

##print(train_df.describe())

## This includes text and option columns only
## print(train_df.describe(include=['O']))

## Groups by passenger class to start with, then gets
## the average and sorts it by Survived (sort values not really needed)
## Higher class = higher chance of survival
##print(
##    train_df[['Pclass', 'Survived']].groupby(
##        ['Pclass'], as_index=False).mean().sort_values(
##            by='Survived',
##            ascending=False)
##    )
##
##
#### Females = more likely to survive
##print(
##    train_df[['Sex', 'Survived']].groupby(
##        ['Sex'], as_index=False).mean().sort_values(
##            by='Survived',
##            ascending=False)
##    )
##
#### Slightly stranger
##print(
##    train_df[['SibSp', 'Survived']].groupby(
##        ['SibSp'], as_index=False).mean().sort_values(
##            by='Survived',
##            ascending=False)
##    )
##
#### Slightly stranger
##print(
##    train_df[['Parch', 'Survived']].groupby(
##        ['Parch'], as_index=False).mean().sort_values(
##            by='Survived',
##            ascending=False)
##    )

# Getting into visualisation now

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
plt.show()

