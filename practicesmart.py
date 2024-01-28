#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from scipy import sparse
x=np.array([[1,2,0,3],[0,4,5,6]])
print('x:{}'.format(x))
print("CSR format: {}".format(sparse.csr_matrix(x)))
print("Converting to array: {}".format(sparse.csr_matrix(x).toarray()))
print("Data can be grabbed from np array:{}".format(sparse.csr_matrix(x).data))
print("No. of non-zero elements are:  {}".format(sparse.csr_matrix(x).count_nonzero()))


# In[3]:


from scipy import sparse
eye=np.eye(3) #2d numpy array with diagonal of ones and zeros everywhere else
print('eye: {}'.format(eye))


# In[2]:


import numpy as np
from scipy import sparse
data=np.ones(3)
print(data)
row_indices=np.arange(3)
column_indices=np.arange(3)
eye_coo=sparse.coo_matrix((data,(row_indices,column_indices)))
print("COO: {}".format(eye_coo))
print(eye_coo.toarray())


# In[6]:


#Using either inline/notebook or plt.show() to see the graph

get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt

#Dividing the interval [-10,10] into 100 points here (see notepad for more)
x=np.linspace(-10,10,100)
plt.plot(x)
plt.show()

print("Values with in interval also include in sine fuction",np.sin(-10),np.sin(10))

#for each value present in x there will be sine function implemented on them
y=np.sin(x)
plt.plot(x,y,marker='x')


# In[9]:


import pandas as pd
from IPython.display import display
data={
    'Name':["AB","CD","EF"],
    'Location':["New York","Kya be ji","Kaun hein re tu"],
    'Age':[20,22,40]
}

data_pandas=pd.DataFrame(data)

print(data_pandas)
print("\n\n")
print("Display using display function:")
display(data_pandas)

print("Now just printing which satisfies the condition if satisfied then true else false:")
print(data_pandas.Age>21)

print("\n\n")
print(data_pandas[data_pandas.Age>21])
#Like SQL query
display(data_pandas[data_pandas.Age>25])


# In[12]:


pip install mglearn


# In[14]:


import pandas as pd
import numpy as np
import mglearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris_dataset=load_iris()


X_train,X_test,y_train,y_test= train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)


# Assuming iris_dataframe and y_train are defined correctly

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)


pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',hist_kwds={'bins': 3}, s=60, alpha=.8, cmap=mglearn.cm3)


# In[1]:


#How to read a csv file using Panda
import pandas as pd
url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

names=['sepal-length','sepal-width','petal-length','petal-width','class']
dataset=pd.read_csv(url, names=names)

print(dataset.shape)


# In[12]:


print(dataset[:30])

print(dataset.head(30))


# In[13]:


print(dataset.describe())


# In[25]:


#as we are pandas so we can use describe or groupby
print(dataset.groupby('class').size())
print(dataset.groupby('class').describe())
print(dataset['class'].value_counts())


# In[3]:


import matplotlib.pyplot as plt
dataset.plot(kind="box",subplots=True,layout=(2,3),sharex=False,sharey=False)

#here kind="hist" for histogram
dataset.plot.hist()
plt.show()


# In[37]:


dataset.plot(kind="box",subplots=True,layout=(2,3),sharex=False,sharey=True)
plt.show()


# In[8]:


dataset.plot(kind="box",layout=(2,3),sharex=False,sharey=False)
plt.show()


# In[5]:


dataset.plot(kind="line",layout=(2,2),sharex=False,sharey=False)
plt.show()


# In[6]:


dataset.plot(kind="line",subplots=True,layout=(2,2),sharex=False,sharey=False)
plt.show()


# In[48]:


dataset.hist()
plt.show()


# In[11]:


#for mutlivariate plot like relation ship between attributes of dataset can be see using graph
pd.plotting.scatter_matrix(dataset)
plt.show()


# In[53]:


print(dataset.values)


# In[54]:


print(dataset.values[:,0:4])


# In[55]:


from sklearn.model_selection import train_test_split
X=dataset.values[:,0:4]
y=dataset.values[:,4]

X_train,X_test,y_train,y_test= train_test_split(X , y, random_state=0)


# In[67]:


from sklearn.tree import DecisionTreeClassifier


model_using_decisiontree= DecisionTreeClassifier()
model_using_decisiontree.fit(X_train,y_train)

print("Accuracy on training set is {}".format(model_using_decisiontree.score(X_train,y_train)))
print("Accuracy on test set is {}\n\n".format(model_using_decisiontree.score(X_test,y_test)))

print("These are model predicted values:")
print()
print(model_using_decisiontree.predict(X_test))
print("\n\n")
print("NOW lets see the actual Target value for each:\n\n",y_test)


# In[13]:


#Demo on statistics(not all just few basics)
import pandas as pd

df=pd.DataFrame({
    'Name':["AB","CD","EF","Noor","Villan","MyLife"],
    'Location':["New York","Kya be ji","Kaun hein re tu","A Human","I am Villan of my Life","Song"],
    'Age':[20,22,40,30,32,10],
    'Salary':[50000,53000,100000,44000,75000,50000]
})
print(df.head()) #will prints first five rows of the table frame

print("Mean of Salary: ",df['Salary'].mean())
print("Median of Salary: ",df['Salary'].median())
print("Mode of Salary: ",df['Salary'].mode())
print()
print("MIN: ",df['Salary'].min())
print("MAX: ",df['Salary'].max())


# In[16]:


#For particular Column we can show graph instead of just displaying the Values

salary=df['Salary']

salary.plot.hist(title="Salary Distribution",color="grey",bins=25)
plt.axvline(salary.mean(),color="red",linestyle="--",linewidth=1)
plt.axvline(salary.max(),color="green",linestyle="--",linewidth=1)
plt.axvline(salary.median(),color="violet",linestyle="--",linewidth=1)

plt.text(85000,1.8,'--Mean', color='r',fontsize=13)
plt.text(85000,1.5,'--Max value', color='g')
plt.text(85000,1.25,'--Median',color='violet',fontsize=12)


plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.show()


# In[1]:


import numpy as np
import matplotlib.pyplot as plt

# Generate positively skewed data
data_pos_skew = np.random.gamma(1, 2, 1000)

# Generate negatively skewed data
data_neg_skew = 10 - np.random.gamma(1, 2, 1000)

# Plot the positively skewed distribution
plt.subplot(1, 2, 1)
plt.hist(data_pos_skew, bins=30, color='skyblue')
plt.title("Positively Skewed Distribution")

# Plot the negatively skewed distribution
plt.subplot(1, 2, 2)
plt.hist(data_neg_skew, bins=30, color='lightgreen')
plt.title("Negatively Skewed Distribution")

# Adjust the layout
plt.tight_layout()

# Display the plot
plt.show()


# In[4]:


import numpy as np
import matplotlib.pyplot as plt

# Generate positively skewed data
data_pos_skew = np.random.gamma(1, 2, 1000)

# Generate negatively skewed data
data_neg_skew = 10 - np.random.gamma(1, 2, 1000)

# Plot the positively skewed distribution as a curve
plt.subplot(1, 2, 1)
plt.plot(np.sort(data_pos_skew), np.linspace(0, 1, len(data_pos_skew)), color='skyblue')
plt.title("Positively Skewed Distribution (Curve)")

# Plot the negatively skewed distribution as a curve
plt.subplot(1, 2, 2)
plt.plot(np.sort(data_neg_skew), np.linspace(0, 1, len(data_neg_skew)), color='lightgreen')
plt.title("Negatively Skewed Distribution (Curve)")

# Adjust the layout
plt.tight_layout()

# Display the plot
plt.show()


# In[5]:


#Dice
#Creating a set(as it will remove duplicates) of 6 numbers from 1 to 6, a dice

from itertools import product as p

n_dices=2 #number of dices

A={1,2,3,4,5,6}
B={1,2,3,4,5,6}

total_outcomes= list(set(p(A,B)))
print("Total outcomes of rolling 2 dices:\n")
print(total_outcomes)
    #Or
print("\n\nAnother method but same result:\n")
print(list(set(p(A, repeat=n_dices))))      #This will repeat the number of permutations to A only based on repeat value


# In[6]:


#Sum is 3
favourable_outcomes=[]

for each_outcome in total_outcomes:
    x,y = each_outcome
    if (x+y)%3==0:
        favourable_outcomes.append(each_outcome)
print(favourable_outcomes,"\nLength is :",len(favourable_outcomes))
print("Probability is :\n",len(favourable_outcomes)/len(total_outcomes))


# In[2]:


import pandas as pd

# Create a DataFrame
data = {'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]}
df = pd.DataFrame(data)
print(df)
# Access a specific row using iloc
row = df.iloc[1]
print(row)


# Access a specific element using iloc
element = df.iloc[2, 0]
print(element)
print()
print("kaun hein re tu")

#retrieving all rows of column zero, i.e at zero column taking all rows data means whole complete column vales
print(df.iloc[:,0])

#retrieving first 2(0 and 1 rows) rows of column zero, i.e at zero column(first) taking first 2 values.
print(df.iloc[:2,0])

#taking all columns values in first row, means whole row value
print(df.iloc[0,:])

print(df.iloc[0,:2])


print("HI")
print(df.iloc[0:2])

print("h")
#Using LOC()
print(df.loc[2])
print(df.loc[:,'B'])

print("\n hiii")
df.set_index('A')
print(df.iloc[:,0])
print(df.loc[2])

#after using inplace parameter which will modify the current dataframe instead of creating a new dataframe
#default value of inplace is false which will create a new one and not modify the current dataframe.

df.set_index('A', inplace=True)
print(df)
print(df.iloc[:,0])
print(df.loc[2])  #once we set_index only we can request the row with value 2, which searches in index column which is set as A column here
print(df.loc[3])


# In[34]:


import numpy as np
import matplotlib.pyplot as plt

a=np.linspace(1,10,100)

x=np.sin(a)
y=np.cos(a)

plt.subplot(2,2,1)
plt.plot(x)
plt.show()

plt.subplot(2,2,3)
plt.plot(y)


# plt.tight_layout()
plt.show()


# In[31]:


import numpy as np
import matplotlib.pyplot as plt

a=np.linspace(1,10,100)

x=np.sin(a)
y=np.cos(a)

# plt.subplot(2,2,1)
plt.plot(x)

# plt.subplot(2,2,3)
plt.plot(y)


# plt.tight_layout()
plt.show()


# In[30]:


import numpy as np
import matplotlib.pyplot as plt

a=np.linspace(1,10,100)

x=np.sin(a)
y=np.cos(a)

# plt.subplot(2,2,1)
plt.plot(x)
plt.show()

# plt.subplot(2,2,3)
plt.plot(y)


# plt.tight_layout()
plt.show()


# In[49]:


import matplotlib.pyplot as plt
import numpy as np

# Sample data for plotting
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(2*x)
y4 = np.cos(2*x)

# Create a 2x2 grid of subplots
plt.subplot(2, 2, 1)
plt.plot(x, y1)
plt.title('Plot 1: sin(x)')

plt.subplot(2, 2, 2)
plt.plot(x, y2)
plt.title('Plot 2: cos(x)')

plt.subplot(2, 2, 3)
plt.plot(x, y3)
plt.title('Plot 3: sin(2x)')

plt.subplot(2, 2, 4)
plt.plot(x, y4)
plt.title('Plot 4: cos(2x)')

# Adjust layout and display the plots
# plt.tight_layout()
plt.show()


# In[51]:


import matplotlib.pyplot as plt
import numpy as np

# Sample data for plotting
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(2*x)
y4 = np.cos(2*x)

# Create a 2x2 grid of subplots
plt.subplot(2, 2, 1)
plt.plot(x, y1)
plt.title('Plot 1: sin(x)')

plt.subplot(2, 2, 2)
plt.plot(x, y2)
plt.title('Plot 2: cos(x)')

plt.subplot(2, 2, 3)
plt.plot(x, y3)
plt.title('Plot 3: sin(2x)')

plt.subplot(2, 2, 4)
plt.plot(x, y4)
plt.title('Plot 4: cos(2x)')

# Adjust layout and display the plots
plt.tight_layout()
plt.show()


# In[4]:


import matplotlib.pyplot as plt
import numpy as np

# Sample data for plotting
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create a figure with a single subplot
plt.figure(figsize=(8, 4))#By giving this figure size,we can specify height and width alongs with labels for each particular plot...
plt.plot(x, y1, label='sin(x)',linestyle="dotted")
#plt.show() if you write this line then it will run commands related to plot for above statements , for below they will be executed as seperate plots...once u run this in various way u will get idea
plt.plot(x, y2, label='cos(x)')
plt.title('Plot of sin(x) and cos(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()   #with out this statement the plot-labels named will not get printed.

# Display the plot
plt.show()


# ## NAIVE BAYES

# In[3]:


#Implementing Naive Bayes Algorithm using User Defined code i.e without using inbuilt functions like scikit-learn library

import csv
import math
import random


# In[4]:


#for loading csv file

def loadcsv(filename):
    lines= csv.reader(open('pima-indians-diabetes.csv'))
    dataset= list(lines)
    #now converting them to float
    
    for i in range(len(dataset)):
        dataset[i]= [float(x) for x in dataset[i]]
    
    return dataset


# In[5]:


#For Spliting dataset (note: not using any inbuilt function) with ratio value entered

#here splitratio is like percentage of dataset
def splitDataset(dataset, splitratio):
    train_size= int( len(dataset) * splitratio ) #say if splitraito is 0.6 and dataset is 10, 
                                        #it means train dataset should be 60% percent of dataset or 0.6*10= 6 random rows
    train_set=[]
    
    duplicate_dataset= list(dataset)  #as we take random rows from dataset and remove those for next iterations, 
                                    #so we don't want to change the original dataset
    while len(train_set) < train_size:
        index=random.randrange(len(duplicate_dataset))
        #print('index',index)
        train_set.append(duplicate_dataset.pop(index))
        #print("After pop operation , the duplicate dataset contains:")
        #print(duplicate_dataset)
    return [train_set,duplicate_dataset]   #now duplicate_dataset contains rows/data which are not in train_set i.e it is test_set
        


# In[6]:


#Getting targetvalue or class of rows in dataset, means getting rows of same output value
#here target values are 0 and 1 ,so getting rows whose target value is 0 and storing all those in dictionary with key name 0
#similarly for 1

def seperateByClass(dataset):
    seperated={}
    for i in range(len(dataset)):
        values=dataset[i]
        if values[-1] not in seperated:  #if target value is not present we are initializing it
            seperated[values[-1]]= []
        seperated[values[-1]].append(values)  #appending each and every row containing the target_value same as the key name as target_value in dictionary.
    print("class wise seperated datarows according to the same target value:")
    for i in seperated:
        print(i,seperated[i])
    return seperated


# In[7]:


def mean(numbers):
    return sum(numbers)/ float(len(numbers))  #as numerator and denominator should be of same datatype na


# In[8]:


#Calculating StandardDeviation
def stdev(numbers):
    x_mean=mean(numbers)
    #now using variance formula
    variance= sum([pow( x-x_mean, 2 ) for x in numbers]) / float(len(numbers)-1)   #always for subset, 
                                                                #calculating variance we should take denominator length(n)-1
    #square root of variance is standard deviation
    return math.sqrt(variance)


# In[32]:


#Here we are doing mean,standarddeviation of each column,data of same target value rows
def summarize(dataset):
    s=dataset[0][-1]
    print("Dataset is:\n",dataset)
    #zip will take first value from each row and store them in attribute variable here, then will take second value from each row and so on same process till last
    summaries=[ (mean(attribute),stdev(attribute)) for attribute in  zip(*dataset)]  #example see down
    del summaries[-1]#deleting the target value ...
    print("Summaries of each column(mean and standard deviation) of target value: ",s)
    print(summaries)
    return summaries


#     dataset = [[1, 2, 3],
#            [4, 5, 6]]
#     zip(*dataset) gives [(1, 4), (2, 5), (3, 6)]
#     [(2.5, 1.5),   # Mean and stdev of attribute 1: (1+4)/2 = 2.5, stdev([1, 4]) = 1.5
#      (3.5, 1.5),   # Mean and stdev of attribute 2: (2+5)/2 = 3.5, stdev([2, 5]) = 1.5
#      (4.5, 1.5)]   # Mean and stdev of attribute 3: (3+6)/2 = 4.5, stdev([3, 6]) = 1.5


# In[10]:


def summarizeByClass(dataset):
    seperated= seperateByClass(dataset)  #getting dictionary with keys as target values and rows of the target value
    summaries={}
    for classvalue,instances in seperated.items():  #for each target value and its instances(rows) calculating mean and 
                                                    #standarddeviation for each column of same target value
        summaries[classvalue]= summarize(instances)  #and storing them in dictionary with key as target value and 
                                                    #value as list of each column mean and standarddeviation
    print("Summaries of each of both target values are:")
    print(summaries)
    return summaries


# In[24]:


def calculateProbability(x,mean,std):  #standard probability formula for Naive Bayes
    exponent= math.exp(-( math.pow(x-mean,2) / (2*math.pow(std,2)) ))
    return (1/ (math.sqrt(2* math.pi)*std)) * exponent


# In[16]:


#Same as we did in aiml ,here done in code
#Here we will perform probability of each target value against the input(test) row
#for each key,value in summaries dictionary we have key as target value and its associated mean,std values for each columns
#Now we will calculate the probability of each column against input row of each column
#and will multiply those calculated probability values
#here x is the value of specific column and taking parallely the values of mean,std of the same specific column and calculating probability
#here we will return probability of each target value(which is calculated by multiplying the individual column probabilities...)
#and returning the probabilities of each target value


def calculateClassProbabilites(summaries, inputvector):
    probabilities={}
    for classvalue, classsummaries in summaries.items():
        probabilities[classvalue]=1 #just initializing
        for i in range(len(classsummaries)):
            mean,std= classsummaries[i]
            x=inputvector[i]
            probabilities[classvalue] *= calculateProbability(x,mean,std)
    print("Probabilities of target values are:")
    print(probabilities)
    return probabilities


# In[12]:


#for each test case or input row we will run this function
#here will take the probability which is highest among the target values for that instance of input(test here) row
def predict(summaries, inputvector):
    #now we are doing for all test cases
    probabilities= calculateClassProbabilites(summaries, inputvector) #taking target value and probability of it against the input row
    bestlabel=None
    bestprob=-1
    for classvalue,probability in probabilities.items():  #this is a for loop for calculating the highest probability value 
                                                            #through if condition and storing in bestprob variable
        if bestlabel is None or probability > bestprob:
            bestprob= probability
            bestlabel= classvalue
            print(bestlabel,bestprob)
    print("Final best one is :")
    print(bestlabel, bestprob)
    return bestlabel               #returning the best target value the input test case row may lie or more suited


# In[13]:


#here will call predict function for each testcase and will store the result calculated for all test case in list.
def getPredictions(summaries, testset):
    predictions=[]
    for i in range(len(testset)):
        result= predict(summaries,testset[i])
        predictions.append(result)
    print("Predictions based on each testset:")
    print(predictions)
    return predictions


# In[14]:


#finding accuracy through no. of correct predictions did by total number of test case(or input rows) given
def getAccuracy(testset, predictions):
    correct=0
    for x in range(len(testset)):
        if testset[x][-1]==predictions[x]:
            correct+=1
    print("Correctly predicted are :")
    print(correct)
    return (correct/ float(len(testset)))*100.0


# In[33]:


#first split the data into train and test
#later group same target value rows means if 0,1 are target values then maintain list of rows whose value is 0,similarly to 1
#calculate mean and standard deviation of column data which are related to that particular target value
#means seperate rows with same target value and calculate mean,std on those

#Then for each test case to be predicted calculate probability of each column in test case against each column(parallely) of target value
#just multiply each column calculated probability which is probability value of that target value to the test case
#see which target value is having highest probability value, highest value is the output for that test case

#later calculate accuracy


def main():
    filename="pima-indians-diabetes.csv"
    splitratio= 0.67 #ratio of splitting data into train and test cases
    dataset = loadcsv(filename)
    trainingset,testset = splitDataset(dataset, splitratio)
    print("Spliting {0} rows into training {1} rows and testing {2} rows".format(len(dataset),len(trainingset),len(testset)))
    summaries=summarizeByClass(trainingset)
    predictions= getPredictions(summaries, testset)
    accuracy=getAccuracy(testset,predictions)
    print("Accuracy is:",accuracy)
main()


# ## Naive Bayes Using Inbuilt-Library Modules

# In[42]:





#Now doing the above logic using an inbuilt library which performs all of the above calculations


from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection

from sklearn.naive_bayes import GaussianNB


# In[20]:


dataset=datasets.load_iris()


print(dataset['feature_names'])

#Remember it is an array of dictionary with key,value pairs it is not pd dataframe so head and all methods cannot be used
print(dataset.keys())
print(dataset.values())
print()


# In[39]:


#Converting some data as train and some as test
#Remember X_train then X_test later only y_train and y_test

X_train,X_test,y_train,y_test = model_selection.train_test_split(dataset['data'],dataset['target'],random_state=0,train_size=0.65)
print(X_train.size,X_test.size)


# In[26]:


model=GaussianNB()
#Training the model using fit
model.fit(X_train,y_train)


# In[40]:


expected = y_test
predicted = model.predict(X_test)
for i,j in zip(expected,predicted):
    print("Expected is :",i,"   Predicted is:",j)


# In[41]:


print("Classfication report :")
print(metrics.classification_report(expected, predicted))
print()

print("Confusion Matrix:")
print(metrics.confusion_matrix(expected, predicted))


# In[44]:



#Another example of Naive Bayes using SocialNetwork.. dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[51]:


dataset= pd.read_csv('SocialNetworkAds.csv')
print(dataset.head())
print("\nNow Printing head method without curved braces:\n")
print(dataset.head)


# In[54]:


#Retrieving all rows of dataset and as well as target values in X,y

X= dataset.iloc[:,[2,3]]  #We are taking only Age and Estimated Salary
y= dataset.iloc[:,4]

print(X.head())
print()
print(y.head())


# In[69]:


#Training dataset and testing as well

from sklearn.model_selection import train_test_split

#Testsize will change the confusion matrix so be careful bhai
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size= 0.25)
print("X_train first 5 rows :")
print(X_train.head())
print()
print("X_test first 5 rows :")
print(X_test.head())


# In[70]:


#Preprocessing the dataset

from sklearn.preprocessing import StandardScaler

sc= StandardScaler()

X_train= sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("First 5 rows After Preprocessing using StandardScaler")
print("X_train first 5 rows:")
print(X_train[:5])
print()
print("X_test 5 rows:")
print(X_test[:3])


# In[71]:



from sklearn.naive_bayes import GaussianNB

classifier=GaussianNB()
#Training the model using fit
classifier.fit(X_train,y_train)


# In[72]:



expected= y_test
predicted= classifier.predict(X_test)


# In[74]:


from sklearn import metrics

print("Confusion matrix is :")
print(metrics.confusion_matrix(expected,predicted))


# In[5]:


import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create a Naive Bayes classifier
nb = GaussianNB()

# Fit the model
nb.fit(X, y)

# Make predictions
y_pred = nb.predict(X)


# Create a confusion matrix
cm = confusion_matrix(y, y_pred)
print(cm.shape)
print(cm,"\n")
# Plot the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Naive Bayes")
plt.colorbar()

# tick_marks = np.arange(len(iris.target_names))
# plt.xticks(tick_marks, iris.target_names)
# plt.yticks(tick_marks, iris.target_names)

# print(list(itertools.product(range(cm.shape[0]), range(cm.shape[1]))))
# thresh = cm.max() / 2
# for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#     print(j,i,format(cm[i, j], 'd'))
#     plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.show()


# In[17]:


import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create a Naive Bayes classifier
nb = GaussianNB()

# Fit the model
nb.fit(X, y)

# Make predictions
y_pred = nb.predict(X)


# Create a confusion matrix
cm = confusion_matrix(y, y_pred)
print(cm.shape)
print(cm,"\n")
# Plot the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Naive Bayes")
plt.colorbar()

tick_marks = np.arange(len(iris.target_names))
print("tick_marks\n",tick_marks,"\n")
plt.xticks(tick_marks, iris.target_names)
plt.yticks(tick_marks, iris.target_names)

print(range(cm.shape[0]))
print(list(range(cm.shape[0])),"\n")

print(list(itertools.product(range(cm.shape[0]), range(cm.shape[1]))))
thresh = cm.max() / 2
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    print(j,i,format(cm[i, j], 'd'))
    plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")  #as here we are using even dark blue where if we keep text as black color we cannot find so for those cells in array display if value is greater than thresh then white else black.

plt.tight_layout()
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.show()


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create a Naive Bayes classifier
nb = GaussianNB()

# Fit the model
nb.fit(X, y)

# Make predictions
y_pred = nb.predict(X)

# Create a confusion matrix
cm = confusion_matrix(y, y_pred)

# Define custom colormap
colors = ['lightblue', 'lightgreen', 'lightyellow']
cmap = ListedColormap(colors)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title("Confusion Matrix - Naive Bayes")
plt.colorbar()

tick_marks = np.arange(len(iris.target_names))
plt.xticks(tick_marks, iris.target_names)
plt.yticks(tick_marks, iris.target_names)

thresh = cm.max() / 2
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="black")

plt.tight_layout()
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.show()


# In[3]:


#Visualising training dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from matplotlib.colors import ListedColormap

# Load the Iris dataset
iris = load_iris()
X = iris.data[:,:2]  # Consider only the first two features for simplicity
y = iris.target

# Create a Naive Bayes classifier
nb = GaussianNB()

# Fit the model
nb.fit(X, y)

# Generate a grid of points to evaluate the decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

print("Data",X[:5],"Target ",y[:5])
print()
print("As we are drawing 2 columns or feautures, so one column/feature in x-axis and other column/feature in y-axis")
print("Now X axis min and max  , Y axis min and max")
print(x_min,x_max,y_min,y_max)
h = 0.02  # Step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
print("XX\t",xx,"\n","YY\t",yy,"\n")
print("SO X-axis horizantally increment and Y-axis vertically increment\n")

#Z = nb.predict(np.c_[xx.ravel(), yy.ravel(),np.zeros_like(xx.ravel()), np.zeros_like(yy.ravel())])
Z = nb.predict(np.c_[xx.ravel(), yy.ravel()])

print(xx[0:4],xx[-1])
print("\n\n")
print(yy[0:4],yy[-2],yy[-1])
print("Z\n",Z[0:4])

print(xx[50:54])
print(yy[50:54])

print(Z[40050])
print(Z[9050])
print(Z[10050])
# Define custom colormap
colors = ['blue', 'green', 'red']
cmap = ListedColormap(colors)

# Plot the decision boundaries
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k')
plt.contourf(xx, yy, Z.reshape(xx.shape), alpha=0.5, cmap=cmap)   #for sure see reason for reshape in file (Backupmlusingpython)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Naive Bayes Decision Boundaries")
plt.colorbar()
plt.text(7,4.05,'Kya bey',color='white')
# Create legend
legend_elements = [plt.scatter([], [], marker='o', color='blue', label='Class 0'),
                   plt.scatter([], [], marker='o', color='green', label='Class 1'),
                   plt.scatter([], [], marker='o', color='red', label='Class 2')]
plt.legend(handles=legend_elements)
#plt.legend()
plt.show()


# In[39]:



#Visualising training dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from matplotlib.colors import ListedColormap

# Load the Iris dataset
iris = load_iris()
X = iris.data[:,:2]  # Consider only the first two features for simplicity
y = iris.target

# Create a Naive Bayes classifier
nb = GaussianNB()

# Fit the model
nb.fit(X, y)

# Generate a grid of points to evaluate the decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

print("Data",X[:5],"Target ",y[:5])
print()
print("As we are drawing 2 columns or feautures, so one column/feature in x-axis and other column/feature in y-axis")
print("Now X axis min and max  , Y axis min and max")
print(x_min,x_max,y_min,y_max)
h = 0.02  # Step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
print("XX\t",xx,"\n","YY\t",yy,"\n")
print("SO X-axis horizantally increment and Y-axis vertically increment\n")

#Z = nb.predict(np.c_[xx.ravel(), yy.ravel(),np.zeros_like(xx.ravel()), np.zeros_like(yy.ravel())])
Z = nb.predict(np.c_[xx.ravel(), yy.ravel()])

print(xx[0:4],xx[-1])
print("\n\n")
print(yy[0:4],yy[-2],yy[-1])
print("Z\n",Z[0:4])

print(xx[50:54])
print(yy[50:54])

print(Z[40050])
print(Z[9050])
print(Z[10050])
# Define custom colormap
colors = ['blue', 'green', 'red']
cmap = ListedColormap(colors)

# Plot the decision boundaries
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k',label="C0")
plt.scatter([],[],color='green',label="C1")
plt.scatter([],[],color='red',label="C2")
plt.contourf(xx, yy, Z.reshape(xx.shape), alpha=0.5, cmap=cmap)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Naive Bayes Decision Boundaries")
plt.colorbar()
plt.text(7,4.05,'Kya bey',color='white')
# # Create legend
# legend_elements = [plt.scatter([], [], marker='o', color='blue', label='Class 0'),
#                    plt.scatter([], [], marker='o', color='green', label='Class 1'),
#                    plt.scatter([], [], marker='o', color='red', label='Class 2')]
# plt.legend(handles=legend_elements)
plt.legend()
plt.show()


# In[22]:


import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
# Define the colors for the colormap
colors = ['red', 'blue', 'green']

# Create a ListedColormap object
cmap = ListedColormap(colors)


# Load the Iris dataset
iris = load_iris()
X = iris.data  # Consider only the first two features for simplicity
y = iris.target
print(X[:5])
print(X[50:55])
print(X[101:106])

# Scatter plot with different colors for each category
plt.scatter(X[:, 2], X[:, 3], c=y, cmap=cmap)
plt.colorbar()
plt.show()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap)
plt.show()


# In[18]:



#If we are taking all columns/features instead of 2 columns as we did in above cases
#then use  zeros_like function

from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

# Load the Iris dataset
iris = load_iris()
X = iris.data     #Here we are taking all columns for ploting regions but taking only 2 columns in graph as 2-d so
y = iris.target
        
# Create a Gaussian Naive Bayes classifier
nb = GaussianNB()
nb.fit(X, y)  #here u have trained on all columns/features so even predict method should get all columns as we have trained here.

# Generate a grid of points to evaluate the decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = 0.02  # Step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = nb.predict(np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel()), np.zeros_like(yy.ravel())])

# Define custom colormap
colors = ['lightblue', 'green',"red"]
cmap = ListedColormap(colors)

# Plot the decision boundaries
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k')
# plt.contourf(xx, yy, Z.reshape(xx.shape), alpha=0.5, cmap=cmap)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Decision Boundaries for Naive Bayes')
plt.show()


# ## Linear Regression Model

# In[5]:


#Linear Regression Model
#Firstly we do with out any sklearn library, later will do using it and also preprocessing ,even train_test methods...


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#setting the default size for all plots, so that no need again to set plot figure size
plt.rcParams['figure.figsize']=(20.0,10.0)


# In[12]:


#reading data

data= pd.read_csv('headbrain.csv')
print(data.shape)
print(data.size)

print(data.head())


# In[32]:


#Storing in X and y

X=data['Head Size(cm^3)'].values
y=data['Brain Weight(grams)'].values


# In[33]:


#will do later preprocessing, now just plotting the graph and see which regression to choose whether linear relation between x-axis(data) and y-axis(target) then linear regression

plt.scatter(X,y,label="Scatter Plot")

plt.xlabel("Head Size(cm^3)")
plt.ylabel('Brain Weight(grams)')
plt.legend()
plt.show()


# In[22]:


#AS We can see from above it is kind of linear ration between Head Size(x) and Brain Weight(y) so,
#Use linear regression model


# In[34]:


#As we know that linear regression model is of type
#       y= b0 + b1x
#  b1 =(∑(xi - xmean)*(yi - ymean)) / (∑(xi - xmean)^2)
    #For calculating, b0=ymean - b1*xmean

xmean= np.mean(X)
ymean= np.mean(y)

n=len(X) #or len(y)

numerator=0
denominator=0
for i in range(n):
    numerator += (X[i] -  xmean) * (y[i] - ymean)
    denominator += (X[i] - xmean)**2
    
b1= numerator / denominator
b0= ymean - (b1*xmean)

print("So the equation is :")
print("\t","y=",b0,"+",b1,"*x")

print("Which states that based on x(Head Size) value we can predict y(Brain weight) value")


# In[39]:



#Now plotting the regressiong line


# xmax= np.max(X)
# xmin= np.min(X)

# #now just having various values of x, calculatin target y
# x= np.linspace(xmin, xmax, 100)


yp=[]

#for all values in X(data) what is our predicted y(target value)
for x in X:
    yp.append(b0 + b1*x)

plt.plot(X,yp,color="red",label="Regression line")
plt.scatter(X,y,label="Scatter Plot")

plt.xlabel("Head Size(cm^3)")
plt.ylabel('Brain Weight(grams)')
plt.legend()
plt.show()


# In[51]:


#Now as we got regression line ,how far this will fit the data overall
#lets find out using R-Squared value
#  R2= 1−(SSres / SStot)
#Where,
    #SSres is the sum of squared residuals (the differences between the actual y values and the predicted y values).
    #SStot is the total sum of squares (the differences between the actual y values and the mean of y values).

ssres=0
sstot=0

for i in range(n):
    ssres+= (y[i] -  yp[i])**2
    sstot+= (y[i] -  ymean)**2

r2= 1-(ssres / sstot)
print(r2)
#As value is around 0.64 which is good ,higher value of r2 better the relation ship between target(y) and data(x) or say feature relation ship
print("This shows that the relationship between Head Size(X-axis) and Brain Weight(Y-axis) variables or features explains {0:.2f} percentage of variation in data".format(r2*100))


# ## Linear Regression Model using Inbuilt-Library

# In[54]:


#Now will do using sklearn libraries for linear regression


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#reading data

data= pd.read_csv('headbrain.csv')
print(data.shape)
print(data.size)

print(data.head())


# In[55]:


#Storing in X and y

X=data['Head Size(cm^3)'].values
y=data['Brain Weight(grams)'].values


# In[56]:


#now just plotting the graph and see which regression to choose whether linear relation between x-axis(data) and y-axis(target) then linear regression

plt.scatter(X,y,label="Scatter Plot")

plt.xlabel("Head Size(cm^3)")
plt.ylabel('Brain Weight(grams)')
plt.legend()
plt.show()


# In[59]:


#Creating Regression Model

regmodel= LinearRegression()


#go down pakkkaaa for more examples
# The -1 means "whatever is needed" 
#firstargument is number of rows and second column in number of columns.
#.reshape(-1, 1) Purpose: The .reshape(-1, 1) operation is used to reshape the original 1D array into a 2D column vector. 
#The -1 argument in the .reshape() function indicates the number of rows which you want numpy to automatically determine the size of that dimension based on the length of the input array. 
#The 1 in .reshape(-1, 1) specifies that you want one column.

X=X.reshape((-1,1))  #

regmodel.fit(X,y)

y_pred= regmodel.predict(X)

mse= mean_squared_error(y, y_pred)

rsquare= r2_score(y , y_pred)

print(mse,rsquare)


# In[65]:


#Now will perform preprocessing,train_test....

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



#reading data

data= pd.read_csv('headbrain.csv')
print(data.shape)
print(data.size)

print(data.head())

X_train,X_test,y_train,y_test= train_test_split(data['Head Size(cm^3)'].values,data['Brain Weight(grams)'].values,random_state=0,test_size=0.2)

X_train=X_train.reshape(-1,1)
X_test=X_test.reshape(-1,1)

#preprocessing

scaler = StandardScaler()


# Transform the training data
X_train= scaler.fit_transform(X_train)

# Transform the test data using the same scaler
X_test_scaled = scaler.transform(X_test)


regmodel= LinearRegression()

regmodel.fit(X_train,y_train)

y_pred= regmodel.predict(X_test_scaled)


# In[72]:


#Now plotting regression lines and data points of tested data and predicted 
plt.plot(X_test_scaled,y_pred,color="red",label="Regression line")
plt.scatter(X_test_scaled,y_test,label="Scatter Plot")

plt.xlabel("Head Size(cm^3)")
plt.ylabel('Brain Weight(grams)')
plt.legend()
plt.show()


#The train data points scatter and plot of regression line of tested and predicted here, so that the regression line how much it fits with trained data points
plt.plot(X_test_scaled,y_pred,color="red",label="Regression line")
plt.scatter(X_train,y_train,label="Scatter Plot")

plt.xlabel("Head Size(cm^3)")
plt.ylabel('Brain Weight(grams)')
plt.legend()
plt.show()


# In[68]:



mse= mean_squared_error(y_test, y_pred)

rsquare= r2_score(y_test , y_pred)

print(mse,rsquare)
print("This shows that the relationship between Head Size(X-axis) and Brain Weight(Y-axis) variables or features explains {0:.2f} percentage of variation in data".format(r2*100))


# In[7]:


#Reshape

import numpy as np
arr = np.arange(1, 7)
reshaped_arr = np.reshape(arr, (2, 3))
print(reshaped_arr)
r2=arr.reshape(-1,1)
r2


# ## KNN

# In[2]:


#KNN- Non Linear Model
#Firstly will do without any inbuilt sklearn knn module, later will do using sklearn library(as shown down while scrolling)



import numpy as np
import matplotlib.pyplot as plt

# Generate random data
np.random.seed(0)
X = np.random.randn(20, 2)
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])

new_point = np.array([0.5, 0.5])  # New point to predict

print(X)

#plotting the data along with random data with red color, see in diagram
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='orange', label='Class 1')
plt.scatter(new_point[0], new_point[1], color='red', label='New Point')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

#Euclidean Distance calculating through this method, see once pakka axis=1 , axis=2 in chatgpt or Backupmlusingpython.txt
distances = np.linalg.norm(X - new_point, axis=1)
print(distances)

k = 5  # Number of neighbors

#based on distances for each point in X(data) to the new point plotted, sorting them in A.O and taking only first k (neighbours) points
nearest_indices = np.argsort(distances)[:k]
print("Nearest indices are:",nearest_indices)
print(np.argsort(distances)[:k],sorted(distances)[:k])

#the points which are nearer to the new point, we are grabing those class labels whether of class 0 or 1.
k_nearest_classes = y[nearest_indices]
print(y[nearest_indices])

#finding how many times each class label(0,1) repeated or can say which class label is in major times repeated near new point
predicted_class = np.bincount(k_nearest_classes).argmax()

print(np.bincount(k_nearest_classes))


#plotting the points in 'x' which are nearer to the new point as calculated above
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='orange', label='Class 1')
plt.scatter(new_point[0], new_point[1], color='red', label='New Point')
plt.scatter(X[nearest_indices][:, 0], X[nearest_indices][:, 1], color='green', label='Nearest Neighbors', s=60,marker='x')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()


#the value(class label) which repeated most is the class label for the new point
print(f"Predicted Class for the New Point: {predicted_class}")


# In[22]:


#If X is a 2D array (matrix) in this case, then specifying axis=2 would result in an error because there is no third axis in a 2D matrix.

#If X is a 3D array (e.g., a stack of matrices), then specifying axis=2 would calculate the Euclidean norm along the third axis, which is typically along the "depth" of the data. 

X = np.array([[1, 2],
              [3, 4],
              [5, 6],
              [7, 8]])
new_point = np.array([4, 5])

diff= X - new_point
# difference will be:
# [[-3, -3],
#  [-1, -1],
#  [ 1,  1],
#  [ 3,  3]]

distances = np.linalg.norm(X - new_point, axis=1)
print(distances)


# In[5]:


#here X is 3D array so axis should be 2.
X = np.array([[[1, 2],
               [3, 4],
               [5, 6]],
              [[7, 8],
               [9, 10],
               [11, 12]]])

new_point = np.array([4, 5])

distances = np.linalg.norm(X - new_point, axis=2)
print(X-new_point,distances)


# In[20]:


#example np.bincount()
counts = np.bincount([2, 1, 3, 2, 2,5])
print(counts)

#in output we have indexes from 0 starting and till 5 as largest value present in input array is 5 so, first value is 0 as zero(0) is not present in input array, next value in output is 1 as one(1) is present in array one time, next vale in output is 3 as two(2) is present 3 times in input array.


# In[18]:


#random.randn() will give array numbers generated randomly with mean 0 and variance of 1

import numpy as np

# Generate an array of random numbers from a standard normal distribution
random_numbers = np.random.randn(5,2)  # Generates 5 random numbers with mean 0 and variance 1.

print(random_numbers)


# ## KNN using Inbuilt-Library

# In[26]:


#KNN - using inbuilt modules from sklearn library

from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


#importing data
data=load_iris()

#will take only first 2 features of dataset

X_train,X_test,y_train,y_test= train_test_split(data['data'][:,:2],data['target'],random_state=42,test_size=0.3)


#now preprocessing
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test= sc.transform(X_test)

print(X_train[:5],X_test[:5])
print(y_train[:5],y_test[:5])


# In[27]:


#Now knn model

k=3
model=KNeighborsClassifier(n_neighbors=k)

model.fit(X_train,y_train)


# In[37]:


print(data['target_names'])


# In[28]:


for i,j in zip(model.predict(X_test),y_test):
    print("Expected : ",j,"   Predicted : ",i)


print(model.predict(X_test))
#now lets take one test data point and plot it
new_point=np.array(X_test[0])
new_point=new_point.reshape(-1,2)
print(new_point)
new_point_original= y_test[0]

#plotting
#we have to seperately plot the points of labels(target)
#for this we have first plot points with class label(target) Setosa-0, then next plot points of Virginica-1, ...
plt.figure(figsize=(10,8))
plt.scatter(X_train[y_train==0][:, 0], X_train[y_train==0][:, 1], color="blue",label="Class Setosa")
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color="green",label="Class Versicolor")
plt.scatter(X_train[y_train==2][:, 0], X_train[y_train==2][:, 1], color="lightblue",label="Class Virginica")
plt.scatter(new_point[:,0],new_point[:,1],color="red",label="New Point to be Predicted")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()
plt.show()


# In[29]:


print("The point to be predicted is:",new_point)


new_point_predicted= model.predict(new_point)
print(new_point[0] in X_train)
# if new_point[0] in X_train:
#     distances = np.linalg.norm(X_train - new_point, axis=1)
#     print(y_train[np.argsort(distances)[:k][0]])
# else:
# #nearest data points to the new point
# distances, indices= model.kneighbors(new_point,n_neighbors=6) 
# print(indices)


# #as when we do X_train[indices] where indices value is [[1 1 1 1 2]] 
# #when give apply X_train on each value of [[1 1 1 1 2]]
# #say for X_train[1] is [-0.21352735 -0.58900572] so result will be [[ [-0.21352735 -0.58900572], .... ]]
# print(X_train[indices[0][1:]],y_train[indices[0][1:]],distances[0][1:])  #here we are doing slicing from first list inside indices,we are removing first value inside list as
# #as it considers itself as the nearest distance so.
    #Euclidean Distance calculating through this method, see once pakka axis=1 , axis=2 in chatgpt or Backupmlusingpython.txt
distances = np.linalg.norm(X_train - new_point, axis=1)
print(X_train)
print(distances)

#based on distances for each point in X(data) to the new point plotted, sorting them in A.O and taking only first k (neighbours) points
indices = np.argsort(distances)[1:k+1]
print("Nearest indices are:",indices)
print(np.argsort(distances)[1:k+1],sorted(distances)[1:k+1])


plt.figure(figsize=(10,8))
plt.scatter(X_train[y_train==0][:, 0], X_train[y_train==0][:, 1], color="blue",label="Class Setosa")
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color="green",label="Class Versicolor")
plt.scatter(X_train[y_train==2][:, 0], X_train[y_train==2][:, 1], color="lightblue",label="Class Virginica")
plt.scatter(new_point[:, 0],new_point[:, 1],color="red",label="New Point to be Predicted")
# plt.scatter(X_train[indices][0][1:,0],X_train[indices][0][1:,1],color="orange",s=40,label="Nearest points")
plt.scatter(X_train[indices][:,0],X_train[indices][:,1],color="orange",s=40,label="Nearest points")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()
plt.show()

print(new_point_original,new_point_predicted)

yp= model.predict(X_test)
print("Accuracy score is: ",accuracy_score(y_test, yp))


# In[18]:


#For deleting row or column in pandas dataframe

import pandas as pd

# Create a sample DataFrame
data = {'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]}

df = pd.DataFrame(data)
print("Before deleting:\n",df)
# Drop a row by specifying its index
df = df.drop(1, axis=0)  

print("After deleting row:\n",df)


# In[17]:


import pandas as pd

# Create a sample DataFrame
data = {'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]}

df = pd.DataFrame(data)
print("Before deleting:\n",df)

# Drop a column by specifying its name
df = df.drop('B', axis=1)  # 'B' is the name of the column to be dropped


print("After deleting column:\n",df)


# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 4, 1, 3, 5]

# Create a scatter plot with Seaborn
sns.scatterplot(x=x, y=y)

# Show the plot
plt.show()


# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Create some sample data
x = np.random.randn(100)

# Use Seaborn's distribution plot to create a histogram
sns.histplot(x, kde=True, color='skyblue')

# Customize the Seaborn plot (Matplotlib functions can be used here)
plt.title("Distribution Plot")
plt.xlabel("Values")
plt.ylabel("Frequency")

# Show the plot
plt.show()


# ## Categorical Data Handlation- LabelEncoder and OneHotEncoder

# In[165]:


#Handling Categorical Data present in the dataset
#ANOTHER METHOD IS ALSO THERE SEE GET_DUMMIES IN BELOW

#being used is Linear Regression Model on dataset which has categorical values present
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split


# In[166]:


#Reading Data
data= pd.read_csv('1000_Companies.csv')

print(data.head())


# In[167]:


#If we don't specify values then it will store even column names in first row and values in next rows, which again headache to
#convert the Categorical values into numerical values which we don't want as we will not use this or should care, 
#so keep values while parsing the data

X= data.iloc[:, :4].values  #X is numpy object see
y=data.iloc[:, 4].values

print(data.head(),"\n\n")
print(data.head())

print()

print(X[:5])
print("\n\n",y[:5])


# In[168]:


#Data visualization using seaborn or pyplot or both can be done
#as any how seaborn is built on top of pyplot library so.

sns.heatmap(data.corr(),annot=True)   #Or we can do using plt.plotting.scatter_matrix() method which should specify arguments externally,so...

#Here we can see that R&D Speed vs Profit is more correlation so we can classify any instance row with these two,
#Now lets see the plot and decide which model is best


# In[79]:


plt.scatter(X[:, 0][:90],y[:90],color="green")
plt.xlabel("R&D Speed")
plt.ylabel("Profit")
plt.show()

#As u can see for some amount of data plotted here is of linear type relationship so,
#Linear Regression is to be used


# In[7]:


#As we can see last column in X which has categorical values in it which are from [New York,California,Florida]
#Linear Regression will not understand anything apart from numbers so should convert them into numerical values.
from sklearn.compose import ColumnTransformer
print(X[:5,3])
labelencoder= LabelEncoder()
X[:, 3]= labelencoder.fit_transform(X[:, 3])

print(X[:5, 3]) #as we see anyhow our dataset of this column is not inherited or any meaningful order or ranking so,
#should use OneHotEncoder for sure

# onehotencoder= OneHotEncoder(categorical_features=3)   #categorical_features is not supported in new versions, 
#as we have to specify the OneHotEncoder to work/make affect only on Categorical Column or say text values columns 
#and let leave the remaining columns untouched we use below code for it

# X=onehotencoder.fit_transform(X).toarray()
# print(X[0])


one_hot_columns = [3]  # As only Column 3 is categorical

# Create transformers
transformers = [('one_hot', OneHotEncoder(), one_hot_columns)]

# Create and apply a column transformer
column_transformer = ColumnTransformer(transformers=transformers, remainder='passthrough')
X_encoded = column_transformer.fit_transform(X)
print("\n\n")
print(X_encoded)


# In[44]:


print(X_encoded[:5])


# In[8]:


#Now splitting the data

X_train,X_test, y_train,y_test= train_test_split(X_encoded,y,test_size=0.2,random_state=0)
print("X_Train: \n",X_train[:5])
print("y_Train: \n",y_train[:5])
print("\nX_Test:\n",X_test[:5])
print("y_Test: \n",y_test[:5])


# In[10]:


#As from graph we have seen how the data varies of first column to target variable

from sklearn.linear_model import LinearRegression

model= LinearRegression()
model.fit(X_train, y_train)


# In[13]:


y_pred= model.predict(X_test)

for i,j in zip(y_test, y_pred): 
    print("Expected value: ",i,"\t Predicted value: ",j)


# In[19]:


# y= mx+c for linear regression
# y= m1x1 + m2x2 + ... + c for multiple regression , where m1 is coefficient of column x1, m2 is coefficient of column x2...
        #c is the intercept

print("Co-efficients(m1, m2, ..,m6) in equation(y=m1x1 + m2x2 + m3x3 +...m6x6) are:")
print(model.coef_)

#As we have 6 columns(after doing label and onehot encoding) in this example, so 6 intercepts


# In[20]:


print("The intercept(c) in equation(y=m1x1+m2x2+....+c) is:")
print(model.intercept_)


# In[21]:


#Now r-Square calculation (performance measurement of model)
from sklearn.metrics import r2_score

print(r2_score(y_test, y_pred))


# In[1]:




#Reshape examples

import numpy as np

# Create a 1D array
arr = np.array([1, 2, 3, 4, 5])

# Reshape it into a column vector (2D)
column_vector = arr.reshape((-1, 1))

print("Original 1D Array:")
print(arr)

print("\nReshaped Column Vector:")
print(column_vector)


# In[4]:


#Reshape examples

import numpy as np

# Create a 1D array
arr = np.array([1, 2, 3, 4, 5])

# Reshape it into a column vector (2D)
column_vector = arr.reshape((-1, 2))

print("Original 1D Array:")
print(arr)

print("\nReshaped Column Vector:")
print(column_vector)


# In[9]:


#Reshape examples

import numpy as np

# Create a 1D array
arr = np.array([1, 2, 3, 4, 5,6])

# Reshape it into a column vector (2D)
column_vector = arr.reshape((-1, 2))

print("Original 1D Array:")
print(arr)

print("\nReshaped Column Vector:")
print(column_vector)


# In[10]:


#Reshape examples

import numpy as np

# Create a 1D array
arr = np.array([1, 2, 3, 4, 5,6])

# Reshape it into a column vector (2D)
column_vector = arr.reshape((2, 3))

print("Original 1D Array:")
print(arr)

print("\nReshaped Column Vector:")
print(column_vector)


# In[11]:


#Reshape examples

import numpy as np

# Create a 2D array
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6]])

# Flatten it into a 1D array
arr1d = arr2d.reshape(-1)  # The -1 means "whatever is needed"

print("Original 2D Array:")
print(arr2d)

print("\nFlattened 1D Array:")
print(arr1d)


# In[15]:


#Reshape examples

import numpy as np

# Create a 2D array
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6]])

# Flatten it into a 1D array
arr1d = arr2d.reshape(2,1) 

print("Original 2D Array:")
print(arr2d)

print("\n Reshaped 2D Array:")
print(arr1d)


# In[16]:


#Reshape examples

import numpy as np

# Create a 2D array
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6]])

# Flatten it into a 1D array
arr1d = arr2d.reshape(6,1)  

print("Original 2D Array:")
print(arr2d)

print("\n Reshaped 2D Array:")
print(arr1d)


# In[17]:


#Reshape examples

import numpy as np

# Create a 2D array
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6]])

# Flatten it into a 1D array
arr1d = arr2d.reshape(3,-1)  

print("Original 2D Array:")
print(arr2d)

print("\n Reshaped 2D Array:")
print(arr1d)


# In[19]:


#Reshape examples

import numpy as np

# Create a 2D array
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6]])

# Flatten it into a 1D array
arr1d = arr2d.reshape(-1,-1) 

print("Original 2D Array:")
print(arr2d)

print("\n Reshaped 2D Array:")
print(arr1d)


# ## Logistic Regression

# In[2]:


#Logistic Regression

#loading libraries, data sets,...
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import metrics
from sklearn.datasets import load_digits

from sklearn.linear_model import LogisticRegression


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# In[3]:


#Loading data

digits= load_digits()

print(digits.data.shape)
print(digits.target.shape)

print("\n",digits.data[:5],len(digits.data[0]))
print(digits.target[:5])


# In[9]:


#As u can see it is of 64 array , 
#The dataset has 64 features (8x8 pixels), and each feature contains an integer value representing the brightness of the corresponding pixel.
print(digits.data[0].reshape((8,8)))

#We will convert into 8x8 matrix, as the matrix of values which specify color shade impact like if value is 16 then gray..
#which will help to convert into image of appropriate numbers display


# In[152]:


#We reshaped as now u can see a matrix format of 8x8 and values typically range from 0 to 16, 
#where 0 is white (no ink) and 16 is black (fully inked). Values in between represent various shades of gray.

plt.figure(figsize=(30,9))
for i,(image,target) in enumerate(zip(digits.data[:5], digits.target[:5])):
    plt.subplot(2,3,i+1)
    print(target)
    plt.imshow(np.reshape(image,(8,8)), cmap="Reds" )
    plt.title("Target is: " +str(target),fontsize="27")
plt.tight_layout()
plt.show()


# In[4]:


#now splitting the data

X_train, X_test, y_train,y_test= train_test_split(digits.data,digits.target, test_size=0.23, random_state=2)
print(X_train.shape,y_train.shape)
print(X_test.shape, y_test.shape,"\n")

print("X_TRAIN:\n",X_train[:5])
print("y_train\n",y_train[:5])

print()

print("X_TEST:\n",X_test[:5])
print("y_test:\n",y_test[:5])


# In[79]:



model= LogisticRegression()

model.fit(X_train,y_train)
print(X_test,np.reshape(X_test[0],(1,64)),np.reshape(X_test[0],(1,-1)))
#As you are sending one particular row of dataset, if we retrieve it then it will be of 1D array,
#so converting to 2D array of all values in one row and all columns as required or say 64
print(model.predict(np.reshape(X_test[0],(1,-1))))


# In[5]:



model= LogisticRegression(solver='liblinear')

model.fit(X_train,y_train)

print(model.predict(np.reshape(X_test[0],(1,-1))))


# In[6]:


y_pred=model.predict(X_test)
print(y_pred,y_test,X_test)
accuracy_score= model.score(X_test,y_test)
print(metrics.accuracy_score(y_pred,y_test))
print("Accuracy score is: ",accuracy_score)


# In[107]:


print(metrics.confusion_matrix(y_test,y_pred))
cm=metrics.confusion_matrix(y_test,y_pred)


# In[119]:


plt.figure(figsize=(15,7))
sb.heatmap(cm,linewidth=.5,cmap="Blues_r",square=True,annot=True,fmt=".2f")  #See document if any doubts
plt.xlabel("Actual Label")3
plt.ylabel("Predicted Label")
plt.show()


# In[7]:


index=0
classifiedindex=[]
for original,predicted in zip(y_test,y_pred):
    if original==predicted:
        classifiedindex.append(index)
    index +=1
plt.figure(figsize=(10,22))
for plotindex, index in enumerate(classifiedindex[:10]):
    print(plotindex)
    plt.subplot(5,2,plotindex+1)
    plt.imshow(np.reshape(X_test[index], (8,8)), cmap="Greens")
    plt.title("Actual : {} , Predicted : {}".format(y_test[index],y_pred[index]))


# In[162]:


#An example of scatter plot and categorical plot using seaborn

import seaborn as sns
import matplotlib.pyplot as plt

# Load a sample dataset
tips = sns.load_dataset("tips")
print(tips[:5])
# Set the aesthetic style
sns.set_style("whitegrid")

# Create a scatter plot with hue (color by a categorical variable)
sns.relplot(x="total_bill", y="tip", data=tips, hue="sex", style="time", size="size")

# Add a title
plt.title("Scatter Plot of Tips")

# Show the plot
plt.show()


# In[163]:


sns.catplot(x="day", y="total_bill", data=tips, hue="sex", kind="box")


# ## Handling with Categorical Variable Using get_dummies instead of LabelEncoding and OneHotEncoding
# ## Seaborn countplot(), heatmap() and
# ## Pandas more and effective Inbuilt methods on Datasets

# In[78]:


#Importing modules

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


# In[55]:


data= pd.read_csv("titanic.csv")
# pd.set_option('display.max_columns', None)  # To display all columns
# pd.set_option('display.width', 1000)
from IPython.core.display import display, HTML

#Default width value is 75% ,so after using or changing the width data value to 100% or 200% and using to see the output of all columns in one row ,do change back to 75%
display(HTML('<style>.container {width: 200% !important}</style>' ))


# In[31]:


print(data.head())


# In[70]:


plt.figure(figsize=(10,6))
sns.heatmap(data.corr(),annot=True)


# ## Analyzing Data and Each Column
# 
# # Plotting graph of Survivours vs other columns which will help to get more about data

# In[17]:


#Now as you can see, some columns say PassengerID ,Name,Sex,Cabin which are not required as passengerid is nothing but Serial No., also some null values in Cabin, string values
#Where Logistic Regression will not understand Null values or string values

    sns.countplot(x="Survived",data=data)

#From this we can see that majority of them are not Survivours.


# In[18]:


#Let's see how many are male survivours and female survivours

sns.countplot(x="Survived", hue="Sex",data=data)

#From below we can see most survivours are female where most of the male population is non-survivours


# In[23]:


#Now lets see PClass which of first class, second class and third class are survivours

sns.countplot(x="Survived",data=data,hue="Pclass")
#From below we can see that most of the survivours are of First Class and most of the non-survivours are of Thrid Class


# In[25]:


#Now lets calculate Age Column like how many are young,children,old..

data['Age'].plot.hist()

#From below we can see that most of are young age(20-30)


# In[4]:


#Now lets calculate Fare Column

data['Fare'].plot.hist(bins=20,figsize=(10,5))


# In[6]:


#Now lets calculate SibSp Column

data['SibSp'].plot.hist()


# In[24]:


sns.boxplot(x="Pclass",y="Age",data=data)

#We conclude that Passengers traveling in first class are of more age and where as second and third class are less maybe still earning process..


# In[32]:


data.info()


# ## Analyzing and Eliminating Null Values

# In[8]:


#As you can see from above few columns like Age, Cabin.. are having Null Values

sns.heatmap(data.isnull())


# In[20]:


#On y-axis we can see that many values are there which makes the visualization not much effective, so lets remove those values as for large datasets it will still effect the visualization more.


sns.heatmap(data.isnull(), yticklabels=False,cmap='viridis')


# In[56]:


#As u can see more values of Cabin are null,
#Either we have to replace using fillna() method or remove(dropna() which will remove either rows(0) or columns(1) based on axis value
#(or)  drop() if whole column is of more null values and not usable to us) that column.


print(data.head())

data.drop("Cabin",axis=1,inplace=True)
print("\n\n")

print(data.head())


# In[57]:


#Removing less values of Null in data ,if more in one column then we should remove whole column
#Now removing any null values rows(which are present very few hereandthere) so that model have only numerical data

data.dropna(inplace=True)

data.info()


# In[13]:


#Now lets make a graph of if anywhere null values present in the dataset

sns.heatmap(data.isnull(),yticklabels=False,cmap='viridis')

#As we can see no null values present


# In[16]:


print("Null values count in each column respectively")
data.isnull().sum()


# In[17]:


data.head(5)


# In[58]:


#As You can see all null values are being removed
#But we still have string values which model will not understand,
#so must eliminate them apart from using other methods like labelencoding and onehotencoding

sex= pd.get_dummies(data['Sex'])
print(sex.head(5))
print("END1\n\n")

#OR, Use below method of mentioning more than one columns at once

s=pd.get_dummies(data, columns=['Sex'])
#This will automatically gets whole dataset and eliminates that columns mentioned and do onehotencoding
print(s.head(5))
print("END2\n\n")

#Original data
print(data)
print("END3\n\n")


# In[59]:


#As we can see above first method , when female value is 1 then male value is 0
#So instead of both columns we can take one column which defines other

sex= pd.get_dummies(data['Sex'], drop_first=True)    #Removing first column from result sex variable
print(sex.head(5))


# In[50]:


print(data.head(5))


# In[60]:


#As you can see the Embarked column is of values strings , so lets do that too

embarked= pd.get_dummies(data['Embarked'], drop_first=True)
print(embarked.head(20))


# In[61]:


pclass = pd.get_dummies(data['Pclass'], drop_first= True)
print(pclass.head(20))


# In[25]:


print(data.head(5))


# In[62]:


#As u can see we have created dummies for string type columns(instead of using labelencoding and onehotencoding),
#Lets concat them to the original data

data= pd.concat([data,pclass,sex,embarked], axis=1)   #axis =1 for column wise append
print(data.head(5))


# In[41]:


print(data.head(5))


# In[63]:


#Now lets remove the old columns from the new data as we have added new columns with no strings present
#Also removing string values of Name column and Ticket column

data.drop(['Pclass','Sex','Embarked','Name','Ticket'],axis=1, inplace=True)

print(data.head(5))


# In[65]:


#After writing train_test_split, press shift tab will get the detailed doccumnetation along with example.

X=data.drop("Survived",axis=1)
y=data["Survived"]

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=1)

print(X_train,y_train)


# In[67]:


model= LogisticRegression()


# In[68]:


model.fit(X_train,y_train)


# In[76]:


predicted=model.predict(X_test)
print(predicted)


# In[77]:


confusion_matrix(y_test, predicted)


# In[79]:


accuracy_score(y_test,predicted)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[25]:


#Demo of Dropping Rows(0) and Columns(1)
import pandas as pd
import numpy as np

# Create a sample DataFrame with missing values
data = {'A': [1, 2, np.nan, 4],
        'B': [5, np.nan, np.nan, 8],
        'C': [9, 10, 11, 12]}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Remove rows containing NaN values (default behavior as axis=0)
df_cleaned_rows = df.dropna()
print("\nDataFrame after removing rows with NaN values:")
print(df_cleaned_rows)

# Remove columns containing NaN values by specifying axis=1
df_cleaned_columns = df.dropna(axis=1)
print("\nDataFrame after removing columns with NaN values:")
print(df_cleaned_columns)


# ## sns.lmplot() similar to Scatter Matrix

# In[7]:


#sns.lmplot() see more about in Backup.

import seaborn as sns
import matplotlib.pyplot as plt

# Load a sample dataset
tips = sns.load_dataset("tips")
print(tips.head(),tips.values[0],tips.head(0))
# Create an lmplot
sns.lmplot(x="total_bill", y="tip",hue='sex', data=tips)

# Show the plot
plt.show()


# ## Categorical.from_codes()

# In[12]:


import pandas as pd

#Best to do with values as A,B,C.. and target as 0,1,2..
#as can convert string values to integer values.

# Define a list of categories and corresponding codes
data={"Values":[0, 1, 2,1,2],
"Target":['A', 'B', 'C']}

# Create a Categorical object from the codes and categories
cat = pd.Categorical.from_codes(data["Values"], categories=data["Target"])

# Now, you have a Categorical object
print(cat)


# In[64]:


import pandas as pd

# Define a list of categories and corresponding codes
categories = [0, 1, 2]
codes = [0, 1, 2, 2, 0, 0]  # Convert codes to integers

# Create a Categorical object from the codes and categories
cat = pd.Categorical.from_codes(codes, categories=categories)

# Now, you have a Categorical object
print(cat)


# ## Decision Tree
# ## Built Using Libraries
# ## Will do with out using Built in methods or modules

# In[1]:


#importing modules

import  numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


# In[26]:


data= pd.read_csv('Decision_Tree_ Dataset.csv')

print(data.head(),"\n",data.columns)


# In[27]:


#Splitting the data
X=data.values[:,0:4]
y=data.values[:, 4]
print(X[:5])
print(y[:5])

X_train,X_test,y_train,y_test= train_test_split(X,y,random_state=100, test_size=0.3)


# In[28]:


#Model(See more about in Backup file)

model= DecisionTreeClassifier(criterion='entropy',max_depth=3,min_samples_leaf=5, random_state=100)
model.fit(X_train, y_train)


# In[5]:


y_pred= model.predict(X_test)
for i,j in zip(y_test,y_pred):
    print("Actual Value is: ",i,"\tPredicted Value is: ",j)


# In[6]:


accuracy= accuracy_score(y_test,y_pred)
print("Accuracy is:\n\t",accuracy*100)
print(data.columns)


# In[7]:


plt.figure(figsize=(12, 8))
tree.plot_tree(model, filled=True, feature_names=data.columns[:3], class_names=data['Result'].unique())
plt.show()


# In[29]:


plt.figure(figsize=(12, 8))
tree.plot_tree(model, filled=True, feature_names=data.columns, class_names=data['Result'].unique())
plt.show()


# ## Random Forest (Made up of Multiple Decision trees, so as to get precise output)
# ## Another logic for Splitting Data

# In[30]:


import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import load_iris
from sklearn import tree


# In[45]:


#importing data

iris= load_iris()
print(iris)

#As u can see iris holds dictionary along with data,feature_names,descr etc not like Column values accessible, we can access only rows

data=pd.DataFrame(iris.data, columns=iris.feature_names)
print("\n\n")
print(data.head(5))


# In[46]:


print(iris.target[:5],"\n",iris.target_names)
print("\nNow zeros are replaces with setosa and ones with versicolour and two with virginica")

#To know more about Categorical.from_codes see above
data['species']= pd.Categorical.from_codes(iris.target, categories= iris.target_names)
data.head()


# In[47]:


#Now Splitting data using logic instead of inbuilt method

#Splitting data for train size as 75% of original data and test size of 25% data
#if is_train value is True then it comes into train set else it comes into test set
data['is_train']= np.random.uniform(0, 1, len(data)) <0.75

data.head()


# In[48]:


#Retrieving train set and test set using Condition if that column is true then it is train set
#NOTE: data[data['is_train']==True] or can use data[data.is_train==True]

# print(data[data.is_train==True][:5],"\n\nlkj")

train, test= data[data['is_train']==True], data[ data['is_train']==False]

print(train.head(5),"\n",test.head(5),train.size,test.size)


# In[49]:


#We must take only first four columns for training the model
features=data.columns[:4]
print(features)


# In[50]:


#Model

model=RandomForestClassifier(n_jobs=2,random_state=0, max_depth=3)

#As u can see species are having string values so converting them to numerical values
print(train['species'][:5],"\n")
print(pd.factorize(train['species']),"\n\n\n",pd.factorize(train['species'])[0][:5])

model.fit(train[features], pd.factorize(train['species'])[0])


# In[51]:


pred= model.predict(test[features])
print(pred[:5],pd.factorize(test['species'])[0][:5])


# In[52]:


print(accuracy_score(pd.factorize(test['species'])[0], pred))


# In[53]:


print(model.predict_proba(test[features])[:14])


# ## Crosstab

# In[57]:


pd.crosstab(test['species'],pred, rownames=['Actual Names'],colnames=['Predicted Names'])


# In[56]:


# Get all the decision trees from the random forest
estimators = model.estimators_
print(len(estimators))

# Number of trees to plot (e.g., first 3)
num_trees_to_plot = 3

# Plot the selected trees
for i in range(num_trees_to_plot):
    plt.figure(figsize=(10, 5))
    tree.plot_tree(estimators[i], filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
    plt.title(f'Decision Tree {i + 1}')
    plt.show()


# In[ ]:





# In[69]:


import pandas as pd

# Create a sample DataFrame
data = {
    'Gender': ['Male', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male'],
    'City': ['NY', 'LA', 'LA', 'NY', 'NY', 'LA', 'NY'],
    'Satisfaction': ['Satisfied', 'Dissatisfied', 'Satisfied', 'Satisfied', 'Dissatisfied', 'Satisfied', 'Dissatisfied']
}

df = pd.DataFrame(data)

# Create a cross-tabulation
cross_tab = pd.crosstab(df['Gender'], df['Satisfaction'])
print(cross_tab)
pd.crosstab(df['Gender'], df['Satisfaction'])


# In[65]:


#Factorize Method another example
import pandas as pd

# Sample data with categorical values
data = ['cat', 'dog', 'cat', 'fish', 'dog']

# Use pd.factorize() to encode categorical values
encoded_data, labels = pd.factorize(data)    #Or can access only data using slicing pd.factorize(data)[0]

# 'encoded_data' contains the integer-encoded values
# 'labels' contains the unique labels

# Let's print the results
print("Original Data:")
print(data)
print("\nEncoded Data:")
print(encoded_data)
print("\nLabels:")
print(labels)


# In[13]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import time

# Load the Iris dataset
data = load_iris()
X, y = data.data, data.target

# Create a Random Forest Classifier with n_jobs=1
rfc_1core = RandomForestClassifier(n_estimators=100, n_jobs=1)

# Create a Random Forest Classifier with n_jobs=-1 (use all available cores)
rfc_allcores = RandomForestClassifier(n_estimators=100, n_jobs=-1)

# Measure the time it takes to fit each model
start_time = time.time()
rfc_1core.fit(X, y)
end_time = time.time()
print("Training time with 1 core: {:.4f} seconds".format(end_time - start_time))

start_time = time.time()
rfc_allcores.fit(X, y)
end_time = time.time()
print("Training time with all available cores: {:.4f} seconds".format(end_time - start_time))


# In[3]:


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load the Iris dataset as an example
iris = load_iris()
X = iris.data
y = iris.target

# Create a Decision Tree classifier
clf = DecisionTreeClassifier(random_state=0)

# Fit the classifier to the data
clf.fit(X, y)

# Plot the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()


# In[4]:


from sklearn.tree import DecisionTreeClassifier

# Sample data for classification
X = [[25, 40000], [30, 60000], [35, 80000], [40, 75000], [45, 100000], [20, 30000]]


y = ['No', 'No', 'Yes', 'No', 'Yes', 'No']

# Create a Decision Tree classifier
clf = DecisionTreeClassifier()

# Fit the classifier to the data
clf.fit(X, y)

# Use the classifier to predict a new customer
new_customer = [[33, 70000]]
prediction = clf.predict(new_customer)
print("New customer will purchase:", prediction[0])


# In[6]:


import matplotlib.pyplot as plt
import numpy as np

# Generate non-linear data
X = np.linspace(0, 10, 50)
Y = X**2

plt.scatter(X, Y)
plt.plot(X, Y, color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Non-Linear Relationship')
plt.grid()
plt.show()


# In[25]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create a Decision Tree classifier with a low max_depth
clf_low_depth = DecisionTreeClassifier(max_depth=2)
clf_low_depth.fit(X, y)

# Create a Decision Tree classifier with a high max_depth
clf_high_depth = DecisionTreeClassifier(max_depth=None)  # No limit on depth
clf_high_depth.fit(X, y)

# Create subplots to visualize the trees
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
# Visualize the low-depth tree
tree.plot_tree(clf_low_depth, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
axes[0].set_title("Decision Tree (max_depth=2)")
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
# Visualize the high-depth tree
tree.plot_tree(clf_high_depth, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
axes[1].set_title("Decision Tree (max_depth=None)")

plt.show()


# In[37]:


# Create a Decision Tree classifier with a high max_depth and min_samples_leaf
clf_custom = DecisionTreeClassifier(criterion ="entropy",max_depth=None, min_samples_leaf=5)
clf_custom.fit(X, y)

# Visualize the custom tree
tree.plot_tree(clf_custom, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Decision Tree (max_depth=None, min_samples_leaf=5)")
plt.show()


# In[ ]:





# ## SVM(Support Vector Machine)
# ## Decision Function(analyzed using meshgrid) instead of prediction in SVM and Using Logic yyup and yydown

# In[1]:


#importing libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import svm

from sklearn.datasets import make_blobs


# In[2]:


#About centers parameter go some down for more about it
X,y= make_blobs(n_samples=40, centers=2, random_state=20)
print(X[:5],len(X[:,0]),len(X[:,1]))
print("\n",y[:5],len(y))


# In[3]:


#Model

clf= svm.SVC(kernel='linear', C=1)
clf.fit(X,y)


# In[4]:


plt.scatter(X[:,0],X[:,1],c=y,s=30,cmap=ListedColormap(['Yellow','Red']))
plt.show()


# In[111]:


newData= [[3,4],[5,6],[7,7]]
print(clf.predict(newData))


# In[6]:


#Decision function
plt.scatter(X[:,0],X[:,1],c=y,s=30,cmap=ListedColormap(['Yellow','Red']))

#Plot the Decision Function
ax= plt.gca()
xlim= ax.get_xlim()
ylim=ax.get_ylim()
print(ax)
print("\n",xlim,ylim)

#Creating a grid to evaluate model
xx= np.linspace(xlim[0], xlim[1], 30)
yy= np.linspace(ylim[0], ylim[1], 30)
print("xx and yy are:")
print(xx[:5],xx[-1],yy[:5],yy[-1])

YY,XX = np.meshgrid(yy,xx)
print("\nXX and YY are:",XX[:5],YY[:5])
xy= np.vstack([XX.ravel(), YY.ravel()]).T
print("\nxy are:",xy[:5])
Z=clf.decision_function(xy).reshape(XX.shape)
print("Z:",Z[:5])

#plotting the decision boundary and margins
ax.contour(XX, YY, Z, colors='k',levels=[-1,0,1], alpha=0.5, linestyles=['--','-','--'])

sv = clf.support_vectors_
plt.scatter(sv[:, 0], sv[:, 1], c='g', marker='x')

# ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],s=200,linewidth=1, facecolors='none')
plt.show()

print(clf.support_vectors_)


# In[91]:


#Decision function
plt.scatter(X[:,0],X[:,1],c=y,s=30,cmap=ListedColormap(['Yellow','Red']))

#Plot the Decision Function
ax= plt.gca()
xlim= ax.get_xlim()
ylim=ax.get_ylim()
print(ax)
print("\n",xlim,ylim)

#Creating a grid to evaluate model
xx= np.linspace(xlim[0], xlim[1], 30)
yy= np.linspace(ylim[0], ylim[1], 30)
print("xx and yy are:")
print(xx[:5],yy[:5])

YY,XX = np.meshgrid(yy,xx)
print("\n\n",XX[:5],YY[:5])
xy= np.vstack([XX.ravel(), YY.ravel()]).T
print("\n",xy[:5])
Z=clf.decision_function(xy).reshape(XX.shape)
print("Z:\n",Z[:5])

#plotting the decision boundary and margins
ax.contour(XX, YY, Z, colors='k',levels=[0], alpha=0.5, linestyles=['-'])

ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],s=100,linewidth=1, facecolors='none')
plt.show()


# In[92]:


#Decision function
plt.scatter(X[:,0],X[:,1],c=y,s=30,cmap=ListedColormap(['Yellow','Red']))

#Plot the Decision Function
ax= plt.gca()
xlim= ax.get_xlim()
ylim=ax.get_ylim()
print(ax)
print("\n",xlim,ylim)

#Creating a grid to evaluate model
xx= np.linspace(xlim[0], xlim[1], 30)
yy= np.linspace(ylim[0], ylim[1], 30)
print("xx and yy are:")
print(xx[:5],yy[:5])

YY,XX = np.meshgrid(yy,xx)
print("\n\n",XX[:5],YY[:5])
xy= np.vstack([XX.ravel(), YY.ravel()]).T
print("\n",xy[:5])
Z=clf.decision_function(xy).reshape(XX.shape)
print("Z:\n",Z[:5])

#plotting the decision boundary and margins
ax.contour(XX, YY, Z, colors='k',levels=[0,1], alpha=0.5, linestyles=['-','--'])

ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],s=100,linewidth=1, facecolors='none')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[27]:


import numpy as np
from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt

# Create a simple dataset
X, y = datasets.make_classification(n_samples=100, n_features=2, n_classes=2, n_clusters_per_class=1, n_redundant=0)

# Initialize the SVM classifier
clf = svm.SVC(kernel='linear')

# Fit the SVM model on the data
clf.fit(X, y)

# Get the separating hyperplane parameters
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(X[:, 0].min(), X[:, 0].max())
yy = a * xx - (clf.intercept_[0]) / w[1]

# Get support vectors
sv = clf.support_vectors_

# Create a plot
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.plot(xx, yy, 'k-')
plt.scatter(sv[:, 0], sv[:, 1], c='r', marker='x')
plt.title('SVM with Best Hyperplane')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


# In[28]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

# Create a simple dataset
X, y = datasets.make_classification(n_samples=100, n_features=2, n_classes=2, n_clusters_per_class=1, n_redundant=0)

# Initialize the SVM classifier
clf = svm.SVC(kernel='linear')

# Fit the SVM model on the data
clf.fit(X, y)

# Get the separating hyperplane parameters
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(X[:, 0].min(), X[:, 0].max())
yy = a * xx - (clf.intercept_[0]) / w[1]

# Get support vectors
sv = clf.support_vectors_

# Plot the data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')

# Plot the hyperplane
plt.plot(xx, yy, 'k-')

# Plot the support vectors
plt.scatter(sv[:, 0], sv[:, 1], c='r', marker='x')

# Plot the margin lines
margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
yy_down = yy - np.sqrt(1 + a**2) * margin
yy_up = yy + np.sqrt(1 + a**2) * margin
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

# Draw lines along the support vectors
for i, vector in enumerate(sv):
    plt.plot([vector[0], vector[0] + w[0]], [vector[1], vector[1] + w[1]], 'g-')

plt.title('SVM with Best Hyperplane and Support Vectors')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


# In[98]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

# Create a simple dataset
X, y = datasets.make_classification(n_samples=100, n_features=2, n_classes=2, n_clusters_per_class=1, n_redundant=0)

# Initialize the SVM classifier
clf = svm.SVC(kernel='linear')

# Fit the SVM model on the data
clf.fit(X, y)

# Get the separating hyperplane parameters
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(X[:, 0].min(), X[:, 0].max())
yy = a * xx - (clf.intercept_[0]) / w[1]

# Get support vectors
sv = clf.support_vectors_

# Plot the data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')

# Plot the hyperplane
plt.plot(xx, yy, 'k-')

# Plot the support vectors
plt.scatter(sv[:, 0], sv[:, 1], c='r', marker='x')

# Plot the margin lines
margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
yy_down = yy - np.sqrt(1 + a**2) * margin
yy_up = yy + np.sqrt(1 + a**2) * margin
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')


# # Draw complete horizontal lines passing through the margin vectors
# for i, vector in enumerate(sv):
#     margin_vector = vector + [0, -1 / w[1]]  # Move downward by 1 unit along the y-axis
#     plt.plot([vector[0], margin_vector[0]], [vector[1], margin_vector[1]], 'b--')

plt.title('SVM with Best Hyperplane and Support Vectors')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


# In[99]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a simple dataset
X, y = datasets.make_classification(n_samples=100, n_features=2, n_classes=2, n_clusters_per_class=1, n_redundant=0, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SVM classifier
clf = svm.SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)

# Predict using the predict method
predictions = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy (using predict): {accuracy:.2f}")

# Use decision_function to get decision values
decision_values = clf.decision_function(X_test)

# Plot the decision values and the decision boundary
plt.scatter(X_test[:, 0], X_test[:, 1], c=decision_values, cmap='viridis', marker='o')
plt.colorbar(label='Decision Values')
plt.title('Decision Values and Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot the decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

# Plot decision boundary and margins
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])

plt.show()


# In[102]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a simple dataset
X, y = datasets.make_classification(n_samples=100, n_features=2, n_classes=2, n_clusters_per_class=1, n_redundant=0, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SVM classifier
clf = svm.SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)

# Use decision_function to get decision values
decision_values = clf.decision_function(X_test)
# ... (previous code)
# Plot the decision values and the decision boundary
plt.scatter(X_test[:, 0], X_test[:, 1], c=decision_values, cmap='viridis', marker='o')
plt.colorbar(label='Decision Values')
plt.title('Decision Values and Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot the decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

# Plot decision boundary and margins
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])

# Plot lines along the support vectors
for i, vector in enumerate(clf.support_vectors_):
    plt.plot([vector[0], vector[0] + clf.coef_[0][0]], [vector[1], vector[1] + clf.coef_[0][1]], 'r--', linewidth=1)
    plt.plot([vector[0], vector[0] - clf.coef_[0][0]], [vector[1], vector[1] - clf.coef_[0][1]], 'r--', linewidth=1)

plt.show()


# In[65]:


#one way of drawing margin lines by taking xx of margin and calculating yy_down and yy_up position(where is the destination) and drawing line.
#SEE ANOTHER METHOD DOWN BEST USED
import numpy as np
from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt

# Create a simple dataset
X, y = datasets.make_classification(n_samples=100, n_features=2, n_classes=2, n_clusters_per_class=1, n_redundant=0)

# Initialize the SVM classifier
clf = svm.SVC(kernel='linear')

# Fit the SVM model on the data
clf.fit(X, y)

# Get the separating hyperplane parameters
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(X[:, 0].min(), X[:, 0].max())
yy = a * xx - (clf.intercept_[0]) / w[1]

# Get support vectors
sv = clf.support_vectors_

# Plot the margin lines
margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
yy_down = yy - np.sqrt(1 + a**2) * margin
yy_up = yy + np.sqrt(1 + a**2) * margin
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

# Create a plot
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.plot(xx, yy, 'k-')
plt.scatter(sv[:, 0], sv[:, 1], c='r', marker='x')
plt.title('SVM with Best Hyperplane')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


# In[66]:


#Another efficient and most used method
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a simple dataset
X, y = datasets.make_classification(n_samples=100, n_features=2, n_classes=2, n_clusters_per_class=1, n_redundant=0, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SVM classifier
clf = svm.SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)

# Predict using the predict method
predictions = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy (using predict): {accuracy:.2f}")

# Use decision_function to get decision values
decision_values = clf.decision_function(X_test)

# Plot the decision values and the decision boundary
plt.scatter(X_test[:, 0], X_test[:, 1], c=decision_values, cmap='viridis', marker='o')
plt.colorbar(label='Decision Values')
plt.title('Decision Values and Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot the decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

# Plot decision boundary and margins
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', levels=[-1,0,1], alpha=0.5, linestyles=['-'])

plt.show()


# ## centers parameter in make_blobs() datasets creation(ALSO SEE ABOUT THE make_classification() method below)
# ## Use make_blobs when datapoints must revolve around nearby like kind of sun to other planets like SVM model type or kmeans, But Mostly make_classification() is used

# In[4]:


from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Example 1: Using an integer to specify the number of centers
X1, y1 = make_blobs(n_samples=300, centers=3, random_state=42)

# Example 2: Using an array to specify fixed center locations
centers = [[-5, 0], [0, 0], [5, 0]]
X2, y2 = make_blobs(n_samples=300, centers=centers, random_state=42)

# Plotting the results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X1[:, 0], X1[:, 1], c=y1, cmap='viridis')
plt.title("make_blobs with 3 centers (integer)")

plt.subplot(1, 2, 2)
plt.scatter(X2[:, 0], X2[:, 1], c=y2, cmap='viridis')
plt.title("make_blobs with fixed center locations (array)")

plt.show()


# In[74]:


#sample line
plt.figure(figsize=(10,6))
#(x1,x2) and (y1,y2)
plt.plot([1.9,8.4],[2.1,7.5],"k--")
plt.show()


# In[99]:


#Variation of model performance in C value -SVM
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM classifier with different values of C
C_values = [0.1, 1, 10]

for C in C_values:
    # Create SVM with specified C
    clf = SVC(C=C, kernel='linear')
    
    # Fit the model on the training data
    clf.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = clf.predict(X_test)
    
    # Print accuracy for each value of C
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for C={C}: {accuracy}")


# In[107]:


import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM classifier
clf = SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)

# Use the decision function on the test set
decision_values = clf.decision_function(X_test)

# Print the decision values for the first few samples
print("Decision Values:")
print(decision_values[:5])


# ## np.vstack (Vertical Stack)

# In[1]:


import numpy as np

# Create two arrays
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])

# Vertically stack the arrays
result = np.vstack([array1, array2])

print(result)


# ## Transpose(T)  SEE MORE ABOUT IN BACKUP FILE

# In[2]:


import numpy as np

# Create a 2D array
array_2d = np.array([[1, 2, 3], [4, 5, 6]])

# Transpose the array
transposed_array = array_2d.T

print(transposed_array)


# In[13]:


xx= np.linspace(0, 10, 5)
yy= np.linspace(0, 10, 5)
YY,XX = np.meshgrid(yy,xx)

print("\n\n",XX[:5],YY[:5])
print(XX.ravel(),YY.ravel())
print(np.vstack([XX.ravel(), YY.ravel()]))
xy= np.vstack([XX.ravel(), YY.ravel()]).T
xy


# In[ ]:





# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

# Create a simple dataset
X, y = datasets.make_classification(n_samples=100, n_features=2, n_classes=2, n_clusters_per_class=1, n_redundant=0)

# Initialize the SVM classifier
clf = svm.SVC(kernel='linear')

# Fit the SVM model on the data
clf.fit(X, y)

# Get the separating hyperplane parameters
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(X[:, 0].min(), X[:, 0].max())
yy = a * xx - (clf.intercept_[0]) / w[1]

# Get support vectors
sv = clf.support_vectors_

# Plot the data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')

# Plot the hyperplane
plt.plot(xx, yy, 'k-')

# Plot the support vectors
plt.scatter(sv[:, 0], sv[:, 1], c='r', marker='x')

# Plot the margin lines
margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
yy_down = yy - np.sqrt(1 + a**2) * margin
yy_up = yy + np.sqrt(1 + a**2) * margin
print("xx:\n",xx)
print("yy:\n",yy)
print("margin:\n",margin)
print("yy_down:\n",yy_down)
print("yy_up:\n",yy_up)

plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')


# # Draw complete horizontal lines passing through the margin vectors
# for i, vector in enumerate(sv):
#     margin_vector = vector + [0, -1 / w[1]]  # Move downward by 1 unit along the y-axis
#     plt.plot([vector[0], margin_vector[0]], [vector[1], margin_vector[1]], 'b--')

plt.title('SVM with Best Hyperplane and Support Vectors')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


# In[4]:


print(w,a)


# ## make_classification method for generating datasets
# ## Most popular used when compared with make_blobs, as make_classification uses parameters which are helpful in specifying data classification
# ## number of features,number of classes most important data points will will be around randomly generated in linearity approach.

# In[1]:


from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Generate a 2D dataset for binary classification
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_clusters_per_class=1, n_redundant=0)

# Plot the generated data
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='viridis')
plt.title("make_classification")
plt.show()


#For this kind of graph, we can use Logistic Regression,SVM,Decision tree(max_depth=3) or Random Forest to provide best output on this type of dataset.


# ## np.c_ method

# In[3]:



import numpy as np

# Create two arrays
array1 = np.array([[1, 2, 3],[0,90,8]])
array2 = np.array([[4, 5, 6],[8,3,2]])

# Using np.c_ to concatenate along columns
result = np.c_[array1, array2]

print("Array 1:")
print(array1)
print("\nArray 2:")
print(array2)
print("\nResult after concatenation:")
print(result)


# ## K-means using Elbow method
# ## Visualizing WCSS vs k values
# 
# ## convert_objects() method

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


# In[2]:


#Importing data
dataset= pd.read_csv('cars (dataset for k means).csv')

print(dataset.head())


# In[54]:


print(dataset.columns)


# In[3]:


#As u can see we have Brand column with string values,we don't need it

X=dataset[dataset.columns[:-1]] #or can use iloc or any other method
print(X.head())
print(X.dtypes)
print(X.columns)


# In[4]:


#Converting cubicinches column and weightlbs column of type string object to int

# X[' cubicinches']=pd.to_numeric(X[' cubicinches'],downcast='float',errors='coerce')
# X[' weightlbs']=pd.to_numeric(X[' weightlbs'],downcast='float',errors='coerce')

X[' cubicinches']=pd.to_numeric(X[' cubicinches'],errors='coerce')
X[' weightlbs']=pd.to_numeric(X[' weightlbs'],errors='coerce')


print(X.dtypes)

print(X.head())


# In[5]:


#Checking whether any null values present in the dataset

for i in X.columns:
    print(X[i].isnull().sum())


# In[6]:


#As you can see there are few null values present in two columns

#lets fill those null values or can eliminate them(See more about in Backup file)
for i in X.columns:
#     X[i]=X[i].fillna(int(X[i].mean()))
    X[i]=X[i].fillna(float(X[i].mean()))

for i in X.columns:
    print(X[i].isnull().sum())


# In[12]:


#Now first step is to identify optimal k value

wcss=[]

#Here keep range(1,11) instead of range(0,11)
#When you have 0 clusters, it essentially means that each data point is its own cluster, which doesn't provide any meaningful grouping or clustering. In k-means clustering, the idea is to group similar data points into clusters, and the number of clusters (k) needs to be greater than 0 for this grouping to occur.
for i in range(1,11):
    model=KMeans(n_clusters=i,init="k-means++",max_iter=300,n_init=10,random_state=0)
    model.fit(X)
    wcss.append(model.inertia_)
print(wcss)


# ## Visulization of Elbow Graph

# In[8]:


#Visualizing WCSS vs k values

plt.plot(range(1,11), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()


# In[15]:


#From above graph u can see we an consider values of k at 3 or 4,lets go with 3

model=KMeans(n_clusters=3,init="k-means++",max_iter=300,n_init=10,random_state=0)
print(model.labels_)
#Before training if we print labels attribute(which is nothing but target values or cluster based divided groups values)  from model,will get error
y_pred=model.fit_predict(X)


# In[23]:


#From above graph u can see we an consider values of k at 3 or 4,lets go with 3
#Down is best way to fit and predict model.
model=KMeans(n_clusters=3,init="k-means++",max_iter=300,n_init=10,random_state=0)
model.fit(X)
print(model.labels_)
#Before training if we print labels attribute(which is nothing but target values or cluster based divided groups values) from model,will get error
#It will assign values based on training say 0 class label to a set of similar features, 1 to other set of features, ...
y_pred=model.fit_predict(X)   #Here we are fitting data so give whole dataset(X) for training,if want predict then give seperately to predict function
#As if i give fit_predict(X[:5]) thinking that i want few say 5 datapoints to predict and it will give 5 correct values as output
#But it will even train the model with only 5 datapoints, and it will give based on those trained 5 datapoints.
#Instead use model.fit(X) and later predict statement model.predict(X[:5]) ,now it will give correct 5 predicted values.
print(y_pred)
#Here we are predicting the same dataset or points so same output, try with any similar data point feature and predict it to see perfect output.


# In[17]:


#From above graph u can see we an consider values of k at 3 or 4,lets go with 3
#This is best way
model=KMeans(n_clusters=3,init="k-means++",max_iter=300,n_init=10,random_state=0)
y_pred=model.fit_predict(X)
print(y_pred)
print(model.labels_)


# ## as_matrix() is old version, so in new version usage of to_numpy() or values()

# In[71]:


#As u can see from above the predicted output is either of 0,1,2
#As 3 clusters

#Now to visualize in X-Y axis, we get only values of dataset or can access through specific column names which we want to draw in X-axis and Y-axis

# X=X.as_matrix(columns=None) #or can use values() or to_numpy()
X=X.to_numpy() #Or can use values() method
print(X)

#Just those rows are converted to 2d array accessable instead of columns
#and values are same just in decimal places(10^) different.


# In[75]:


#Visualizing

#!!!!!!IMP!!!!!!!!
#Here label or target names like Toyota,Nissan,Honda are given based 
#on the features of data points from dataset and value is assigned to 
#them by predict method say 0,1,2
#This target names or labels must be done by us by seeing the similarity
#in features or datapoints and assign them target values
plt.scatter(X[y_pred==0, 0], X[y_pred==0, 1],s=100, c='red',label='Toyota')
plt.scatter(X[y_pred==1, 0], X[y_pred==1, 1],s=100, c='blue',label='Nissan')
plt.scatter(X[y_pred==2, 0], X[y_pred==2, 1],s=100, c='green',label='Honda')

plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=300,c='yellow',label="Centroids")
plt.title("Clusters of cara make")
plt.legend()
plt.show()


# In[ ]:





# ## Market Basket Analysis
# ## Association and Apriori Algorithm
# ## Association and Apriori combined is shown next

# In[1]:


#importing libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df= pd.read_csv('Groceries_dataset.csv')
print(df.info())
print(df.head())
print(df['Date'][0])


# ## sort_values() method is used here and even later with by as argument

# In[3]:


#Checking for null values
df.isnull().sum().sort_values(ascending=False)


# ## to_datetime of pandas method

# In[4]:


#As u can see in above info about the dataset, we are having object type for date(means string class object)
#So we have to convert it into date of pandas inbuilt

df['date_conversion']=pd.to_datetime(df['Date'])
df.info()


# In[5]:


#Now lets get the frequency of each unique Item Description in the dataset

print("\n",df.groupby("itemDescription"))

#This size() method will tell how many rows are there with of same value(in itemDescription column) say Instant food products ,UHT-milk etc.. in the dataset
print("\n\n",df.groupby("itemDescription").size())

#Giving the result of above statement name Frequency and reseting the index.
print("\n",df.groupby("itemDescription").size().reset_index(name="Frequency"))

print("\n",df.groupby("itemDescription").size().reset_index(name="Frequency").sort_values(by="Frequency", ascending=False))

item_distr=df.groupby("itemDescription").size().reset_index(name="Frequency").sort_values(by="Frequency", ascending=False).head()
print(item_distr)


# In[6]:


#As we can see item_distr will contain only 5 rows,lets draw a bar graph for those only

width=item_distr['itemDescription']
height=item_distr['Frequency']
xaxis= np.arange(len(width))

plt.figure(figsize=(16,9))
plt.bar(xaxis, height)
plt.xlabel("item names")
plt.ylabel("No. of times the item being sold")
plt.title("Top 5 Items Sold")
plt.xticks(xaxis,width)

plt.show()


# ## set_index() method of dataframe on date column

# In[7]:


#Now converting the index with date(date as datatype not Date column as it is string,so we cannot apply resample method) and creating a new dataset
df_date= df.set_index(["date_conversion"])


# In[16]:


df


# In[17]:


df_date


# In[8]:


#now u can see df and df_date are both different in index and also df_date have few features additional.

#Lets plot the graph for monntly wise no. of items sold, using resample() which is specifically designed for date as index dataframe.

df_date.resample('M')['itemDescription'].count().plot(figsize=(12,6),grid=True,title="number of items sold by month").set(xlabel="Date",ylabel='number of items sold')

    # OR CAN USE .size() method instead of count()
print(df_date.resample('M')['itemDescription'].size())
print(df_date.resample('M')['itemDescription'])
# print(df_date)
print("\n",df_date.resample('M')['itemDescription'].count())


# In[9]:


#Now removing spaces in the string type(itemDescription,Member_number)

#NOTE here again we are dealing/using with df not df_date
cust_level= df[['Member_number','itemDescription']].sort_values(by='itemDescription', ascending=False)
cust_level['itemDescription']= cust_level['itemDescription'].str.strip()
cust_level


# In[74]:


from mlxtend.frequent_patterns import apriori


# In[75]:


pip install mlxtend


# In[19]:


# from mlxtend.frequent_patterns import apriori
#Also other module named 'apyori'
#from apyori import apriori

transactions=[a[1]['itemDescription'].tolist() for a in list(cust_level.groupby('Member_number'))]

l2=[a for a in list(cust_level.groupby('Member_number'))]
print(l2[0][0])
print(l2[0][1])
print([a for a in list(cust_level.groupby('Member_number'))])
print("\n\n\n\n\n")
print([a[0] for a in list(cust_level.groupby('Member_number'))])
print("\n\n\n")
#Here a[1]['itemDescription']- as first value in tuple is row number given in starting after reading from csv file through pd.
#So to get access onto 'itemDescription',which is present at 2nd place so index 1 for accessing it.
#now a is nothing but l2[0] when compared with above l2 result
print([a[1]['itemDescription'].tolist() for a in list(cust_level.groupby('Member_number'))])


# In[18]:


from mlxtend.frequent_patterns import apriori

#training model

model=apriori(pd.DataFrame(transactions), min_support=0.002,use_colnames=True)
model


# In[ ]:


#As u can see that kenral is getting struck when we run the above module of mlxtend.frequent_patterns(which is suitable for small or say medium datasets)
#So lets try with apyori


# In[22]:


pip install apyori


# In[ ]:


# from apyori import apriori

# transactions=[a[1]['itemDescription'].tolist() for a in list(cust_level.groupby('Member_number'))[:10]]

# model=apriori(transactions=transactions, min_support=0.002,min_confidence=0.05, min_lift=3, min_length=2)
# # model[:5]
# print(list(model)[0])


# In[ ]:


#Even for this same issue, the kernal is taking long time to train the model
#So lets install efficient_apriori module,
#which is designed for large datasets to get trained than mlxtend.frequent_patterns in terms of speed.


# ## efficient_apriori instead of mlxtend.frequent_patterns(mostly used module for apriori) and apyori modules
# ## transactions(should not perform one hot encoding) vs categorial(must do one hot encoding)

# In[11]:


from efficient_apriori import apriori

#here no need of one-hot encoding, as each customer purchases some items and as in dataset form,in each column few items will purchased by the customer.
#it is of kind transactions not like categorical format,here transactions list in which again list of items purchased by that customer
#so will use transactions argument in apriori model
#else we should convert into one hot encoding then must pass the data.
#See more in explore mglearn chatgpt chat

#transactions arguments in apriori model must be list of lists.
transactions=[a[1]['itemDescription'].tolist() for a in list(cust_level.groupby('Member_number'))[:10]]

# model=apriori(transactions=transactions, min_support=0.002,min_confidence=0.05)
itemset,rules=apriori(transactions=transactions, min_support=0.002,min_confidence=0.05)
#https://www.youtube.com/watch?v=GwIo3gDZCVQ 6:39:00
#model gives itemset (mostly occured items with length 1,2(combined),3(triplet combination)
#             rulesare nothing but association rules within a given dataset of transactions.
#Association rules express relationships between different itemsets, indicating that the 
#presence of one set of items might imply the presence of another set.


# In[21]:


for key,value in list(itemset.items())[:4]:
    print(key,value,end="\n\n")


# In[12]:


print(rules[:50])


# ## rule, support, confidence and lift

# In[13]:


#support measures how often the rule occurs, 
#confidence measures the strength of the rule i.e It indicates how often the rule has been found to be true, and
#lift compares the rule's likelihood to random chance.
for rule in rules[:50]:
    print(f"Rule: {rule}, Support: {rule.support}, Confidence: {rule.confidence}, Lift: {rule.lift}")


# ## Top 15 Rules sorted by confidence and lift,with reverse argument as it will sort in ascending order so

# In[15]:


# Sort rules by confidence and lift
sorted_rules = sorted(rules, key=lambda x: (x.confidence, x.lift), reverse=True)

# Print the top N rules
top_n = 15
for i, rule in enumerate(sorted_rules[:top_n]):
    print(f"Rule #{i + 1}: {rule}, Support: {rule.support}, Confidence: {rule.confidence}, Lift: {rule.lift}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[27]:


# model=apriori(transactions=transactions, min_support=0.002,min_confidence=0.05)
#if i used model as variable output to trained algorithm as commented above,the output looks like,
model[0]


# In[28]:


# model=apriori(transactions=transactions, min_support=0.002,min_confidence=0.05)

model[:2]


# In[29]:


# model=apriori(transactions=transactions, min_support=0.002,min_confidence=0.05)

model[:3]


# In[31]:


len(model)
model[:5]


# In[ ]:





# In[ ]:





# In[6]:


#Converting numeric string type to integer using convert_objects() method is used in older version of jupyter
#In newer versions, use .to_numeric() on that column or columns
import pandas as pd

# Create a DataFrame with an object column
data = {'Name': ['John', 'Jane', 'Bob'], 'Age': [25, '26', '30']}
df = pd.DataFrame(data)

# Display the DataFrame
print("Before conversion:")
print(df.dtypes)

# Use convert_objects to attempt conversion
df = df.convert_objects(convert_numeric=True)

# Display the DataFrame after conversion
print("\nAfter conversion:")
print(df.dtypes)


# In[7]:


df['Age'] = pd.to_numeric(df['Age'], errors='coerce')


# In[11]:


df.head()
df.dtypes


# In[66]:


import pandas as pd

# Create a DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

print(df)
# Using the values attribute
array1 = df.values

# Using the to_numpy() method (recommended)
array2 = df.to_numpy()

print(array1)
print(array2)


# In[7]:


import pandas as pd

# Create a DataFrame
df = pd.DataFrame({
    'Category': ['A', 'B', 'A', 'B', 'A', 'B'],
    'Value': [10, 15, 20, 25, 30, 35]
})
    #OR
# Group by 'Category' and calculate the sum
grouped_df = df.groupby('Category').size()
print(grouped_df)
grouped_df1 = df.groupby('Category')
print(grouped_df1.sum(),"\n",grouped_df1.size())


# In[9]:


import pandas as pd

# Create a DataFrame
df = pd.DataFrame({
    'Category': ['A', 'B', 'A', 'B', 'A', 'B'],
    'Value': [10, 15, 20, 25, 30, 35]
})
    #OR
# Group by 'Category' and calculate the sum
grouped_df = df.groupby('Category').size()
# Reset the index of grouped DataFrame
grouped_df_reset = grouped_df.reset_index()
print(grouped_df_reset)


# In[17]:


import pandas as pd

# Create a DataFrame with a named index
df = pd.DataFrame({
    'Value': [10, 15, 20, 25, 30, 35],
}, index=['A', 'B', 'A', 'B', 'A', 'B'], name='Category')

# Reset the index with a new name for the index column
reset_df = df.reset_index()

print(reset_df)


# In[39]:


import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame
df = pd.DataFrame({
    'Category': ['A', 'B', 'A', 'B', 'A', 'B'],
    'Value': [10, 15, 20, 25, 30, 35]
})
print(df['Category'][0])
# Group by 'Category' and calculate the sum
print( df.groupby('Category').size(),"ohyhukbjhb")
grouped_df = df.groupby('Category').sum()
print(grouped_df)
print(grouped_df.loc["A",'Value'])
print(grouped_df.iloc[0])
# Reset the index
grouped_df_reset = grouped_df.reset_index()
print(grouped_df_reset)

print(grouped_df_reset['Category'][0])
# Sort by the sum of values
sorted_df = grouped_df_reset.sort_values(by='Value', ascending=False)
print(sorted_df)


# Plot a bar graph
plt.bar(sorted_df['Category'], sorted_df['Value'])
plt.xlabel('Category')
plt.ylabel('Sum of Values')
plt.title('Bar Graph of Sum of Values by Category')
plt.show()

#It will show error as index now is 0,1... not category so we cannot access using loc
print(grouped_df_reset.loc['A'])


# In[89]:


import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    'Member_number': [1, 2, 1, 2, 3],
    'itemDescription': ['apple', 'banana', 'orange', 'banana', 'apple']
})

# Group by 'Member_number'
grouped_df = df.groupby('Member_number')

# Iterate over groups
for group_key, group_df in grouped_df:
    print(f"Group {group_key}:")
    print(group_df['itemDescription'])
    print("---")


# ## resample with freq='D'

# In[7]:


import pandas as pd

# Create a DataFrame with a datetime index
data = {
    'Value': [10, 15, 20, 25, 30],
}

dates = pd.date_range('2023-01-01', periods=5, freq='D')
df = pd.DataFrame(data, index=dates)

# Resample to monthly frequency
resampled_df = df.resample('M').sum()

print("Original DataFrame:")
print(df)

print("\nResampled DataFrame:")
print(resampled_df)


# ## resample with freq='M'

# In[12]:


import pandas as pd

# Create a DataFrame with a datetime index
data = {'Value': [10, 15, 20, 25, 30]}

# Specify the start date and the number of periods
start_date = '2023-01-15'
num_periods = 5

# Create a datetime index with a specific start date
dates = pd.date_range(start=start_date, periods=num_periods, freq='M')   #Even we h for hours, W for weeks
df = pd.DataFrame(data, index=dates)

# Resample to monthly frequency
resampled_df = df.resample('M').sum()

print("Original DataFrame:")
print(df)

print("\nResampled DataFrame:")
print(resampled_df)

# Example 1: Basic Usage
date_index1 = pd.date_range(start='2023-01-01', end='2023-01-10')
print("Example 1:")
print(date_index1)
print("\n")

# Example 2: Specifying Periods
date_index2 = pd.date_range(start='2023-01-01', periods=5)
print("Example 2:")
print(date_index2)
print("\n")

# Example 3: Specifying Frequency
date_index3 = pd.date_range(start='2023-01-01', periods=5, freq='2D')
print("Example 3:")
print(date_index3)
print("\n")

# Example 4: Specifying End and Frequency
date_index4 = pd.date_range(end='2023-01-10', periods=5, freq='B')
print("Example 4:")
print(date_index4)
print("\n")


# In[ ]:





# In[ ]:





# In[10]:


pip install efficient-apriori


# In[17]:


#Practicing
from efficient_apriori import apriori
from itertools import chain
# Assuming cust_level is your DataFrame and 'itemDescription' is the column containing items
transactions = [list(set(a[1]['itemDescription'].tolist())) for a in list(cust_level.groupby('Member_number'))]
print(len(transactions))
# Flatten the list of lists
flat_transactions = list(chain.from_iterable(transactions))
print(len(flat_transactions))
# Convert to a set to get unique items
unique_items = set(flat_transactions)
print(len(unique_items))
# Create a list of transactions with unique items
unique_transactions = [list(set(transaction)) for transaction in transactions]
print("Finally",len(unique_transactions))
# Apply Apriori
itemsets, rules = apriori(unique_transactions, min_support=0.002, min_confidence=0.05)

# Print the first 5 rules
print(list(rules)[:5])


# In[15]:


print(list(rules)[:50])


# In[17]:


from efficient_apriori import apriori

# Assuming cust_level is your DataFrame and 'itemDescription' is the column containing items
transactions = [list(set(a[1]['itemDescription'].tolist())) for a in list(cust_level.groupby('Member_number'))]

# Flatten the list of lists
flat_transactions = [item for sublist in transactions for item in sublist]

# Convert to a set to get unique items
unique_items = set(flat_transactions)

# Create a list of transactions with unique items
unique_transactions = [list(set(transaction)) for transaction in transactions]

# Apply Apriori
itemsets, rules = apriori(unique_transactions, min_support=0.002, min_confidence=0.05)

# Print itemsets with support values
for itemset, support in list(itemsets.items())[:10]:
    print(f"Itemset: {itemset}, Support: {support}")

# Print rules with support, confidence, and lift values
for rule in rules:
    print(f"Rule: {rule}, Support: {rule.support}, Confidence: {rule.confidence}, Lift: {rule.lift}")


# In[18]:


# Sort rules by confidence and lift
sorted_rules = sorted(rules, key=lambda x: (x.confidence, x.lift), reverse=True)

# Print the top N rules
top_n = 10
for i, rule in enumerate(sorted_rules[:top_n]):
    print(f"Rule #{i + 1}: {rule}, Support: {rule.support}, Confidence: {rule.confidence}, Lift: {rule.lift}")


# In[19]:


from efficient_apriori import apriori

# Assuming cust_level is your DataFrame and 'itemDescription' is the column containing items
transactions = [list(set(a[1]['itemDescription'].tolist())) for a in list(cust_level.groupby('Member_number'))]

# Flatten the list of lists
flat_transactions = [item for sublist in transactions for item in sublist]

# Convert to a set to get unique items
unique_items = set(flat_transactions)

# Create a list of transactions with unique items
unique_transactions = [list(set(transaction)) for transaction in transactions]

# Apply Apriori
itemsets, rules = apriori(unique_transactions, min_support=0.002, min_confidence=0.05)
print(itemsets)
# Print itemsets with support values
for itemset, support in list(itemsets.items())[:10]:
    print(f"Itemset: {itemset}, Support: {support}")

# Print rules with support, confidence, and lift values
for rule in rules:
    print(f"Rule: {rule}, Support: {rule.support}, Confidence: {rule.confidence}, Lift: {rule.lift}")


# In[11]:


import pandas as pd

# Sample DataFrame
data = {
    'Product': ['A', 'B', 'C', 'D', 'E','F'],
    'Price': [10, 25, 15, 30, 20,30],
    'Quantity': [100, 50, 75, 25, 60,80]
}

df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)
# Get the top 2 rows based on the 'Price' column
top_price_rows = df.nlargest(4, 'Price')
top_price_quantity_rows= df.nlargest(2, ['Price','Quantity'])

ground_price_rows = df.nsmallest(1, 'Price')

print("\nTop 2 Rows Based on Price:")
print(top_price_rows)
print("\nTop 2 Rows Based on Price and qunatity:")
print(top_price_quantity_rows)
print("\nTop 1 Row Based on lowest Price:")
print(ground_price_rows)


# In[22]:


import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Sample transactional data
transactions = [['bread', 'butter'], ['bread', 'milk'], ['butter', 'milk'], ['bread', 'eggs']]

# Convert transactions to a DataFrame with a single column 'items'
df = pd.DataFrame({'items': transactions})

# Convert the 'items' column to a string, making each row a single string of items
df['items'] = df['items'].apply(lambda x: ', '.join(x))
print(df)
# Convert transactions to one-hot encoding
df_encoded = pd.get_dummies(df['items'].str.split(', ', expand=True).stack(), prefix='', prefix_sep='').groupby(level=0).max()
print(df_encoded)
# Apply Apriori algorithm
frequent_itemsets = apriori(df_encoded, min_support=0.5, use_colnames=True)
print(frequent_itemsets)
# Generate association rules
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)

# Display the rules
print(rules)


# In[ ]:




