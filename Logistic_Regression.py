
import sklearn 
import pandas  
import seaborn  
import matplotlib


from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler



# Getting CSV WITH pandas read CSV
data_set = pandas.read_csv ('suv.csv')
# Priting top 10 Header points
print(data_set.head(10))

# getting the whole shape of dataset
# print(data_set.shape)
# getting info of datset...
# data_set.info()


data_set.groupby ('Purchased').size()


# CLeaned the data set by dropping user id which is not important on axis 1
cleaned_data_set = data_set.drop (columns = ['User ID'], axis = '1')
# Then again printed up head of dataset
cleaned_data_set.head ()

# Then discried the dataset 
cl = cleaned_data_set.describe ()
# print(cl)




# Then PLotting our dataset according to Purchase of cleaned one 
seaborn.countplot (x = 'Purchased', data = cleaned_data_set)


# Then plotting according to gender type of purchases of cleaned data
seaborn.countplot ( x = 'Purchased', hue = 'Gender', data = cleaned_data_set)


# Then a histogram of Age type With bins of 20
data_set ['Age'].hist(bins = 20)


# Then a for loop, which select age catgory
# AN empty list for appending age category of A < B C < D
age_category = []
# A for loop with a range of 0 and length of dataset of [AGe variable]

for i in range (0, len  (data_set ['Age'])):
    if cleaned_data_set ['Age'][i] <= 20:
        age_category.append ('A')
    elif 20 < cleaned_data_set ['Age'][i] <= 26:
        age_category.append ('B')
    elif 26 < cleaned_data_set ['Age'][i] <= 30:
        age_category.append ('C')
    elif 30 < cleaned_data_set ['Age'][i] <= 40:
        age_category.append ('D')
    elif 40 < cleaned_data_set ['Age'][i] <= 50:
        age_category.append ('E')
    else:
        age_category.append ('F')


age_data_frame = pandas.DataFrame (data = age_category, columns = ['AgeCategory'])
augmented_data_set = pandas.concat([cleaned_data_set, age_data_frame], axis = 1)
augmented_data_set.head()




# Then again plotting the new data with the new category of AGe category with the argumental data.
seaborn.countplot ( x = 'Purchased', hue = 'AgeCategory', data = augmented_data_set)

# ANother histogram for THe EStimated salary with 20 bins
data_set ['EstimatedSalary'].hist(bins = 20)



# Again categorizing income varibalbe  as high low very low very high etc....
# MAkin a Income category empty list for appending new variables
income_category = []
# A for loop for (i as income) in rage of 0 to lenght of dataset of variavel estimated dalary
for i in range (0, len  (data_set ['EstimatedSalary'])):
    if cleaned_data_set ['EstimatedSalary'][i] <= 19500:
        income_category.append ('Very Low')
    elif 19500 < cleaned_data_set ['EstimatedSalary'][i] <= 40000:
        income_category.append ('Low')
    elif 40000 < cleaned_data_set ['EstimatedSalary'][i] <= 60000:
        income_category.append ('Moderately Low')
    elif 60000 < cleaned_data_set ['EstimatedSalary'][i] <= 80000:
        income_category.append ('Medium')
    
    elif 80000 < cleaned_data_set ['EstimatedSalary'][i] <= 100000:
        income_category.append ('Moderately high')
    elif 100000 < cleaned_data_set ['EstimatedSalary'][i] <= 130000:
        income_category.append ('Very High')
    elif 130000 < cleaned_data_set ['EstimatedSalary'][i] <= 145000:
        income_category.append ('Very High')
    else:
        income_category.append ('Extremely High')

# Then again putting(replcaing) that old data with new categories.......... 
income_data_frame = pandas.DataFrame (data = income_category, columns = ['IncomeCategory'])
# in argumental data 2
augmented_data_set_2 = pandas.concat([augmented_data_set, income_data_frame], axis = 1)
# Again printing head of argumental data 2
augmented_data_set_2.head()





# Then plotting  tha data 
seaborn.countplot ( x = 'Purchased', hue = 'IncomeCategory', data = augmented_data_set_2)

# Then making a binary data of gendar as 0 for male and 1 for female with pandas. get dummies and dropping the first female column
binary_gender = pandas.get_dummies (augmented_data_set_2 ['Gender'],drop_first = True)
binary_gender.head ()


# SImilar binirazing the age category data in 0 and 1 for machine undersatnding.....
binary_age = pandas.get_dummies (augmented_data_set_2 ['AgeCategory'])
binary_age.head ()


# similar Biniarzing the income data categories as 0,1
binary_income = pandas.get_dummies (augmented_data_set_2 ['IncomeCategory'])
binary_income.head ()

# NOw concataniting all the biniary forms of data and argumental data categories that logisic regression can understand....
final_data_set  = pandas.concat ([augmented_data_set_2, binary_age, binary_gender, binary_income], axis = 1)
# AND here we've just cleaned our whole data by dropping real age gender, salary, incomecategory, age category
final_data_set_1 = final_data_set.drop (columns = ['Age', 'Gender', 'EstimatedSalary', 'IncomeCategory', 'AgeCategory'], axis = 1)
final_data_set_1.head ()



# NOw Training or model
# Y = new dataset Varible purchased 0,1(0 for not 1 for purcahesd)
Y = final_data_set_1 ['Purchased']
# X = drop 1 column which is purchased........
X = final_data_set_1.drop (columns = ['Purchased'], axis = 1)
# Now getiing first 5 values of Y
Y.head()


# Now spliting our data test size is 20%
test_set_size = 0.2
# setted seed = 0 whish is our stimater  ,is not that important
seed = 0
# Then setted values for xtrain y train xtest ytest as modelselctor of model train test split== X And Y test size = testsetsize which is 20% And random state is none
X_train, X_test, Y_train, Y_test = model_selection.train_test_split (X,Y, test_size = test_set_size, random_state = seed)
# Setted model as logistic regrssion inside regression we've setted solver as liblinear
model = LogisticRegression (solver = 'liblinear')
# Then fitted xtrain and ytrain in model with fit keyword
model.fit (X_train, Y_train)



# Then setted predictions as model.predict xtest
predictions = model.predict (X_test)


# The checked for report with classification report of Y test and predictions
report = classification_report (Y_test, predictions)
print (report)
# Here we used what is the true value with confusion matrix
print (confusion_matrix (Y_test, predictions))

# Then accurac score 
accuracy_score (Y_test, predictions)

# IN last we are setting iloc for Y and X whihc means only for indexing between 0 adn 1 for an sliced data with numpy iloc is pandas modeule

Y = data_set.iloc [:, 4]
X = data_set.iloc [:, 2:4]

print (X.head())
print (Y.head())


# then setted test size as 0.20 which means 20%
test_set_size = 0.2
seed = 0
X_train, X_test, Y_train, Y_test = model_selection.train_test_split (X,Y, test_size = test_set_size, random_state = seed)
# Again traid our data

# And scales = standard scaler method 
scaler = StandardScaler ()
# And transformed our Xtrain in scaler transform to xTrain new
X_train = scaler.fit_transform(X_train)
# and Xtest ad scaler fit transform with XTest new
X_test = scaler.fit_transform(X_test)

# Last classifyng our model with random steate = none(0) and slover = linlinera
classifier = LogisticRegression (random_state = seed, solver = 'liblinear')
classifier.fit (X_train, Y_train)


predictions = classifier.predict (X_test)

report = classification_report (Y_test, predictions)
print (report)

acc = accuracy_score (Y_test, predictions)
print(acc)
