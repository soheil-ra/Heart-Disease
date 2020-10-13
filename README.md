## Predicting Presence of Heart Diseases Using Machine Learning
**Abstract**

Machine learning involves artificial intelligence, and it is used in solving many problems in data science. One common application of machine learning is the prediction of an outcome based upon existing data. The machine learns patterns from the existing dataset, and then applies them to an unknown dataset in order to predict the outcome. Classification is a powerful machine learning technique that is commonly used for prediction. Some classification algorithms predict with satisfactory accuracy, whereas others exhibit a limited accuracy. In this project a comparative analytical approach was done to determine how the prediction technique can be applied for improving prediction accuracy in heart disease, which is the major cause for human mortality rate. Correct diagnosis and treatment at an early stage will save people from heart disease and will decrease mortality rate due to heart problems. Using Machine Learning techniques facilitate the prediction of heart diseases. In this project relevant features are determined for heart disease prediction with known dataset using correlation measures. To model the data, three different algorithms, Logistic Regression, Random Forest Classifier, and K-Nearest Neighbors Classifier, were selected. The comparison accuracy results of these models were presented. You can find the following sections in this project:<br><br>
- **1.    Introduction**<br>
- **2.    Importing Python libraries**<br>
- **3.    Reading CSV Files**<br>
- **4.    Cleaning Data**<br>
- **5.    Exploring Data**<br>
   - **5.1.   Converting Categorical Variables to Numerical**<br>
   - **5.2.   Measuring the Correlation**<br>
- **6.    Visualization**<br>   
- **7.    Proposed Methods and Experiments**<br>
   - **7.1.  Selecting Attribute Set**<br>
   - **7.2.  Proposed Methods**<br>
- **8.  Results and Discussion**<br>
- **9.  Conclusion and Summary**<br>
- **10.  References**<br>

**1. Introduction**

In health care industry, predicting heart disease is a challenging issue. In early days medical tests such as Electrocardiogram (ECG) and blood tests have been used for predicting heart diseases. In addition to clinical tests, computer aided diagnosis systems, namely, patient information, medical diagnosis and medical images are being used for predicting heart diseases. Machine learning algorithms have significant role in predicting diseases. In this project, we have proposed a conceptual framework for the prediction of diseases using three different algorithms, Logistic Regression, Random Forest Classifier, and K-Nearest Neighbors Classifier. It is prerequisite to identify features that are relevant to the prediction of diseases. From the dataset, it is found that eleven attributes, namely, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, and slope are being used for predicting heart diseases. These features are analyzed for their relevance for prediction of heart disease using correlation techniques. The results show that ten attributes, age, sex, cp, restecg, trestbps, chol, fbs, restecg, exang, oldpeak, and slope are found as most relevant attributes in predicting heart diseases. Accuracy obtained using different classifiers with different sets of attributes are reported in this project. Comparing the accuracy results of three models show that Random Forest Classifier has the highest accuracy among three models.<br><br>
**2. Importing Python libraries**

To build the model, we need some tools to make the process as seamless as possible. Here is the list of a few of them:      
- **Pandas:** Is a library used for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series.
- **Sklearn:** Is a library built on the following machine learning libraries: NumPy, a library for manipulating multi-dimensional arrays and matrices.
- **Scipy:** Is a library used to solve scientific and mathematical problems. It is built on the NumPy extension and allows the user to manipulate and visualize data with a 
         wide range of high-level commands.
- **Matplotlib:** Is a plotting library for the Python programming language and its numerical mathematics extension NumPy. It provides an object-oriented API for     
         embedding plots into applications using general-purpose GUI toolkits like Tkinter, wxPython, Qt, or GTK+.
- **And More ...**<br>

**3. Reading CSV Files**

First step to any data science project is to import the data. Often, we’ll work with data in Comma Separated Value (CSV) files. In order to start working on 'heart.csv' file, the dataset has to be read and loaded to the data frame by using **read_csv()** method.<br><br>
![DataDescription](https://user-images.githubusercontent.com/71153587/95864655-b10f5f80-0d33-11eb-949b-a4451bccf6b6.png)

**4. Cleaning Data**

The chance of getting a perfectly cleaned dataset that meets all of our requirements is slim to none. So, we need to clean the data before we start modeling. Trying to skip the data cleaning steps, often runs into problems getting the raw data and not cleaned to work with traditional data cleansing tools for analysis the data. Thus, it becomes important to take into consideration, the data cleaning steps and data cleaning methods. Here is a checklist when it comes to cleaning the data:<br>
  - Checking for missing values by using **dataframe.isnull()** function:  The result came out False for all the columns in the dataset which meant that there were no null values within the data.
  - Checking for the zero values. The two features, 'chol' and 'trestbps’ include zero values which were removed from the dataset and replaced with the mean of the respective column.
  - Checking if any outliers existed. The boxplot method was applied to find the outliers for the numerical features, ‘chol', 'trestbps’, 'thalach', 'oldpeak', ‘age'.  Using **DelOutliers()** function to remove features (cp, trestbps, chol, thalach, fbs, oldpeak) outliers by using IQR Method.<br><br>

![Outliers](https://user-images.githubusercontent.com/71153587/95880171-39e2c700-0d45-11eb-82c9-b1e6b6aa59c2.PNG)

**5. Exploring Data** 

This is very important step as it helps us understand some of the following questions:<br>
-	 How is the data distributed?<br> 
-	 Are there any outlier<br>
-	 Is there any relationship between variables?<br>
-	 Is the data balanced?<br>
By better understanding the answers to these questions we can validate whether we need to do further transformations or if we need to change the model that we picked.<br><br>
     **5.1. Converting Categorical Variables to Numerical:** 
            Types of existing data in the dataset:
             •	bins = ['sex', 'fbs', 'exang']
             •	cats = ['cp', 'restecg', 'slope']
             •	nums = ['age', 'oldpeak', 'trestbps', 'chol', 'thalach']
             •	diagnosis(target) = ['diagnosis']
             To work with categorical variables, they were broken into dummy variables with 1s and 0s, before training the Machine Learning models. To get this done, we use the               **get_dummies()** method from pandas.<br><br>
    **5.2. Measuring the Correlation**
           For data scientists, checking correlations is an important part of the exploratory data analysis process. This analysis is one of the methods used to decide which
           features affect the target variable the most, and in turn, get used in predicting this target variable. In other words, it’s a commonly-used method for feature
           selection in machine learning and because visualization is generally easier to understand than reading tabular data, heatmaps are typically used to visualize 
           correlation matrices. A simple way to plot a heatmap in Python is by importing and implementing the Seaborn library. The heatmap shows that the maximum heart
           rate(thalach) doesn't seems to correlate significantly with a higher risk of heart disease but chest pain(cp), exercise induced angina(exang), ST depression induced
           by exercise relative to rest(oldpeak), and the slope of the peak exercise ST segment(slope) are highly corrolate with a higher risk of heart disease.<br> 
![heatmap](https://user-images.githubusercontent.com/71153587/95865721-fb451080-0d34-11eb-9b97-7789583fcefa.PNG) 

**6. Visualization**<br><br>
The below graphs show that from 1190 patients 629 were diagnosed and 561 were not diagnosed with heart disease and male more diagnosed with heart disease than female.<br><br>

![V1](https://user-images.githubusercontent.com/71153587/95881857-099c2800-0d47-11eb-8df3-90ee46990fee.PNG)


The below graphs show, that men are more susceptible to heart disease than women, the ST segment/heart rate slope as a predictor of coronary artery disease, individuals with heart disease are/have more likely to present with a flat slope and Less likely to present with a down slope, individuals with heart disease have Less likely to present with fbs <= 120 mg/dl. Blood sugar levels on fasting > 120 mg/dl represents as 1 in case of true and 0 as false, patients with the type 1 chest pain have less chance of getting heart disease and patients with type 4 have more chance and in the results of the types 2 and 3 chest pain is less heart problem. Here we see that individuals with heart disease have more likely to present with asymptomatic angina and less likely to present with typical angina.<br><br>


![V2](https://user-images.githubusercontent.com/71153587/95882421-afe82d80-0d47-11eb-8e28-b97f85d46118.PNG)

**7. Proposed Method and Experiments**<br><br>
     **7.1. Selecting Attribute Set**
            There may be many attributes related to a given prediction problem, but not all of them have strong association with the prediction. Hence, finding the relevant
            attributes for a given prediction problem is important. In this project, relevant attributes for heart disease prediction are determined using correlation measure 
            and different combinations of selected features were chosen based on the degree of their correlation to the target value. It is found that the eleven attributes
            (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, and slope) are being used while predicting heart diseases. In order to find the weight or rank
            of these attributes an experiment has been conducted. In this experiment the correlation between each attribute with target value is found out. In order to determine
            which feature set produces optimal accuracy, they were added one by one up to eleven by choosing the one with highest weight as the first attribute.<br><br>    
     **7.2. Proposed Method**
            To verify the study’s goal of predicting Presence of Heart Diseases, three different prediction models, and machine learning algorithms were used. Logistic
            Regression, Random Forest Classifier, and K-Nearest Neighbors Classifier were applied to validate the accuracy of the prediction.<br>
       
**Logistic Regression** is a Machine Learning classification algorithm that is used to predict the probability of a categorical dependent variable. In logistic regression, the dependent variable is a binary variable that contains data coded as 1 (yes, success, etc.) or 0 (no, failure, etc.). In other words, the logistic regression model predicts P(Y=1) as a function of X.<br>

**Random Forest Classifier**
This classifier takes the concept of decision trees to the next level. It creates a forest of trees where each tree is formed by a random selection of features from the total features. Here, we can vary the number of trees that will be used to predict the class. It functions by breaking down a dataset into smaller and smaller subsets based on different criteria. Different sorting criteria will be used to divide the dataset, with the number of examples getting smaller with every division. Once the network has divided the data down to one example, the example will be put into a class that corresponds to a key. When multiple random forest classifiers are linked together, they are called Random Forest Classifiers.<br>

**K-Neighbors Classifier**
This algorithm is used for Classification and Regression. In both uses, the input consists of the k closest training examples in the feature space. On the other hand, the output depends on the case.
    In K-Nearest Neighbors Classification the output is a class membership.<br>
    In K-Nearest Neighbors Regression the output is the property value for the object.<br>
    K-Nearest Neighbors is easy to implement and capable of complex classification tasks. K-Nearest Neighbors biggest advantage is that the algorithm can make predictions without training, this way new data can be added.<br><br>
**Confusion Matrix** 
To evaluate the accuracy of the three classifications confusion matrix was used. A confusion matrix is a matrix (table) that can be used to measure the performance of a machine learning algorithm, usually a supervised learning one. Each row of the confusion matrix represents the instances of an actual class and each column represents the instances of a predicted class. It can be the other way around as well, i.e. rows for predicted classes and columns for actual classes. The name confusion matrix reflects the fact that it makes it easy for us to see what kind of confusions occur in our classification algorithms.<br> 
CM for the three following Models:<br>
-	TP = # True Positives,<br> 
-	TN = # True Negatives,<br> 
-	FP = # False Positives,<br> 
-	FN = # False Negatives),<br>
**Accuracy = (TP + TN) / (TP + TN + FP + FN)**<br>
![Capture](https://user-images.githubusercontent.com/71153587/95868104-e74ede00-0d37-11eb-9692-e51bd3322cfe.PNG)

**7. Conclusion:**

Although ML/DM techniques have many advantages, they may not be the perfect methods. According to the no-free-lunch theorem, different ML/DM algorithms are suitable for their own particular problems. One algorithm may work well on a specific dataset while it cannot show a good performance on some others. So, selecting a suitable algorithm for a specific dataset is a big challenge in bioinformatics. Consequently, selecting good feature selection or classification algorithms is also a big challenge in this field. Also, ML/DM algorithms commonly need massive datasets to be trained. These datasets must be inclusive and unbiased with high quality.[5]
In this project, the first approach was to propose attributes for the prediction of heart diseases dataset by analyzing their correlation measures. It was found that the best combination for this purpose was a set of features with eleven attributes(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, and slope). With using the data correlation result, different sets of attributes were selected for predicting process and the result was shown in table 13.1.2. Some of selected feature sets that considered to be tested as a dataset for the three models are; feature set with eleven attribute (age, trestbps, chol, thalach, oldpeak, sex, cp, fbs, restecg, exang, slope), three feature sets with three attributes (cp, exang, oldpeak), (exang, oldpeak, slope), (cp, exang, slope), feature set with four attributes (cp, exang, oldpeak, slope), feature set with seven attributes (sex, cp, fbs, restecg, exang, oldpeak, slope), feature set with eight attributes (tbs, sex, cp, fbs, restecg, exang, oldpeak, slope). These attributes were ranked according to the correlation measures and used with three prediction models for the purpose of accuracy prediction. The result shows that Random Forest Classifier with the accuracy score of 0.90 had the highest value among the three models, taking the eleven features as a dataset.
In this project, it was considered to use the recommended feature set to study the impact of three techniques in enhancing the accuracy of classifiers and their results were compared based on their CM (Confusion Matrix) method.
For the future work, measuring the accuracy values of different classifiers, would be obtained for all possible feature sets.<br><br>
![Result](https://user-images.githubusercontent.com/71153587/95873506-fcc70680-0d3d-11eb-9b46-73619660bb83.PNG)

**8. References:**
-  [1] https://cxl.com/blog/outliers/
-  [2] https://www.kaggle.com/questions-and-answers/97472
-  [3] https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02
-  [4] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
-  [5] https://www.nature.com/articles/s41597-019-0206-3#article-info
-  [6] https://www.sciencedirect.com/science/article/pii/S235291481830217X
-  [7] Prediction of Diseases using Big Data Analysis
-  [8] Heart Disease Prediction with MapReduce by using Weighted Association Classifier and K-Means.
-  [9] https://www.mayoclinic.org/diseases-conditions/high-blood-cholesterol/expert-answers/cholesterol-level/faq-20057952
-  [10] https://stackabuse.com/overview-of-classification-methods-in-python-with-scikit-learn/
-  [11] https://heartbeat.fritz.ai/seaborn-heatmaps-13-ways-to-customize-correlation-matrix-visualizations-f1c49c816f07
-  [12] https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
-  [13] https://www.python-course.eu/confusion_matrix.php
-  [14] https://pythonbasics.org/k-nearest-neighbors/


















