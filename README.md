**GETNET DEMIL-CLGF2[D** ](https://medium.com/search?source=---two_column_layout_nav----------------------------------)**

**Predicting** **APS Failure at Scania Trucks** 

A heavy-duty vehicle's Air Pressure System (APS), which uses pressurized air to force a piston  to  press  on  the  brake  pads  in  order  to  slow  down  the  vehicle,  is  a  crucial component. The ease of accessibility and sustainability of air from nature are benefits of using an APS rather than a hydraulic arrangement. 

The dataset consists of data collected from heavy Scania trucks in everyday usage. The system in focus is the Air Pressure system (APS) which generates pressurized air that are utilized in various functions in a truck, such as braking and gear changes. The dataset’s positive class consists of component failures for a specific component of the APS system. The negative class consists of trucks with failures for components not related to the APS. The data consists of a subset of all available data, selected by experts. The goal is to predict whether the APS system has a failure and should be repaired, too.  A false positive prediction  means  that  the  mechanics  checks  the  APS  system  unnecessarily,  a  false negative prediction that it is faulty, but not checked. Obviously, the latter has larger negative consequences. 

**Content:** 

1. Dataset Overview 
1. Performance Metric 
1. Literature Review 
1. First Cut Solution 
1. Getting Started 
1. Removing Single Valued Features 
1. Handling Missing Values 
1. Separating Features for Analysis 
1. Histogram Feature Analysis 
1. Numerical Feature Analysis 
1. Preparing the data 
1. Experimenting with classical ML models 
1. References 

**Machine Learning Formulation** 

In this binary classification problem, the class "1" indicates that the failure was caused by a particular APS component, while the class "0" indicates that the failure was not caused by that component. We can therefore create an ML model that would determine if the failure was caused by the truck's APS or not given a new data point (sensor information). 

Business Constraints 

1. Latency must be fairly low, to detect a failure in the APS and avoid increase in maintenance cost. 
1. Cost of misclassification is very high since an undetected failure of the APS component can lead to failure of the truck during operation and therefore an increase in maintenance cost. 

**Dataset Overview** 

The training dataset consists of 39,900 data points and 171 features including one class label. The given data is in the form of two files. The first file name X\_train.csv has the data points of the above-mentioned amount. But the class label was in different file name y\_train.csv. for the sake of simplification, I first did merge the class label and the train dataset using python merge function.  

The features are a combination of numerical data and histogram bins data. The feature names are kept anonymized as mentioned for proprietary reasons. 39,178 data points belong to the class “0” and the remaining 722 belong to the class “1”. This tells us that we are dealing with a highly imbalanced dataset. 

![](Aspose.Words.c6b4be27-c9f4-481b-8858-505d56167342.001.png)

Another problem observed is that a large part of the data is missing. In extreme cases, some instances have 80% of the values missing. The dataset is classified as Missing Completely At Random (MCAR), as there is no relationship whether a data point is missing and any value in the dataset is missing or observed.  

**Performance Metric** 

For this project, I'll be using the F3 Score as a performance metric. The weighted harmonic mean of recall and precision makes up the F-beta score, which reaches its best value at 1 and its worst value at 0. The weight of recall in the total score is set by the beta parameter. While beta > 1 favors recall, beta 1 gives more weight to precision (beta -> 0 considers only precision, beta  -> +inf only recall). This is helpful since misclassification has a significant cost because an APS failure that is not identified might cause the truck to fail while it is operating and raise maintenance costs. 

![](Aspose.Words.c6b4be27-c9f4-481b-8858-505d56167342.002.png)

![](Aspose.Words.c6b4be27-c9f4-481b-8858-505d56167342.003.png)

When beta is 3 

![](Aspose.Words.c6b4be27-c9f4-481b-8858-505d56167342.004.png)

**Literature Review** 

Chun Gui, Qiang Lin **“Comparison of three data mining algorithms for potential 4G customers prediction” 2017**. in this the author uses 1,000,000 subscribers of user data as a dataset and 12 features. In data preprocessing, handling of missing value is an important task, there are three common ways to handle missing value: 1) remove these few cases, 2) use some strategies to fill the missing value, 3) to process the missing values with the aid of tools. But dataset only a few data has the missing values problem, and delete those a few data has no effect on the experimental results 

Cerqueira, Vítor, et al. “**Combining Boosted Trees with Metafeature Engineering for Predictive  Maintenance**.”  International  Symposium  on  Intelligent  Data  Analysis. Springer, Cham, 2016. 

This paper mentions that the authors’ approach to this problem consists of 4 steps. (i) A filter that excludes a subset of features and data points based on the number of missing values; (ii) A metafeature engineering procedure used to create new features based on existing information; (iii) a biased sampling method to deal with class imbalance problem (SMOTE); and (iv) use of boosted trees for classification. 

Features having a high percentage of missing values were removed. During their analysis, they found that some features had an extremity of 80% data missing, and 8 out of 170 features had more than 50% missing values. After removing the said features, it was seen that there were duplicate data points, indicating that the removed features have a little effect in getting a good score. 

I used BoxPlot Analysis (for each feature, compare each value to the typical value found in  that  feature),  Local  Outlier  Factor  (compare  data  point  to  it’s  local  neighborhood through density estimation) and Hierarchical Agglomerative Clustering (each step merges two similar group, and the last observation that are merged might be an outlier) for their metafeature engineering. 

[SMOTE ](https://www.geeksforgeeks.org/ml-handling-imbalanced-data-with-smote-and-near-miss-algorithm-in-python/#:~:text=SMOTE%20\(synthetic%20minority%20oversampling%20technique\)%20is%20one%20of%20the%20most,instances%20between%20existing%20minority%20instances.)is a method of duplicating the data points of the minority class of the imbalanced dataset, to balance it out. The use of SMOTE + MetaFeature Engineering with XGBOOST library was seen to give the best result. 

**Getting Started** 

First off, let’s import the required packages and read our training data. 

The dataset consists of 171 features, including the class label. Also, in the class label attribute. 

The class distribution graph shows a serious case of data imbalance, since out of the total 39,900 training points, about 39,718 points belong to the zero class and just 722 points belong to the one class. I can choose to up-sample the minority class data points or use a modified classifier to tackle this problem. Also, the percentage of missing data is significantly high in some features (As high as 82% in a feature). 

**Removing Single Valued Features** 

Out of the available features, the ones that have the same value for all data points do not hold much importance in improving performance of our model. Hence, we can discard those features. I can remove the features that have **standard deviation = 0.** 

One of the features, (‘***cd\_000**'*) is seen to have a constant value for all data points. I removed this feature. 

**Handling Missing Values** 

It is always a good practice to identify and replace missing values for each column in your input data prior to modeling your prediction task. This is called[ missing data imputation,](https://machinelearningmastery.com/statistical-imputation-for-missing-values-in-machine-learning/) or imputing for short. 

I have performed some basic handling of missing data in the following manner: 

- *I discarded features with more than 70% missing values.* 
- *For features with missing values less than 5%, I dropped those rows.* 
- *For features with missing values between 5–15%, I imputed those missing values using mean/median.* 
- *Now for the rest of the features with missing value% between 15–70% missing values, I used model-based imputation technique.* 
- From 39900 rows of data, 2693 rows have more than 70% missing data. 
- From 170 columns of feature, 8 columns have more than 70% missing data. 
- Number of features having missing values below 5%: 128. 

14 features had 5% to 15% of it’s values missing and are passed through sklearn’s SimpleImputer and the missing values are imputed using ‘*median*’. Following which, for features having 15% to 70% missing values, we will perform an Iterative model based imputation  technique  called [ MICE.](https://scikit-learn.org/stable/modules/impute.html#iterative-imputer)  At  each  step,  a  feature  with  missing  values  is designated as output y and the other feature columns are treated as inputs X.  

A regressor (I have used Ridge Regressor) is fit on (X, y) for known y. Then, the regressor is used to predict the missing values of y. This is done for each feature in an iterative fashion, and then is repeated for max\_iter (10 as default) imputation rounds. The results of the final imputation round are returned. 

- All the above models are saved and the preprocessing steps are performed on the test dataset. 

**Separating Features for Analysis** 

It was given to us that certain features are histogram bin information, and the prefix (letter before the ‘ \_ ‘) is the Identifier and the suffix is the bin\_id (Identifier\_Bin). 

To find the features that are contain histogram bin information, we know that all features from a single histogram have the same prefix. 

We can see that there are 7 sets of features having 10 bins each. In other words, there are **7 histograms divided into 10 bins each**. eg: Identifier ‘ag’ consists of ag\_000, ag\_001, ag\_002, ag\_003, ag\_004, ag\_005, ag\_006, ag\_007, ag\_008 and ag\_009. 

The Histogram Identifiers are: [‘ag’, ‘ay’, ‘az’, ‘ba’, ‘cn’, ‘cs’, ‘ee’]. 

There are **70 features that contain histogram bin information** and they are:  

` `['ag\_000', 'ag\_001', 'ag\_002', 'ag\_003', 'ag\_004', 'ag\_005', 'ag\_006', 'ag\_007', 'ag\_008', 'ag\_009', 'ay\_000', 'ay\_001', 'ay\_002', 'ay\_003', 'ay\_004', 'ay\_005', 'ay\_006', 'ay\_007', 'ay\_008', 'ay\_009', 'az\_000', 'az\_001', 'az\_002', 'az\_003', 'az\_004', 'az\_005', 'az\_006', 'az\_007', 'az\_008', 'az\_009', 'ba\_000', 'ba\_001', 'ba\_002', 'ba\_003', 'ba\_004', 'ba\_005', 'ba\_006', 'ba\_007', 'ba\_008', 'ba\_009', 'cn\_000', 'cn\_001', 'cn\_002', 'cn\_003', 'cn\_004', 'cn\_005', 'cn\_006', 'cn\_007', 'cn\_008', 'cn\_009', 'cs\_000', 'cs\_001', 'cs\_002', 'cs\_003', 'cs\_004', 'cs\_005', 'cs\_006', 'cs\_007', 'cs\_008', 'cs\_009', 'ee\_000', 'ee\_001', 'ee\_002', 'ee\_003', 'ee\_004', 'ee\_005', 'ee\_006', 'ee\_007', 'ee\_008', 'ee\_009'] 

**I select the top features from both the datasets using the complete imputed set. But the Analysis will be performed on the data having missing values.** 

**Histogram Feature Analysis** 

We will perform EDA on the top 15 features of the histogram dataset. For selecting the features, we will perform **Recursive Feature Elimination, using Random Forest Classifier** 

The top 15 features are : 

['ag\_001', 'ag\_002', 'ag\_003', 'ay\_005', 'ay\_006', 'ay\_008', 'ba\_002', 'ba\_003', 'ba\_004', 'cn\_000', 'cn\_004', 'cs\_002', 'cs\_004', 'ee\_003', 'ee\_005'] 

The[ PDF,](https://en.wikipedia.org/wiki/Probability_density_function)[ CDF ](https://en.wikipedia.org/wiki/Cumulative_distribution_function)and[ Box plots ](https://en.wikipedia.org/wiki/Box_plot)of each of these features to try to understand the dis[tribu](https://en.wikipedia.org/wiki/Probability_density_function)[tion of](https://en.wikipedia.org/wiki/Cumulative_distribution_function) our[ data. The ](https://en.wikipedia.org/wiki/Box_plot)observations made are as follows: 

- Plots  of  features  **ag\_003,  ay\_008,  ba\_002,  ba\_003,  ba\_004,  cn\_004, cs\_002, cs\_004, ee\_003 and ee\_005** show us that the Lower values of the features indicate no failure in the APS component. A higher value clearly indicates an APS component failure 
- Around 99% values of feature **ag\_001 and ay\_005**, where there is no failure in the APS component, are 0. 
- We can say that in these top features, a higher value may indicate a failure in the truck’s Air Pressure System 
- But, there are few cases when the values are higher than usual, but still do not lead to APS failure. Example: Feature **ee\_005** 

Taking into consideration how each feature is correlated with the target variable (‘class’), we can observe that feature **‘ay\_005’** is the most uncorrelated feature among our top attributes. I perform further **Bivariate Analysis** on how the other top features vary w.r.t feature ‘ay\_005’. 

- **ag\_002, ag\_001, cn\_000**: It can be seen from the scatter plot that for any  value  of  the  other  top  features,  there  is  failure  in  the  APS component (class label = 1) when the value in feature ‘ay\_005’ is nearly 0. 

**Numerical Feature Analysis** 

I perform EDA on the top 15 features of the histogram dataset. For selecting the features, I perform **Recursive Feature Elimination, using Random Forest Classifier** 

The top 15 features are : 

['aa\_000', 'al\_000', 'am\_0', 'ap\_000', 'aq\_000', 'bj\_000', 'bu\_000', 'bv\_000', 'ci\_000', 'cj\_000', 'cq\_000', 'dg\_000', 'dn\_000', 'do\_000', 'dx\_000'] 

The[ PDF,](https://en.wikipedia.org/wiki/Probability_density_function)[ CDF ](https://en.wikipedia.org/wiki/Cumulative_distribution_function)and[ Box plots ](https://en.wikipedia.org/wiki/Box_plot)of each of these features to try to understand the distribution of our data. The observations made are as follows: 

- **aa\_000 :** If there is no failure in the APS (class label = 0), about 95% of the points have a value below 0.1x1e6. A higher value than that usually indicates a failure in the APS component. 
- **al\_000, am\_000 :** The values of instances of failure and non-failure of the  APS  component  are  not  clearly  seperable  in  this  feature. Although points of the failure cases do have a slightly higher value. 
- **ap\_000, aq\_000, bj\_000, bu\_000 :** Instances of failure have a higher value, compared to non-failure cases. But there are few instances of non-failure of the APS component, that see higher values in this feature. 
- In all features, except **dg\_000, cj\_000, am\_0 and al\_000**, the higher values in the features usually indicate failure in APS component. But due to the Imbalanced nature of the data this may not be certain. 

Taking into consideration how each feature is correlated with the target variable (‘class’), we can observe that feature **‘dx\_000’** is the most uncorrelated feature among our top attributes. We can perform further **Bivariate Analysis** on how the other top features vary w.r.t feature ‘dx\_000’. 

The main observation in all plots here is that for any value of the remaining features, if the feature ‘dx\_000’ has a low value (nearly 0 ), it MAY INDICATE that there is a failure in the APS component (class label=1). 

**Preparing Our Data (Standardizing + Synthetic Minority Oversampling Technique (SMOTE) + UnderSampling)** 

SMOTE: this technique was described by Nitesh Chawla, et al. in their 2002 paper named for the technique titled “SMOTE: Synthetic Minority Over-sampling Technique.” SMOTE works by selecting examples that are close in the feature space, drawing a line between the examples in the feature space and drawing a new sample at a point along that line. 

Typically, standardizing a vector entails eliminating a measure of position and dividing by a measure of scale. For example, if the vector comprises Gaussian random values, you may remove the mean and divide by the standard deviation to get a "standard normal" random variable with 

mean 0 and standard deviation 1. I scaled my data using sklearn’s MinMaxScaler**.** 

A problem with imbalanced classification is that there are too few examples of the minority class for a model to effectively learn the decision boundary. One way to solve this problem is to oversample the examples in the minority class. 

[The combination of SMOTE and under-sampling performs better than plai n under-sampling. ](https://arxiv.org/abs/1106.1813) 

Finally, we have *22,060 points belonging to zero class and 11,030 points belonging to the one class*. I will pass our scaled dataset through the linear models (Logistic Regression and Support Vector Machines). 

**Experimenting with Classical ML models** 

Now  that  I  have  prepared  our  performed  our  EDA,  data-preprocessing  and  feature engineering. I pass our data through various models, perform hyperparameter tuning, and evaluate each of them based on our performance metric (Macro-F3 Score) and Confusion Matrix. The different models that I will be trying out here are Logistic Regression, Support Vector Machines, Naive Bayes, Decision Trees, Random Forest, Gradient Boosted Decision Trees, Adaboost Classifier and a Custom Ensemble. 

As a Baseline Model, I will predict all class labels to be 0 (majority class) and calculate the F3 score for the same. I used sklearn’s DummyClassifier to obtain our baseline results. 

For our Custom Ensemble: 

1. Split the train set into D1 and D2 (50–50). 
1. From D1, perform sampling with replacement to create *d1 , d2 , d3 …. dk* (k samples). 
1. Now, create ‘k’ models and train each of these models with each of these k samples. 
1. Pass the D2 through each of these ‘k’ models, which gives us ‘k’ predictions for D2, from each of these models. 
1. Using these ‘k’ predictions create a new dataset, and for D2, since we already know it’s corresponding target values, we can now train a meta model with these ‘k’ predictions as features. 
1. For model evaluation, I passed our test set through each of the base models and get ‘k’ predictions. Then I create a new dataset with these ‘k’ predictions and pass it to the previously trained metamodel to get our final prediction. 
1. Now using this final prediction as well as the targets for the test set,  I have calculate the models performance score. 

I used Decision Trees as our base model and GBDT as the metamodel. This is a custom implemented model. 

After performing hyperparameter tuning and experimenting various models, we see that the Custom Stacking Ensemble  works best as it gets the highest Macro-F3 Score (as seen below). 

Summary of Modelling 

|              Model                                                            |    Macro-F3 Score   | 

`                 `+---------------------------------+---------------------+ 

| Baseline                                               |  0.4991117527880268 | | Naive Bayes                                                     |  0.196557624570462  | 

| Decision Trees                                                       |  0.4878322522903948 | | Logistic Regression                                    | 0.10628388913354114 | | Random Forest                                                         | 0.4563794646429734 | | SVM using HingeLoss                                           | 0.08563794646429734 | | Custom Stacking Ensemble                      |  0.5029621160391942 | | Gradient Boosted Decision Trees                     |  0.4958483850327113 | | Adaptive Boosting                      |  0.486046156578381  | 

**Reference** 

1. [https://towardsdatascience.com/is-f1-the-appropriate-criterion-to-use-what-about-f2-f3-f-beta-4bd 8ef17e285 ](https://towardsdatascience.com/is-f1-the-appropriate-criterion-to-use-what-about-f2-f3-f-beta-4bd8ef17e285)
1. [https://towardsdatascience.com/predicting-a-failure-in-scanias-air-pressure-system-aps-c260bcc4d 038 ](https://towardsdatascience.com/predicting-a-failure-in-scanias-air-pressure-system-aps-c260bcc4d038)
1. Combining Boosted Trees with Metafeature Engineering for Predictive Maintenan ce
1. Comparison of three data mining algorithms for potential 4G customers predictio n” 2017
1. Kaggle.com 
1. YouTube videos 
