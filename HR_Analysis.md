Human Resources Analysis
================
Tasiopoulos Sotos
3 Σεπτεμβρίου 2017

<br><br>

This is an analysis on data of employees. We are going to try predict whether an employee is going to leave the company or not. The data associated is downloadable at the link below :

<https://www.kaggle.com/ludobenistant/hr-analytics/downloads/human-resources-analytics.zip>

<br><br>

Since we have downloaded the file, we will start by loading the appropriate libraries in R.

``` r
library(ggplot2) # Data visualization
library(DMwR) # Making new artificial data
```

    ## Loading required package: lattice

    ## Loading required package: grid

``` r
library(xgboost) # machine learning algorithm
library(ROCR) # For model evaluating
```

    ## Loading required package: gplots

    ## 
    ## Attaching package: 'gplots'

    ## The following object is masked from 'package:stats':
    ## 
    ##     lowess

<br><br>

Next, we are going to read the file and write it to a dataframe.

``` r
hr <- read.csv(file = "HR_comma_sep.csv", header = T, sep = ",")
```

<br><br>

Let's perform some basic tests on the data.

<br>

Let's see if there are any na's in the file:

``` r
sum(is.na(hr))
```

    ## [1] 0

Hopefully, there are no na's in the file.

<br>

Next, let's see the structure of the file.

``` r
str(hr)
```

    ## 'data.frame':    14999 obs. of  10 variables:
    ##  $ satisfaction_level   : num  0.38 0.8 0.11 0.72 0.37 0.41 0.1 0.92 0.89 0.42 ...
    ##  $ last_evaluation      : num  0.53 0.86 0.88 0.87 0.52 0.5 0.77 0.85 1 0.53 ...
    ##  $ number_project       : int  2 5 7 5 2 2 6 5 5 2 ...
    ##  $ average_montly_hours : int  157 262 272 223 159 153 247 259 224 142 ...
    ##  $ time_spend_company   : int  3 6 4 5 3 3 4 5 5 3 ...
    ##  $ Work_accident        : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ left                 : int  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ promotion_last_5years: int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ sales                : Factor w/ 10 levels "accounting","hr",..: 8 8 8 8 8 8 8 8 8 8 ...
    ##  $ salary               : Factor w/ 3 levels "high","low","medium": 2 3 3 2 2 2 2 2 2 2 ...

The file consists of 10 variables. Variable "Left", is the target variable that indicates if an employee has left or not.

<br>

Let's run some tests on the variables, to watch their frequencies.

``` r
# Plotting
qplot(hr$average_montly_hours, geom = "histogram",col=I("blue"), main = "Monthly hours", xlab = "Average monthly hours", ylab = "Frequency",bins = 20)
```

![](HR_Analysis_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-5-1.png)

``` r
qplot(hr$satisfaction_level, geom = "histogram", col=I("blue"), main = "Satisfaction level", xlab = "Levels from 0-1", ylab = "Frequency",bins = 20)
```

![](HR_Analysis_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-5-2.png)

``` r
qplot(hr$last_evaluation, geom = "histogram", col=I("blue"), main = "Last evaluation", xlab = "Evaluation from 0-1", ylab = "Frequency",bins = 20)
```

![](HR_Analysis_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-5-3.png)

``` r
qplot(hr$number_project, geom = "histogram", col=I("blue"), main = "Number of projects", xlab = "Projects", ylab = "Frequency",bins = 20)
```

![](HR_Analysis_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-5-4.png)

``` r
qplot(hr$time_spend_company, geom = "histogram", col=I("blue"), main = "Time in company", xlab = "Time in company in years", ylab = "Frequency",bins = 20)
```

![](HR_Analysis_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-5-5.png)

``` r
qplot(hr$Work_accident, geom = "histogram", col=I("blue"), main = "Work accident", xlab = "Work accident", ylab = "Frequency",bins = 20)
```

![](HR_Analysis_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-5-6.png)

``` r
qplot(hr$promotion_last_5years, geom = "histogram", col=I("blue"), main = "Been promoted last 5y.", xlab = "0 = no, 1 = yes", ylab = "Frequency",bins = 20)
```

![](HR_Analysis_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-5-7.png)

``` r
qplot(hr$sales, geom = "bar", colour = sales, main = "Sales", xlab = "Sales", ylab = "Frequency", data = hr)
```

![](HR_Analysis_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-5-8.png)

``` r
qplot(hr$salary, geom = "bar", colour = salary, main = "Salary", xlab = "Salary in 3 levels", ylab = "Frequency", data = hr)
```

![](HR_Analysis_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-5-9.png)

<br><br>

Next, we will look at the target variable "Left".

``` r
qplot(hr$left, geom = "histogram", col=I("blue"), main = "Has he left?", xlab = "0 = no, 1 = yes", ylab = "Frequency",bins = 20)
```

![](HR_Analysis_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-6-1.png)

We are noticing that the rows of the ones who left, are approx. 30% of the ones who stayed at the company. That means, that we later, will have to construct artificial data of employees, in order to have a balanced class nad get a good result with the machine learning algorithm.

<br><br>

Now, we are subsetting the data and getting only the people who left.

``` r
hr_left <- subset(hr, left == 1)
```

<br>

We are focusing on some of their characteristics.

``` r
qplot(hr_left$promotion_last_5years, geom = "histogram", col=I("red"), main = "Been promoted last 5y.", xlab = "0 = no, 1 = yes", ylab = "Frequency",bins = 20)
```

![](HR_Analysis_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-8-1.png)

``` r
qplot(hr_left$satisfaction_level, geom = "histogram", col=I("blue"), main = "Satisfaction level", xlab = "Levels from 0-1", ylab = "Frequency",bins = 20)
```

![](HR_Analysis_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-8-2.png)

Notice: only a few of the employees, stayed at the company even if they hadn't been promoted for the 5 last years.

<br>

Now let's examine the corellation between the numerical variables.

``` r
corr.matrix <- cor(hr[,-c(9,10)])
round(corr.matrix,2)
```

    ##                       satisfaction_level last_evaluation number_project
    ## satisfaction_level                  1.00            0.11          -0.14
    ## last_evaluation                     0.11            1.00           0.35
    ## number_project                     -0.14            0.35           1.00
    ## average_montly_hours               -0.02            0.34           0.42
    ## time_spend_company                 -0.10            0.13           0.20
    ## Work_accident                       0.06           -0.01           0.00
    ## left                               -0.39            0.01           0.02
    ## promotion_last_5years               0.03           -0.01          -0.01
    ##                       average_montly_hours time_spend_company
    ## satisfaction_level                   -0.02              -0.10
    ## last_evaluation                       0.34               0.13
    ## number_project                        0.42               0.20
    ## average_montly_hours                  1.00               0.13
    ## time_spend_company                    0.13               1.00
    ## Work_accident                        -0.01               0.00
    ## left                                  0.07               0.14
    ## promotion_last_5years                 0.00               0.07
    ##                       Work_accident  left promotion_last_5years
    ## satisfaction_level             0.06 -0.39                  0.03
    ## last_evaluation               -0.01  0.01                 -0.01
    ## number_project                 0.00  0.02                 -0.01
    ## average_montly_hours          -0.01  0.07                  0.00
    ## time_spend_company             0.00  0.14                  0.07
    ## Work_accident                  1.00 -0.15                  0.04
    ## left                          -0.15  1.00                 -0.06
    ## promotion_last_5years          0.04 -0.06                  1.00

<br>

We don't see any high correlated variables, so we include all of them in the model.

<br>

We have seen that the variables "Sales" and "Salary", are factor variables. Since the algorithm we use, needs to have only numerical data, we will construct two more variables that substitute the "Sales" and "Salary".

``` r
# The 2 new variables
hr$sales_new <- NULL
hr$salary_new <- NULL

# Getting the factor levels of sales
levels(hr$sales)
```

    ##  [1] "accounting"  "hr"          "IT"          "management"  "marketing"  
    ##  [6] "product_mng" "RandD"       "sales"       "support"     "technical"

``` r
# Every sales department is presented as a number now :
# "accounting" = 1, "hr" = 2, etc

for(i in 1:nrow(hr)){
  
  if(hr$sales[i] == "accounting")
    hr$sales_new[i] = 1
  else
    if(hr$sales[i] == "hr")
      hr$sales_new[i] = 2
    else
      if(hr$sales[i] == "IT")
        hr$sales_new[i] = 3
      else
        if(hr$sales[i] == "management")
          hr$sales_new[i] = 4
        else
          if(hr$sales[i] == "marketing")
            hr$sales_new[i] = 5
          else
            if(hr$sales[i] == "product_mng")
              hr$sales_new[i] = 6
            else
              if(hr$sales[i] == "RandD")
                hr$sales_new[i] = 7
              else
                if(hr$sales[i] == "sales")
                  hr$sales_new[i] = 8
                else
                  if(hr$sales[i] == "support")
                    hr$sales_new[i] = 9
                  else
                    if(hr$sales[i] == "technical")
                      hr$sales_new[i] = 10
                    
                    
}

# Getting the factor levels of salary
levels(hr$salary)
```

    ## [1] "high"   "low"    "medium"

``` r
# Every salary level, is presented as a number now :
# "low" = 1, "medium = 2", etc

for(i in 1:nrow(hr)){
  
  if(hr$salary[i] == "high")
    hr$salary_new[i] = 1
  else
    if(hr$salary[i] == "medium")
      hr$salary_new[i] = 2
    else
      if(hr$salary[i] == "low")
        hr$salary_new[i] = 3
}
```

<br><br>

Now we will separate the data into 2 parts. The train data and the test data. Train part will be the 80% of the data, and the rest 20%, will be the test data.

<br>

We will also set the target variable "Left" to factor, to be compatible with the algorithm of xgboost.

``` r
# left is a factor now
hr$left <- as.factor(hr$left)

# We remove the 2 variables "Sales" and "Salary", since we don't need them anymore

hr <- hr[,-c(9,10)]

set.seed(2017)

# Creating a random list of numbers

random <- sample(1:14999,3000)


hr.test <- hr[random,]
hr.train <- hr[-random,]


# Taking a look at the proportion of the ones who left and ones who stayed

table(hr.train$left)
```

    ## 
    ##    0    1 
    ## 9113 2886

Since "Left" is an unbalanced class, we will create artficial data in order to have equal number of cases in "Left"

``` r
hr_smoted <- SMOTE(left ~ ., data = hr.train, perc.over = 300, perc.under = 133, k=2)
```

Taking a look at the variable "Left" :

``` r
table(hr_smoted$left)
```

    ## 
    ##     0     1 
    ## 11515 11544

Now the class is balanced.

<br> <br>

We may proceed with the algorithm. The algorithm, that is going to be used, is XGBoost, a winning algorithm in many competitions in Kaggle.com. This algorithm belongs to the family of the tree algorithms and it is used for supervised learning problems.

``` r
# Making a matrix compatible with the algorithm
dtrain <- xgb.DMatrix(as.matrix(hr_smoted[,-7]), label = as.matrix(hr_smoted[,7]))

# Setting a random seed
set.seed(140871)

# Running the model
hr.model <- xgboost(data = dtrain, max.depth = 14,
              eta = 0.3, nthread = 2, nround = 300, 
              objective = "binary:logistic",subsample = 0.8,
              gamma = 5,colsample_bytree = 0.8,print.every.n = 20)
```

    ## [0]  train-error:0.016306
    ## [20] train-error:0.005334
    ## [40] train-error:0.003990
    ## [60] train-error:0.004293
    ## [80] train-error:0.004207
    ## [100]    train-error:0.003990
    ## [120]    train-error:0.003946
    ## [140]    train-error:0.003860
    ## [160]    train-error:0.003860
    ## [180]    train-error:0.003773
    ## [200]    train-error:0.003773
    ## [220]    train-error:0.003773
    ## [240]    train-error:0.003730
    ## [260]    train-error:0.003730
    ## [280]    train-error:0.003730

<br>

The model is created. Let's see how accurate can this model be, when predicting if employee will leave the company or not.

``` r
predictions <- predict(hr.model,as.matrix(hr.test[,-7]))

predictions <- ifelse(predictions > 0.5, 1, 0)

conf.table <- table(hr.test$left,predictions)

conf.table
```

    ##    predictions
    ##        0    1
    ##   0 2293   22
    ##   1   23  662

<b>

Interpreting the confusion table : </b>

<br>

Employees who stayed and right predicted:

    ## [1] 2293

Employees who left and right predicted:

    ## [1] 662

Employees that stayed but algorithm predicted "left":

    ## [1] 22

Employees that left but algorithm predicted "stay":

    ## [1] 23

<br>

Overall hit of the algorithm:

    ## [1] 0.985

Not bad at all.

<br><br>

Now let's see which variables were the ones that gave the most of information for our model.

``` r
importance_matrix <- xgb.importance(colnames(hr_smoted), model = hr.model)
```

``` r
xgb.plot.importance(importance_matrix)
```

![](HR_Analysis_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-22-1.png)

We clearly see that "Satisfaction level" gave us the most of information for the model, followed by "time spent in company" and "average monthly hours".

<br> <br>

Now let's evaluate our model.

First, plotting Fpr against Tpr:

``` r
plot (RP.perf, main = "ROC Curve", col = "blue")
```

![](HR_Analysis_files/figure-markdown_github-ascii_identifiers/unnamed-chunk-24-1.png)

<br> <br>

Calculating the AUC value:

``` r
auc.tmp <- performance(pred,"auc")
auc <- as.numeric(auc.tmp@y.values)
auc
```

    ## [1] 0.9784601

<br> <br> \* This analysis, was created in Rmarkdown and RStudio and it was only made for educational and exhibitional purposes.
