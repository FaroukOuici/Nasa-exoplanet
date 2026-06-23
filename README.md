# Nasa-exoplanet
This repository contains an XGboost machine learning model, along with a collection of space data from official libraries containing planetary and stellar data from NASA. This data is used to train the model, which is a linear regression model capable of predicting a planet's mass with excellent accuracy
The model get 89% accuracy with this parameters and you can test it by your self

## How to use the script?:
First you need to know that the main code which is `training.py` is ready to work you can do just a little changes in the init function which get the data from nasa api through the astroquery library by change the word top 10000 to top x where x is any number that is in the limit the library and servers allows

Second if you want just to train the model in different features like blackholes or galaxies you just need to change the source of the data (**Please notice that if you want to train it in blackholes you need to change the features,labels and the source but the model stay the same**)

Third the model parameters is not the perfect one but it do the job it has very precisly,But of course you can change it in the `train` function where the n_estimators is the number of the trees and the max_depth is the how much depth can one tree reach (Increasing that increase the probability for the model to overfit the data),subsample parameter make sure that every time the model create new tree it gives it 80% of the data randomly only,colsample_bytree parameter is do the same as subsample but for the columns

## Explain how the model works:

### Mean Squared Error:
**The equation:**

$L(y,\hat{y}) =\frac{1}{2}(y-\hat{y})^2$

This function is the one we use in any linear regression model to calculate the loss or difference between the model prediction and the true value of the label **here is an explaining for the symbols:**

$y$ : is the model prediction 

$\hat{y}$ : is the label true value 

$\frac{1}{2}$ : we use it to make derivation easier because it will go with the power 2

${y-\hat{y}}^2$ : We raise the defference to the power 2 to increase the fault penalty **e.g: 2 becomes 4** and to get rid of the minus sign

### The Gradient Descent:

Here the things are different in the linear models we use the familliar derivation we all learn which is : 

### $\frac{\partial L}{\partial w_i} - \alpha$

Here the model use different type which is derivation but for the weights in the leaf of every tree to correct the previous tree which we describe by:

$g_i = \frac{\partial(y_i,\hat{y}^{t-1})}{\partial(\hat{y}^{t-1})}$

By knowing this things the model become a something clear and we can understand why he give us a bad results or good one instead of changing the parameters randomly

## Why the model give us 89% accuracy?:

The main reason that the model give us this accuracy is that we make sure that there is no data-leak (**i will explain how to know it below**) and we clean the data precisly,also because the model configuration is important and i don't suggest raise the max_depth to more than 6 for this type of tasks

### Data Leak:

Data leak means that the model see the answers directly or indirectly with the questions that why when you see your model give you 99% accuracy try to see if one of this mistakes is in your code:

- You combine two or mores columns or features to produce new one and you pass this two columns as inputs to the model to predict the one you create with which is the new column
- You pass the labels to the model as inputs without noticing

## The End:

In the end i hope this project help anyone who loves astronomy and physics to predict the mass of the planets or the radius of the planets or even the star features,Also remembre that you can create a great model without understand it but you can't handle the problems when it appears in it without knowing what's going on the model.

