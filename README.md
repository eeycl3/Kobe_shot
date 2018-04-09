# Kobe_shot

#### To make it more convenient, we put the relevant files in the same folder. If you need to run the entire code, place the code in the top-level repository.**

### Description

Throughout Kobe's long career, he won numerous honors. At the same time, NBA also recorded all the data of his shot attempts. In this project, Kaggle randomly picks out 1/3 of the data as a test set. We used the remaining 2/3 data as the training set to predict on the test set and came up with the final model. We implemented recursive feature elimination (RFE) method and a series of feature extraction methods to pick out the most important features, and then utilized support vector machine, random forest, artificial neural networks algorithms, as well as voting to generate a comprehensive classification model. Extensive comparative analysis regarding the feature engineering methods and classification models were also delivered. 

### Goal and method

The final goal of this project is a binary classification, and throughout the process, we are aiming to mainly exploring the feature engineering techniques and several classification algorithms. We used recursive feature elimination(RFE) to do feature selection, implemented PCA, LLE, ISOMAP, TSNE to realize feature extraction part, and used SVM, random forest, ANNs, as well as voting method to do classification.

### Conclusion and Discussion

Based on the result we can find out that the pattern existing in the classification is weak. Shooting accuracy is highly influenced by random human factors. That is the reason why the accuracy of our results is around 67\%. In this project, we can conclude that using different algorithms does not have much effect on the accuracy of the results, but the time spent by the different algorithms themselves is quite different. For SVM classifier, it has the best accuracy among all the algorithms but it takes a long time for classification at the same time. Regarding this situation, it can be concluded that SVM is suitable for low dimensional and small dataset; For random forest classifier, it has relatively better accuracy with less classification time. In addition, it avoids overfitting problems. In this case, if classification time is what we need to prioritize, random forest can be considered as the best solution. For ANN, it takes the longest time in training but less time in predicting. In this situation, it may not be suitable for this classification problem because its long training time and mediocre accuracy performance.

