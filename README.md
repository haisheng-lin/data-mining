## Data Mining

### Project 1

In this assignment, you will study the application of the k-nearest neighbor, neural network and SVM classifiers on two real-world classification problems. The datasets to be used for this assignment are uploaded under the “Datasets” folder. x\_train, y\_train, x\_test and y\_test denote the training features, training labels, testing features and testing labels respectively. In x\_train and x\_test, each row denotes a data sample and each column denotes a feature.

#### Problem 1

The Human Activity Recognition dataset was created from experiments carried out on a group of 30 volunteers to recognize human activities using smart phone data. Each person performed six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) wearing a smartphone (Samsung Galaxy S II) on the waist. Using its embedded accelerometer and gyroscope, 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz were captured. The data was processed using signal processing algorithms to extract feature vectors of dimension 561. The training set contains 7,352 samples and the test set contains 2,947 samples.

Implement the k-nearest neighbor algorithm with k = 5 on this dataset. Use the simple Euclidean distance measure to compute the distance between two samples.
Train an SVM classifier with a polynomial kernel with parameter 2 on the training set and test on the test set. You need to train one SVM for each class; for predicting a test sample, use the maximum of the values returned by all the SVMs to decide the final class.

Report the percentage accuracy on the test set using each method.

#### Problem 2

The VidTIMIT dataset consists of video and audio recordings of 43 subjects reciting short sentences. In this assignment, we will use a subset of the dataset with 25 subjects. We will also use only the video modality. The videos were sliced into images and the discrete cosine transform function was used to extract feature vectors of dimension 100 from each image. The training set contains 3,500 samples and the test set contains 1,000 samples. Our objective is to recognize a subject from a given image.

Implement the k-nearest neighbor algorithm with k = 5 on this dataset. Use the simple Euclidean distance measure to compute the distance between two samples.
Use the training set to train a feedforward neural network with 1 hidden layer containing 25 neurons. Report the percentage accuracy on the test set. You can use Matlab’s in-built neural network related functions for this part. Take a look at the “feedforwardnet” function.

Train an SVM classifier with a polynomial kernel with parameter 2 on the training set and test on the test set. You need to train one SVM for each class; for predicting a test sample, use the maximum of the values returned by all the SVMs to decide the final class.

Report the percentage accuracy on the test set using each method.

---

### Project 2

In this assignment, you will study the application of the k-means clustering algorithm and active learning for classification problems. The datasets to be used for this assignment are uploaded under the “Datasets” folder. For the active learning problem, the necessary code files are also uploaded in the same folder. In all data files, each row denotes a sample and each column denotes a feature.

#### Problem 1

The seeds dataset contains measurements of geometrical properties of kernels belonging to different varieties of wheat. It has 210 data samples and each sample is described by 7 attributes (features).

Implement the k-means clustering algorithm on this dataset. Use the simple Euclidean distance to compute the distance between any two samples. Start with a random initialization of the centroids and iterate until convergence. The algorithm is assumed to have converged if the number of iterations exceeds 100 OR the change in the sum of squared errors (SSE) between two successive iterations is less than 0.001. Run the algorithm for k = 3, 5 and 7. For each value of k, run the algorithm with 10 random initializations of the centroids. Report the average SSE value (averaged over the 10 initializations) for each value of k.

#### Problem 2

A practical challenge in training a reliable machine learning model is the requirement of a large amount of labeled training data. While gathering unlabeled data is cheap and easy, labeling the data is an expensive process in terms of time, labor and human expertise. Active learning algorithms automatically identify the unlabeled samples that need to be labeled to train a reliable machine learning model. When exposed to large amounts of unlabeled data, they select the salient and exemplar samples for manual annotation. In this project, you will study the application of active learning in the classification setting, on two datasets.

**Classification Model:** We will use the Logistic Regression (LR) model in this project (Note: LR is a classifier, not a regression model). We have not covered LR in class and so, you are given the necessary functions to train and test an LR model. Copy all the files in your current Matlab directory. The following two functions should be used for training and testing:

**train_LR_Classifier:** Function to train a LR classifier. It takes the training samples, labels and the number of classes as inputs and returns the parameters of the trained model.

**test_LR_Classifier:** Function to test the LR classifier. It takes a test sample, the trained weights and the number of classes as inputs and returns a vector containing the probabilities of the sample corresponding to all the classes.

**Active Learning Outline:** The functioning of an active learning system can be outlined as follows:

**Given:** Small amount of initial training set (training samples and labels); large amount of unlabeled data with NO labels (the labels of the unlabeled samples should be used only in response to a user query); a test set to judge the performance of the system; a batch size k (number of unlabeled samples to be queried in each iteration) and the number of iterations N

**Loop over N iterations**

Step 1: Train a machine learning model using the current training set

Step 2: Apply the model on the test set and obtain the accuracy

Step 3: Apply the model on the unlabeled set and select a batch of k unlabeled samples based on an active learning strategy (see below)

Step 4: Obtain the labels of the selected k samples from a human expert (you will use the provided labels of the unlabeled samples here to simulate the human expert)

Step 5: Remove those samples from the unlabeled set and add them to the current training set

**End Loop**

In each iteration, you will get an accuracy value. After the active learning iterations are over, you will get a vector containing N accuracy values. Our objective is to study the rate of increase of accuracy over iterations. This is expressed as a plot of the number of iterations (on the x-axis) and
the accuracy (on the y-axis). An active learning algorithm is better than another if its accuracy grows at a faster rate.

In this project, use k = 10 and N = 50. Also, for each dataset, repeat the process 3 times using 3 different initial training sets, unlabeled sets and test sets and report the average accuracy results.

**Active Learning Strategies:** We will study the performance of two active learning strategies in this project:

(1) Random Sampling: Select a batch of k samples from the unlabeled set at random

(2) Uncertainty-based Sampling: For each unlabeled sample, compute the classification entropy as e = - Σ pi log (pi), where i runs from 1 to the number of classes and pi is the probability that the sample belongs to class i. Select the k samples producing the highest entropy.

For each dataset, plot one graph containing the performance of Random Sampling and Uncertainty-based Sampling (on the same graph). Write your observations in your report.

**Datasets:** We will use two facial expression recognition datasets in this problem: MindReading and MMI. They both contain samples belonging to 6 classes. Each dataset contains an initial labeled training set, an unlabeled set and a test set. Also, each experiment needs to be run 3 times and the average results should be reported.

---

### Project 3

In this assignment, you will study the application of multi-label learning and ensemble learning on real-world applications. The datasets to be used for this assignment are uploaded under the “Datasets” folder. x\_train, y\_train, x\_test and y\_test denote the training features, training labels, testing features and testing labels respectively. In x\_train and x\_test, each row denotes a data sample and each column denotes a feature.

#### Problem 1

In a **multi-class classification** problem, there are multiple classes in the dataset, but each data sample can belong to only one class. **Multi-label classification** is a generalization of multi-class classification, where each data sample can belong to multiple classes simlutaneously. For instance, consider the problem of classifying an outdoor image of a scene. Suppose the possible classes are beach, mountain, field and sunset. It is possible for a particular image to contain both beach and mountain or beach, mountain and sunset all together. The objective of multi-label learning is to predict all the classes present in a data sample.

The Scene dataset consist of 2407 images of an outdoor scene, where each image is represented by a feature vector of dimesnion 294. Also, there are 6 classes in the problem and an image can belong to one or more of the 6 classes. The dataset has been divided into a training set (with 1500 samples) and a test set (with 907 samples). Each row of X_train and X_test denotes a sample and each column denotes a feature. Each row of y_train (y_test) denotes the labels of the corresponding training (testing) sample, where 1 means the class is present and 0 means the class is absent. For instance, in the training set, sample 462 belongs to classes 4 and 5.

One strategy to solve a multi-label learning problem is to train an SVM separately for each class. To predict a test sample, each SVM is applied separately on it. A positive output indicates that the
corresponding class is present and a negative output indicates that it is absent. The accuracy is computed using the following expression:

![alt text](http://os0wqvgrz.bkt.gdipper.com/DataMining_Project3_Problem1.png "equation")

where T is the true class label vector of a test sample and P is the predicted class label vector. Train an SVM classification model on the training set and test on the test set. Report the percentage accuracy on the test set using the following classification models: (i) SVM with polynomial kernel with parameter 2 and (ii) SVM with Gaussian kernel with parameter 2.

#### Problem 2

The Handwritten Digits dataset contains images of handwritten digits. The grayscale values of the pixels in the images are concatenated to yield feature vectors of dimension 64. There are 10 classes in the problem, corresponding to the 10 digits. The training set contains 500 samples and the test set contains 3251 samples. Implement the following algorithms on this dataset (you are allowed to re-use your code from previous submissions):

(i) k-nearest neighbor with k = 7  
(ii) SVM with a polynomial kernel of degree 2  
(iii) Feedforward neural network with a single hidden layer with 25 neurons

Report the accuracies of the individual models on the test set. Also, report the accuracy of the entire ensemble on the test set, where the prediction of each sample is obtained by taking a majority vote on the predictions of the individual models.