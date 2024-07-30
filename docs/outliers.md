# Outliers

### Why Detect Outliers?
- Detect incorrect data
- Detect extreme cases
  - interesting / hard cases
  - fraud
  - ...

### Types of Methods
- **Unsupervised**: Chosen here
- **Semisupervised**
- **Supervised**

## Outlier Detection Techniques

### Outlier Detection Methods

#### Linear Models
- **PCA (Principal Component Analysis, 2003)**: Sum of weighted projected distances to the eigenvector hyperplanes
- **MCD (Minimum Covariance Determinant, 1999)**: Uses the Mahalanobis distances as the outlier scores
- **OCSVM (One-Class Support Vector Machines, 2001)**: Classifies data points as similar or different from the training set
- **LMDD (Deviation-based Outlier Detection, 1996)**: Detects outliers based on deviation from a model

#### Proximity-Based Methods
- **LOF (Local Outlier Factor, 2000)**: Identifies density-based local outliers
- **COF (Connectivity-Based Outlier Factor, 2002)**: Measures the change in density
- **CBLOF (Clustering-Based Local Outlier Factor, 2003)**: Uses clustering to identify outliers
- **LOCI (Local Correlation Integral, 2003)**: Fast outlier detection using local correlation integral
- **HBOS (Histogram-based Outlier Score, 2012)**: Uses histograms to score outliers
- **kNN (k Nearest Neighbors, 2000)**: Uses distance to the kth nearest neighbor as the outlier score
- **AvgKNN (Average k Nearest Neighbors, 2002)**: Uses the average distance to k nearest neighbors as the outlier score

#### Probabilistic Methods
- **FastABOD (Fast Angle-Based Outlier Detection, 2008)**: Uses angles between data points for detection
- **COPOD (Copula-Based Outlier Detection, 2020)**: Uses copulas for anomaly detection
- **SOS (Stochastic Outlier Selection, 2012)**: Probabilistic method based on a stochastic approach

#### Outlier Ensembles
- **IForest (Isolation Forest, 2008)**: Isolates observations by randomly partitioning data
- **LSCP (Locally Selective Combination of Parallel Outlier Ensembles, 2019)**: Combines multiple outlier detection methods
- **LODA (Lightweight On-line Detector of Anomalies, 2016)**: Uses random projections for anomaly detection

#### Neural Networks
- **VAE (Variational AutoEncoder, 2013)**: Uses reconstruction error as the outlier score
- **MO_GAAL (Multiple-Objective Generative Adversarial Active Learning, 2019)**: Uses generative models for outlier detection

### Statistical Methods
- **Z-Score**: Measures the distance from the mean in standard deviations
- **Modified Z-Score**: Adjusted Z-Score using median and MAD
- **IQR (Interquartile Range)**: Uses Q1, Q3, and IQR to detect outliers

### Proximity-Based Methods
- **KNN (K-Nearest Neighbors)**: Computes distances to nearest neighbors
- **LOF (Local Outlier Factor)**: Identifies density-based local outliers
- **DBSCAN**: Density-based clustering to find anomalies

### Machine Learning Methods
- **One-Class SVM**: Learns a boundary to separate normal data from outliers
- **Isolation Forest**: Isolates observations by randomly partitioning data
- **Autoencoders**: Neural networks to detect outliers through reconstruction error

### Ensemble Methods
- **Feature Bagging**: Aggregates multiple detection models
- **LODA**: Linear time anomaly detection using random projections

### Other Methods
- **Graph-Based Methods**
- **Clustering-Based Methods**: Identifies low-density regions as outliers (e.g., DBSCAN)
- **Density-Based Methods**: Finds outliers in low-density regions (e.g., Gaussian Density Estimation, Parzen Window Density)

## Distance Metrics
- **Euclidean Distance**: Straight-line distance between two points
- **Manhattan Distance**: Sum of absolute differences between coordinates
- **Mahalanobis Distance**: Measures distance considering correlations between variables
- **Cosine Distance**: Measures the cosine of the angle between two vectors
- **Hamming Distance**: Counts mismatches between data points
- **Simple Matching Coefficient (SMC)**: Proportion of matching categories
- **Jaccard Index**: Intersection size divided by union size of categories
- **Goodall's D**: More weight to rare category differences
- **Gower's Distance**: Normalizes contributions of different data types

## Libraries
- **scikit-learn**: General machine learning library
- **PyOD**: Outlier detection library
- **statsmodels**: Statistical models library
- **PyCaret**: Automated machine learning library

## Handling Mixed Data Types

### Steps
1. **Identify Data Types**
   - Numerical: Continuous or discrete numbers
   - Categorical: Nominal or ordinal categories
   - Text: Strings or sentences
   - Date/Time: Temporal data
 - 
2. **Handle Missing Values**
   - Numerical: Impute using mean, median, or regression
   - Categorical: Impute using mode or predictive models
   - Text: Impute with placeholder or use NLP techniques

3. **Preprocess Data / Feature Engineering**
   - **Numerical Data**: Standardize or normalize
   - **Categorical Data**: Encode using One-Hot or Label Encoding
   - **Text Data**: Convert to vectors (BoW, TF-IDF, embeddings)
   - **Date/Time Data**: Extract features (year, month, day)


### Algorithms and Methods

#### Numerical and Categorical Data
- **Decision Trees**: Handle mixed types without preprocessing
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Sequential ensemble of weak learners
- **CatBoost**: Gradient boosting algorithm that handles categorical features natively

#### Text Data
- **Bag-of-Words (BoW)**: Convert text to numerical vectors
- **TF-IDF**: Adjusts term frequency by inverse document frequency
- **Word Embeddings**: Use pre-trained embeddings (Word2Vec, GloVe)
- **BERT**: Contextual embeddings for deeper text understanding

#### Date/Time Data
- **Feature Extraction**: Extract parts (year, month, day) or aggregate (week, quarter)
- **Time Series Models**: ARIMA, Prophet, LSTM

## Handling Mixed Data in Models

#### Combining Different Data Types
- **ColumnTransformer (scikit-learn)**: Apply different transformations to different columns
- **FeatureUnion (scikit-learn)**: Combine results of multiple transformers

#### Encoding Methods
- **One-Hot Encoding**: For nominal categorical variables
- **Label Encoding**: For ordinal categorical variables
- **Ordinal Encoding**: For ordinal categorical variables preserving order

#### Handling Missing Data
- **SimpleImputer (scikit-learn)**: Basic imputation strategies
- **KNNImputer (scikit-learn)**: Imputation using k-nearest neighbors
- **IterativeImputer (scikit-learn)**: Multivariate imputation by chained equations

### Notes
- **Gower Distance**: Normalizes data types for mixed data
- **k-modes**: Handles mixed data types automatically

### References

#### Blog Posts
- [Benchmarking Outlier Detection Algorithms](https://longvu2.medium.com/benchmarking-outlier-detection-algorithms-5fed922a65cf)

#### Papers
- [Statistics-Based Outlier Detection and Correction Method for Amazon Customer Reviews](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8700267/)
```