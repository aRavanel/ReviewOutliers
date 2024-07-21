# Outliers

Why:

- detecting incorrect data
- detecting fraud
- detecting extreme cases

Types of methods:

- unsupervised -> chosen here
- semisupervised
- supervised

## Outlier Detection Techniques

### Methods

#### Statistical Methods

- **Z-Score**: Measures the distance from the mean in standard deviations
- **Modified Z-Score**: Adjusted Z-Score using median and MAD
- **IQR (Interquartile Range)**: Uses Q1, Q3, and IQR to detect outliers

#### Proximity-Based Methods

- **KNN (K-Nearest Neighbors)**: Computes distances to nearest neighbors
- **LOF (Local Outlier Factor)**: Identifies density-based local outliers
- **DBSCAN**: Density-based clustering to find anomalies

#### Machine Learning Methods

- **One-Class SVM**: Learns a boundary to separate normal data from outliers
- **Isolation Forest**: Isolates observations by randomly partitioning data
- **Autoencoders**: Neural networks to detect outliers through reconstruction error

#### Ensemble Methods

- **Feature Bagging**: Aggregates multiple detection models
- **Loda**: Linear time anomaly detection using random projections

Others :

- Graph-Based Methods,
- Clustering-Based Methods: Identifies low-density regions as outliers. Ex: DBSCAN,
- Density-Based Methods : Finds outliers in low-density regions. Ex : Gaussian Density Estimation, Parzen Window Density

### Distance Metrics

- **Euclidean Distance**: Straight-line distance between two points
- **Manhattan Distance**: Sum of absolute differences between coordinates
- **Mahalanobis Distance**: Measures distance considering correlations between variables
- **Cosine Distance**: Measures the cosine of the angle between two vectors
- **Hamming Distance**: Counts mismatches between data points.
- **Simple Matching Coefficient (SMC)**: Proportion of matching categories.
- **Jaccard Index**: Intersection size divided by union size of categories.
- **Goodall's D**: More weight to rare category differences.
- **Gower's Distance**: Normalizes contributions of different data types.
- **One-Hot Encoding with Euclidean Distance**: Converts categories to binary and measures Euclidean distance.
- **Frequency-Based Methods**: Uses category frequency, emphasizing rare categories.

### libraries examples

- **scikit-learn**: Popular ML library
- **PyOD**: Comprehensive toolbox for outlier detection
- **statsmodels**: Statistical models and tests
- **SciPy**: Scientific computing library

## Handling Mixed Data Types

### Steps

1. **Identify Data Types**

   - Numerical: Continuous or discrete numbers
   - Categorical: Nominal or ordinal categories
   - Text: Strings or sentences
   - Date/Time: Temporal data

2. **Preprocess Data**

   - **Numerical Data**: Standardize or normalize
   - **Categorical Data**: Encode using One-Hot or Label Encoding
   - **Text Data**: Convert to vectors (BOW, TF-IDF, word embeddings, sentence embeddings, ...)
   - **Date/Time Data**: Extract relevant features (year, month, day, etc.)

3. **Handle Missing Values**

   - Numerical: Impute using mean, median, or regression
   - Categorical: Impute using mode or predictive models
   - Text: Impute with placeholder or use NLP techniques

4. **Feature Engineering**

   - Combine or create new features to better represent the data
   - Use domain knowledge to enhance feature set

5. **Scale and Transform Data**
   - Ensure numerical features are on a similar scale
   - Apply transformations if necessary (log, sqrt, etc.)

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

- convert categorical data to numerical

Either:

- use one-hot encoding for categorical data
- use gower distance that will normalize data types

- Notes :
- k-modes tackle mixed data types automatically

### reference papers :

- Statistics-Based Outlier Detection and Correction Method for Amazon Customer Reviews (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8700267/)
- -
