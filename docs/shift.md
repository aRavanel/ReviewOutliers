# Distribution Shift Scoring Methods

Quantify deviations of a sample from the training distribution to detect and adapt to data changes impacting model performance.

## Distances

- **Kullback-Leibler (KL) Divergence**
  - **Library**: `scipy`
  - **Function**: `scipy.stats.entropy`
  - **Usage**: Measures divergence between two probability distributions.
  - ```python
    from scipy.stats import entropy
    kl_divergence = entropy(p, q)
    ```

- **Jensen-Shannon Divergence**
  - **Library**: `scipy`
  - **Function**: `scipy.spatial.distance.jensenshannon`
  - **Usage**: Symmetrized and smoothed version of KL Divergence.
  - ```python
    from scipy.spatial.distance import jensenshannon
    js_divergence = jensenshannon(p, q)
    ```

- **Wasserstein Distance (Earth Mover's Distance)**
  - **Library**: `scipy`
  - **Function**: `scipy.stats.wasserstein_distance`
  - **Usage**: Measures the minimum cost of transforming one distribution into another.
  - ```python
    from scipy.stats import wasserstein_distance
    distance = wasserstein_distance(u_values, v_values)
    ```

- **Maximum Mean Discrepancy (MMD)**
  - **Library**: `pyMMD`
  - **Function**: `mmd`
  - **Usage**: Non-parametric distance measure between two distributions.
  - ```python
    from pymmd import mmd
    mmd_value = mmd(X_train, X_test)
    ```

## Algorithms

List of algorithms:
- **Maximum Mean Discrepancy (MMD)**: Distance between means of two distributions.
- **Kolmogorov-Smirnov (KS) Test**: Compares cumulative distributions of two datasets.
- **Population Stability Index (PSI)**: Measures distribution shift of a variable.
- **Jensen-Shannon Divergence**: Measures similarity between two probability distributions.

### Details

- **Population Stability Index (PSI)**
  - **Library**: `pandas`, custom functions
  - ```python
    def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
        # custom function to calculate PSI
        pass
    ```

- **Chi-Square Test**
  - **Library**: `scipy`
  - **Function**: `scipy.stats.chisquare`
  - ```python
    from scipy.stats import chisquare
    chi2_stat, p_val = chisquare(f_obs, f_exp)
    ```

- **Covariate Shift Detection**
  - **Library**: `scikit-learn`
  - **Function**: Various classifiers (e.g., `LogisticRegression`)
  - ```python
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    ```

- **Adversarial Validation**
  - **Library**: `scikit-learn`
  - **Function**: Various classifiers (e.g., `GradientBoostingClassifier`)
  - ```python
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    ```

- **Kernel Density Estimation (KDE)**
  - **Library**: `scipy`, `sklearn`
  - **Function**: `scipy.stats.gaussian_kde`
  - ```python
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(dataset)
    density = kde.evaluate(values)
    ```

- **Histogram-Based Methods**
  - **Library**: `numpy`, `scipy`
  - **Function**: `numpy.histogram`, `scipy.stats.ks_2samp`
  - ```python
    import numpy as np
    from scipy.stats import ks_2samp
    hist_train, _ = np.histogram(data_train, bins=10)
    hist_test, _ = np.histogram(data_test, bins=10)
    ks_stat, p_val = ks_2samp(hist_train, hist_test)
    ```