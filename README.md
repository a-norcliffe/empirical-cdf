# Empirical CDF

This is a public implementation of a vectorized empirical cdf calculator using pytorch.

Please note this code is not maintained.

## Why use Empirical CDF?
Empirical CDF is another way to preprocess continuous features. It is different to normalizing to between 0 and 1, or mean 0 standard deviation 1 by scaling the data non-linearly.

In deep learning large features in neural networks can lead to training difficulty. Whilst the universal approximation theorem tells us a neural network **can** model any function, in reality large numbers in networks can create large gradients, even if these are clipped, the large feature values can still dominate the effect on the gradient.
So we want to make the numbers a more manageable size. The issue with shifting and scaling is that the shape of a feature's histogram remains the same. So if there are large gaps, then those remain, and any bins in the histogram that were close before scaling are far closer, where potentially those small differences might not be picked up.

Instead if we use empirical cdf, we non-linearly scale so that we essentially train on the quantiles. This means that points that are very close together can still be distinguished, and gaps are reduced, so that large values are also small.

I have found this can help training significantly, but with most machine learning this should be tested, and should be used with caution, it might not apply for your problem.

## Usage

See the below code for a simple example, where we fit the empirical cdf to train data
and then transform the train and new test data:

```
from empirical_cdf import VectorEmpiricalCDF
empirical_cdf = VectorEmpiricalCDF(num_bins=1000)

# X_train is shape (batchsize, num_features)
# X_test is shape (test_batchsize, num_features)

empirical_cdf.fit(X_train)
X_cdf_train = empirical_cdf.transform(X_train)
X_cdf_test = empirical_cdf.transform(X_test)
```

The transform method carries out the empirical cdf, and then an inverse normal cdf. This way the features are normally distributed. To carry out just the empirical cdf so that features are uniformly distributed the line is:

```
from empirical_cdf import VectorEmpiricalCDF
empirical_cdf = VectorEmpiricalCDF(num_bins=1000)

empirical_cdf.fit(X)
X_cdf = empirical_cdf.empirical_cdf(X)
```

## Initialization Arguments

The inialization arguments for the empirical cdf are:

- num_bins: The number of bins to use when estimating the empirical cdf
- size_normal: We add normally distributed noise before fitting, so that any features that only fit into a small number of possible values are slightly separated during fitting, to create the cdf.
- ratio_uniform: We also add more data points uniformly sampled between the minimum and maximium values of X. The larger this value (can be larger than 1.0) the more that the transformation looks like a simple scaling. The reason is we create fake data points so that in the limit of infinite fake points the data is uniform on the space, so the cdf is a simple scaling.
