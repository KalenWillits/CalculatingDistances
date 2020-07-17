# %% markdown
# ## Euclidean and Manhattan Distance Calculations
#
# While working on this quick case study, you'll see examples and comparisons of distance measures. Specifically, you'll visually compare the Euclidean distance to the Manhattan distance measures. Distance measures have a multitude of uses in data science and are the foundations of many algorithms you'll be using, including Prinical Components Analysis.
# %% codecell
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
%matplotlib inline
# !mkdir figures

plt.style.use('ggplot')
# %% codecell
# Load Course Numerical Dataset
cd_data = 'data/'
df = pd.read_csv(cd_data+'distance_dataset.csv',index_col=0)
df.head()
# %% markdown
# ### Euclidean Distance
#
# Let's visualize the difference between the Euclidean and Manhattan distance.
#
# Please use pandas to load the dataset .CSV file and Numpy to compute the __Euclidean distance__ to the point (Y=5, Z=5) that we've chosen as a reference. On the left, note the dataset projected onto the YZ plane and color coded per the Euclidean distance we just computed. As we are used to, points that lie at the same Euclidean distance define a regular 2D circle.
#
# Note that the __SciPy library__ comes with optimized functions written in C to compute distances (in the scipy.spatial.distance module) that are much faster than our (naive) implementation.
# %% codecell
# In the Y-Z plane, we compute the distance to ref point (5,5)
distEuclid = np.sqrt((df.Z - 5)**2 + (df.Y - 5)**2)
# %% markdown
# **<font color='teal'>Create a distance to reference point (3,3) matrix similar to the above example.</font>**
# %% codecell
distEuclid = np.sqrt((df.Z - 3)**2 + (df.Y - 3)**2)

# %% markdown
# **<font color='teal'>Replace the value set to 'c' in the plotting cell below with your own distance matrix and review the result to deepen your understanding of Euclidean distances. </font>**
# %% codecell
figEuclid = plt.figure(figsize=[10,8])

plt.scatter(df.Y - 5, df.Z-5, c=distEuclid, s=20)
plt.ylim([-4.9,4.9])
plt.xlim([-4.9,4.9])
plt.xlabel('Y - 5', size=14)
plt.ylabel('Z - 5', size=14)
plt.title('Euclidean Distance')
cb = plt.colorbar()
cb.set_label('Distance from (5,5)', size=14)
plt.savefig('figures/EuclideanDistance')

#figEuclid.savefig('Euclidean.png')
# %% markdown
# ### Manhattan Distance
#
# Manhattan distance is simply the sum of absolute differences between the points coordinates. This distance is also known as the taxicab or city block distance as it measures distances along the coorinate axis, which creates "paths" that look like a cab's route on a grid-style city map.
#
# We display the dataset projected on the XZ plane here color coded per the Manhattan distance to the (X=5, Z=5) reference point. We can see that points lying at the same distance define a circle that looks like a Euclidean square.
# %% codecell
# In the Y-Z plane, we compute the distance to ref point (5,5)
distManhattan = np.abs(df.X - 5) + np.abs(df.Z - 5)
# %% markdown
# **<font color='teal'>Create a Manhattan distance to reference point (4,4) matrix similar to the above example and replace the value for 'c' in the plotting cell to view the result.</font>**
# %% codecell
distManhattan = np.abs(df.X - 4) + np.abs(df.Z - 5)

# %% markdown
# Now let's create distributions of these distance metrics and compare them. We leverage the scipy dist function to create these matrices similar to how you manually created them earlier in the exercise.
# %% codecell
import scipy.spatial.distance as dist

mat = df[['X','Y','Z']].values
DistEuclid = dist.pdist(mat,'euclidean')
DistManhattan = dist.pdist(mat, 'cityblock')
largeMat = np.random.random((10000,100))
# %% markdown
# **<font color='teal'>Plot histograms of each distance matrix for comparison.</font>**
# %% codecell
plt.subplots(1,3, figsize=(20,20/3))
plt.subplot(111)
plt.hist(DistEuclid)
plt.subplot(121)
plt.hist(DistManhattan)
plt.subplot(131)
plt.hist(largeMat)
