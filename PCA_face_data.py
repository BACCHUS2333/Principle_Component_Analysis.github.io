## Problem2
#Principle Analysis Component  
#k-means clustering

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
#import the dataset
raw_data = pd.DataFrame({'var1':[1,1,0,5,6,4],'var2':[4,3,4,1,2,0]})
print(raw_data)
print(raw_data.mean(axis=0))
plt.scatter(raw_data['var1'],raw_data['var2'])
plt.title('Scatter plot of the raw data (red x is the centroid)')
plt.xlabel('var1')
plt.ylabel('var2')
#plot centroid
plt.scatter(raw_data.mean(axis=0)[0],raw_data.mean(axis=0)[1],color='red',marker='x')
plt.show()

random_labeled_data = raw_data.copy()
random_labeled_data['cluster'] = np.random.choice([1,0],size=raw_data.shape[0])
print(random_labeled_data)
centroid_0 = random_labeled_data[random_labeled_data['cluster']==0].mean(axis=0)
centroid_1 = random_labeled_data[random_labeled_data['cluster']==1].mean(axis=0)
print("centroid of cluster 0:",centroid_0)
print("centroid of cluster 1:",centroid_1)
plt.scatter(random_labeled_data[random_labeled_data['cluster']==0]['var1'],random_labeled_data[random_labeled_data['cluster']==0]['var2'],c='pink')
plt.scatter(random_labeled_data[random_labeled_data['cluster']==1]['var1'],random_labeled_data[random_labeled_data['cluster']==1]['var2'],c='lightblue')
#plt.legend(['cluster 0','cluster 1'])
plt.title('Scatter plot of the random labeled data')
plt.xlabel('var1')
plt.ylabel('var2')
plt.scatter(random_labeled_data[random_labeled_data['cluster']==0].mean(axis=0)[0],random_labeled_data[random_labeled_data['cluster']==0].mean(axis=0)[1],color='red',marker='x')
plt.scatter(random_labeled_data[random_labeled_data['cluster']==1].mean(axis=0)[0],random_labeled_data[random_labeled_data['cluster']==1].mean(axis=0)[1],color='blue',marker='x')
plt.legend(['cluster 0','cluster 1','cluster 0 centroid','cluster 1 centroid'])
#the legend of centroid
plt.show()
#calculate the distance between each point and the centroid
distance_centorid_0 = np.sqrt((random_labeled_data['var1']-centroid_0['var1'])**2+(random_labeled_data['var2']-centroid_0['var2'])**2)
distance_centorid_1 = np.sqrt((random_labeled_data['var1']-centroid_1['var1'])**2+(random_labeled_data['var2']-centroid_1['var2'])**2)
random_labeled_data['distance_centorid_0'] = distance_centorid_0
random_labeled_data['distance_centorid_1'] = distance_centorid_1
print(random_labeled_data)

raw_data
data =raw_data.drop('cluster',axis=1)
k =3
centroid_df = pd.DataFrame(columns=['centroid_{i}'.format(i=i) for i in range(k)])

data['cluster'] = np.random.choice(range(k),size=data.shape[0])
            #calculate the centroid
for i in range(k):
    centroid_df['centroid_{i}'.format(i=i)] = data[data['cluster']==i].mean(axis=0)
            #calculate the distance between each point and the centroid
for i in range(k):
    data['distance_centorid_{i}'.format(i=i)] = np.sqrt((data['var1']-centroid_df['centroid_{i}'.format(i=i)]['var1'])**2+(data['var2']-centroid_df['centroid_{i}'.format(i=i)]['var2'])**2)
print(data)
np.argmin(data[['distance_centorid_{i}'.format(i=i) for i in range(k)]].values,axis=1)
import pandas as pd
import numpy as np

class KMeansClustering:
    def __init__(self):
        pass
    
    def k_means_clustering(self, input_data, k):
        recluster_count = 0
        centroid_df = pd.DataFrame(columns=['centroid_{i}'.format(i=i) for i in range(k)])

        data = input_data.copy()
        while True:
            if recluster_count == 0:
                print('recluster_count:', recluster_count)
                # randomly assign the cluster
                data['cluster'] = np.random.choice(range(k), size=data.shape[0])
                recluster_count += 1
                # calculate the centroid
                for i in range(k):
                    centroid_df['centroid_{i}'.format(i=i)] = data[data['cluster']==i].mean(axis=0)
                # calculate the distance between each point and the centroid
                for i in range(k):
                    data['distance_centorid_{i}'.format(i=i)] = np.sqrt((data['var1']-centroid_df['centroid_{i}'.format(i=i)]['var1'])**2+(data['var2']-centroid_df['centroid_{i}'.format(i=i)]['var2'])**2)

            else:
                # recluster based on the distance
                data['previous_cluster'] = data['cluster']
                data['cluster'] = np.argmin(data[['distance_centorid_{i}'.format(i=i) for i in range(k)]].values,axis=1)
                recluster_count += 1
                print(data)
                print('recluster_count:', recluster_count)
                # calculate the centroid
                for i in range(k):
                    centroid_df['centroid_{i}'.format(i=i)] = data[data['cluster']==i].mean(axis=0)
                # calculate the distance between each point and the centroid
                for i in range(k):
                    data['distance_centorid_{i}'.format(i=i)] = np.sqrt((data['var1']-centroid_df['centroid_{i}'.format(i=i)]['var1'])**2+(data['var2']-centroid_df['centroid_{i}'.format(i=i)]['var2'])**2)
                # check if the cluster is the same as the previous one
                if (data['cluster'] == data['previous_cluster']).all():
                    print('already stable')
                    break
                else:
                    data['previous_cluster'] = data['cluster']
                    print("recluster again, not stable yet")
        return data, centroid_df

# Example usage:
# Initialize the class
kmeans = KMeansClustering()
# Call the method

final_clustered_data,final_centroid_df = kmeans.k_means_clustering(raw_data.drop('cluster',axis=1),2)

print(final_clustered_data)
plt.scatter(final_clustered_data[final_clustered_data['cluster']==0]['var1'],final_clustered_data[final_clustered_data['cluster']==0]['var2'],c='pink')
plt.scatter(final_clustered_data[final_clustered_data['cluster']==1]['var1'],final_clustered_data[final_clustered_data['cluster']==1]['var2'],c='lightblue')
plt.title('Scatter plot of the final clustered data')
plt.xlabel('var1')
plt.ylabel('var2')
plt.scatter(centroid_df['centroid_0']['var1'],centroid_df['centroid_0']['var2'],color='red',marker='x')
plt.scatter(centroid_df['centroid_1']['var1'],centroid_df['centroid_1']['var2'],color='blue',marker='x')
plt.legend(['cluster 0','cluster 1','cluster 0 centroid','cluster 1 centroid'])
## Problem3
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.datasets import fetch_olivetti_faces

# Download the Olivetti Faces dataset using HTTP mirror
faces_data = fetch_olivetti_faces(data_home="http://mldata.org/repository/data/download/scikit-learn/mldata/olivetti_faces/")

def principle_analysis_component(input_data, n):
    if n > input_data.shape[1]:
        print('n should be smaller than the number of columns of the input data')
        return
    else:
        input_data = input_data - input_data.mean(axis=0)
        pca = PCA(n_components=n)
        principalComponents = pca.fit_transform(input_data)
        principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component {i}'.format(i=i) for i in range(n)])
        print(pca.explained_variance_ratio_,sum(pca.explained_variance_ratio_))
        linear_combination = pd.DataFrame(np.dot(principalDf.values,pca.components_),columns=input_data.columns)
        print(linear_combination.head())
        return pca.explained_variance_ratio_
    

fof_data = pd.DataFrame(faces_data.data,columns=['pixel_{i}'.format(i=i) for i in range(faces_data.data.shape[1])])
explained_variance_ratio_df = pd.DataFrame(index=['explained_variance_ratio'])
for i in [i for i in range(1,min(fof_data.shape)+1) if i%50 == 1 or i ==min(fof_data.shape)]:
    print(principle_analysis_component(fof_data,i))
    explained_variance_ratio_df['n={i}'.format(i=i)] = sum(principle_analysis_component(fof_data,i))
    print('------------------------------------')
print(explained_variance_ratio_df.T)
plt.bar(explained_variance_ratio_df.T.index,explained_variance_ratio_df.T['explained_variance_ratio'])
plt.title('Explained variance ratio of different n')
plt.xlabel('component number')
#rotate the x-axis label
plt.xticks(rotation=90)
plt.ylabel('explained variance ratio')
plt.show()
