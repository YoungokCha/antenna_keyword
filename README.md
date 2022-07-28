# AntennaKeywordPrediction
## Installation

The main project requires `python3.7` and install the dependency package: 

  `pip install -r requirements.txt`
  
## 1. Retrieving Data

```python
import pybliometrics
from pybliometrics.scopus import ScopusSearch
from pybliometrics.scopus.utils import config

config['Authentication']['InstToken'] = ''
print(config['Authentication']['InstToken']) # need an institution token
query = 'TITLE-ABS-KEY(antenna)'
s = ScopusSearch(query, download=True, verbose=True)  
```
This program utilises a Python package named, pybliometrics, to access their data via the Scopus’ RESTful API using HTTP requests.  We collected over 159K antenna related abstracts and relevant data (e.g., title, publication name, author’s keywords, number of citation and affiliation name).

Using the well-referenced Rapid Automatic Keyword Extraction (RAKE) algorithm, we can find many meaningful keyword phrases of each paper, however, all these phrases cannot be used for our prediction task due to its significant amount and complexity. 
```python
from nltk.corpus import stopwords  
from rake_nltk import Metric, Rake

stop_words = set(stopwords.words('english'))
df = pd.read_excel('antenna_data.xlsx', sheet_name='antenna_all', header =0) 
abstracts = df['description']
r = Rake(language="ENGLISH", stopwords=stop_words, ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO, min_length=2, max_length=2)
```
For this, the keywords of each abstract which are overlapped with the RAKE's keyword phrases and author’s keywords or the words in the title of the paper are selected. So I added the below codes:

```python
if (word in author_keyword):
                    keyword_candidate.append(word)

if(len(set(word).intersection(set(title_split)))>1):
                    keyword_candidate.append(word)
```
## 2. Embedding keywords

Using a pre-trained word embedding model, Mat2Vec, which is built from an unsupervised word embedding using 3.3 million scientific abstracts,

```python
model = Word2Vec.load("models/pretrained_embeddings")
```
`find_vector_value()` can the selected keywords are vectorised onto the hyper-dimensional space to analyse their relationship.

For clustering analysis, we plot all 200-dimensional 2,415 keywords into 2D using Principal Component Analysis (PCA). As a result, all the keywords are now clustered into 7 groups (or categories) using k-means algorithm. 

```python
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

twodim = PCA().fit_transform(word_vec)[:,:2]  #2D PCA
model = KMeans(n_clusters =7, random_state=42)  #k = 7
model.fit(twodim)
y_kmeans = model.fit_predict(twodim)
labels = model.labels_

plt.figure(figsize =(5,5))
sns.set_style("ticks")
colormap = np.array(['red', 'orange','aqua', 'purple', 'green', 'magenta', 'blue'])
plt.scatter(twodim[:,0], twodim[:,1], c=colormap[model.labels_], s=10)

plt.xlabel('PC 1', size =10)
plt.ylabel('PC 2', size =10)
```
![image](https://user-images.githubusercontent.com/48100788/181480621-827c33f4-298f-4db5-88c5-a9873bc5a0c7.png)

