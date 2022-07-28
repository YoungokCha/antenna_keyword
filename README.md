# Antenna Keyword Prediction
## Installation

The main project requires `python3.7` and install the dependency package: 

  `pip install -r requirements.txt`
  
## 1. Retrieving Data

This program utilises a Python package named, pybliometrics, to access their data via the Scopus’ RESTful API using HTTP requests.  Following codes enable to obtain over 159K antenna related abstracts and relevant data (e.g., title, publication name, author’s keywords, number of citation and affiliation name).

```python
import pybliometrics
from pybliometrics.scopus import ScopusSearch
from pybliometrics.scopus.utils import config

config['Authentication']['InstToken'] = ''
print(config['Authentication']['InstToken']) # need an institution token
query = 'TITLE-ABS-KEY(antenna)'
s = ScopusSearch(query, download=True, verbose=True)  
```

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

This is the code for using a pre-trained word embedding model.

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


## 3. Analysing trend

We need to build a data set that tracks a frequency over time (i.e., time series) for each keyword. We count the occurrences of each keyword per year over the duration from 1981 to 2021. This data set also is applied by a weight which is considered a number of citaion, journal name and publication year.

```python
for keyword in keywords:
      re.findall(keyword,clean_abstract)
      result.append(len(re.findall(keyword,clean_abstract))*w)   # w = weight
```                

Following code can plot the heatmap graph for top 30 increasing, decreasing and emerging keywords based on their frequency counts.

```python
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

df = pandas.read_excel('keyword_frequency_weight_year.xlsx', sheet_name = 'heatmap_em', index_col= 0)
fig = plt.figure(figsize=(20,25))
sns.set(font_scale=3)
sns.heatmap(df,  cmap="YlOrRd")
plt.show()
```
<img width="400" alt="image" src="https://user-images.githubusercontent.com/48100788/181482269-0add5113-b29e-4959-be50-26f0c0735a30.png"><img width="385" alt="image" src="https://user-images.githubusercontent.com/48100788/181482379-f1f6a4e4-1611-439a-9b61-db73aa6eef92.png"><img width="400" alt="image" src="https://user-images.githubusercontent.com/48100788/181482437-954128ab-0d61-49b5-b4d6-5ccd6649ead2.png">


## 4. Prediction

The time-series data set is now fed into the Encoder-Decoder LSTM to predict the next series. we add the attention component to the network permits the decoder to utilise the most relevant parts of the input sequence in a flexible manner, by a weighted combination of all of the encoded input vectors.
```python
class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()

```


The architecture of LSTM is modified to have two LSTM units (e.g., encoder and decoder) and each unit is composed of 300 layers with rectified linear unit (ReLU). The validation result using average Mean Square Error (MSE).

```python
model_att = Sequential()
model_att.add(LSTM(300, activation='relu', input_shape=(look_back,1),return_sequences=True))
model_att.add(attention()) 
model_att.add(RepeatVector(1))
model_att.add(LSTM(300, activation='relu', return_sequences=True))
model_att.add(TimeDistributed(Dense(1)))
model_att.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model_att.summary()
model_att.fit(train_generator, epochs=num_epochs, verbose=1, shuffle=False)
prediction = model_att.predict(test_generator)

```
Following plot is illustrated the prediction results for next 4 years (from year of 2022 to 2026) both with attention layer and withuout attention layer.

![image](https://user-images.githubusercontent.com/48100788/181509462-1f35f180-2df0-47da-8158-066e712a35f4.png)






