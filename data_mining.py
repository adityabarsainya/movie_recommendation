import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import warnings;

warnings.simplefilter('ignore')

md = pd.read_csv('/home/aditya/Aditya/UnderGrad/Movies_python1/movies_metadata.csv')
md.head()


md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(
    lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()
C


m = vote_counts.quantile(0.95)
m

# In[6]:


md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(
    lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

# In[7]:


qualified = md[(md['vote_count'] >= m) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())][
    ['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')
qualified.shape


# In[8]:


def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)


# In[9]:


qualified['wr'] = qualified.apply(weighted_rating, axis=1)

# In[10]:


qualified = qualified.sort_values('wr', ascending=False).head(250)

# In[11]:


qualified.head(15)

# In[12]:


s = md.apply(lambda x: pd.Series(x['genres']), axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_md = md.drop('genres', axis=1).join(s)


# In[13]:


def build_chart(genre, percentile=0.85):
    df = gen_md[gen_md['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)

    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][
        ['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')

    qualified['wr'] = qualified.apply(
        lambda x: (x['vote_count'] / (x['vote_count'] + m) * x['vote_average']) + (m / (m + x['vote_count']) * C),
        axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)

    return qualified


# In[14]:


build_chart('Romance').head(15)

# In[15]:


links_small = pd.read_csv('/home/aditya/Aditya/UnderGrad/Movies_python1/links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')


# In[16]:


from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
# In[17]:


links_small = pd.read_csv('/home/aditya/Aditya/UnderGrad/Movies_python1/links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

# In[18]:


links_small = pd.read_csv('/home/aditya/Aditya/UnderGrad/Movies_python1/links_small.csv')

# In[19]:


links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

# In[20]:


md = md.drop([19730, 29503, 35587])

# In[21]:


md['id'] = md['id'].astype('int')

# In[22]:


smd = md[md['id'].isin(links_small)]
smd.shape

# In[23]:


smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna('')

# In[24]:


tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])

# In[25]:


tfidf_matrix.shape

# In[26]:


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# In[27]:


cosine_sim[0]

# In[28]:


smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])



def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


# In[29]:


smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])


# In[30]:


def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


# In[31]:


get_recommendations('The Godfather').head(10)


# In[32]:


get_recommendations('The Godfather').head(10)

# In[33]:


get_recommendations('The Dark Knight').head(10)

# In[34]:


credits = pd.read_csv('/home/aditya/Aditya/UnderGrad/Movies_python1/credits.csv')
keywords = pd.read_csv('/home/aditya/Aditya/UnderGrad/Movies_python1/keywords.csv')

# In[35]:


keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
md['id'] = md['id'].astype('int')

# In[36]:


md.shape

# In[37]:


md = md.merge(credits, on='id')
md = md.merge(keywords, on='id')

# In[38]:


smd = md[md['id'].isin(links_small)]
smd.shape

# In[39]:


smd['cast'] = smd['cast'].apply(literal_eval)
smd['crew'] = smd['crew'].apply(literal_eval)
smd['keywords'] = smd['keywords'].apply(literal_eval)
smd['cast_size'] = smd['cast'].apply(lambda x: len(x))
smd['crew_size'] = smd['crew'].apply(lambda x: len(x))


# In[40]:


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# In[41]:


smd['director'] = smd['crew'].apply(get_director)

# In[42]:


smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >= 3 else x)

# In[43]:


smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

# In[44]:


smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

# In[45]:


smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
smd['director'] = smd['director'].apply(lambda x: [x, x, x])

# In[46]:


s = smd.apply(lambda x: pd.Series(x['keywords']), axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'

# In[47]:


s = s.value_counts()
s[:5]

# In[48]:


s = s[s > 1]

# In[49]:


stemmer = SnowballStemmer('english')
stemmer.stem('dogs')


# In[50]:


def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words


# In[51]:


smd['keywords'] = smd['keywords'].apply(filter_keywords)
smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])


# In[52]:


smd['keywords'] = smd['keywords'].apply(filter_keywords)
smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

# In[53]:


smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))

# In[54]:


count = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
count_matrix = count.fit_transform(smd['soup'])

# In[55]:


cosine_sim = cosine_similarity(count_matrix, count_matrix)

# In[56]:


smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])

# In[57]:


get_recommendations('The Dark Knight').head(10)

# In[58]:


get_recommendations('Mean Girls').head(10)


# In[59]:


def improved_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]

    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    qualified = movies[
        (movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    return qualified


# In[60]:


improved_recommendations('The Dark Knight')

# In[61]:


improved_recommendations('Mean Girls')

# In[62]:


reader = Reader()

# In[63]:


ratings = pd.read_csv('/home/aditya/Aditya/UnderGrad/Movies_python1/ratings_small.csv')
ratings.head()

# In[64]:


data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
#data.split(n_folds=5)

# In[65]:


svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'],cv=5)

# In[66]:


trainset = data.build_full_trainset()
svd.fit(trainset)

# In[67]:


ratings[ratings['userId'] == 1]

# In[68]:


svd.predict(1, 302, 3)


# In[69]:


def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan


# In[70]:


id_map = pd.read_csv('/home/aditya/Aditya/UnderGrad/Movies_python1/links_small.csv')[['movieId', 'tmdbId']]
id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')
# id_map = id_map.set_index('tmdbId')


# In[71]:


id_map = pd.read_csv('/home/aditya/Aditya/UnderGrad/Movies_python1/links_small.csv')[['movieId', 'tmdbId']]
id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')
# id_map = id_map.set_index('tmdbId')


# In[72]:


indices_map = id_map.set_index('id')


# In[74]:


def hybrid(userId, title):
    idx = indices[title]
    tmdbId = id_map.loc[title]['id']
    # print(idx)
    movie_id = id_map.loc[title]['movieId']

    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]

    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False)
    return movies.head(10)


# In[75]:


hybrid(1, 'Avatar')

# In[76]:


hybrid(500, 'Avatar')
