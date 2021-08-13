#!/usr/bin/env python
# coding: utf-8

# 
# 
# # Project: Wine Datasets
# 
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# > The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine
# 
# ### What Questions Are We Trying To Answer?
# 
#    ######  How many samples of red wine are there?
#    ###### How many samples of white wine are there?
#    ###### How many columns are in each dataset?
#    ###### Which features have missing values?
#    ###### How many duplicate rows are in the white wine dataset?
#    ###### Are duplicate rows in these datasets significant/ need to be dropped?
#    ###### How many unique values of quality are in the red wine dataset?
#    ###### How many unique values of quality are in the white wine dataset?
#    ###### What is the mean density in the red wine dataset?
#    ###### Is a certain type of wine (red or white) associated with higher quality?
#    ###### What level of acidity (pH value) receives the highest average rating?
#    ###### Do wines with higher alcoholic content receive better ratings?
#    ###### Do sweeter wines (more residual sugar) receive better ratings
#    ###### What level of acidity receives the highest average rating?
# 

# <a id='wrangling'></a>
# ## Data Wrangling
# 
# >  Asessing Data
# 
# ### 2.1 Describe Data's General Properties

# In[2]:


import pandas  as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


url = 'https://raw.githubusercontent.com/patiegm/Udacity_Data_Analysis_Nanodegree/master/Wine%20Case%20Study/winequality-red.csv'
df_red = pd.read_csv(url, sep=';')


# In[4]:


url = 'https://raw.githubusercontent.com/patiegm/Udacity_Data_Analysis_Nanodegree/master/Wine%20Case%20Study/winequality-white.csv'
df_white = pd.read_csv(url, sep=';')


# In[5]:


df_red.shape


# In[6]:


df_white.shape


# In[7]:


df_red.info()


# In[8]:


df_white.info()


# In[9]:


df_red.nunique()


# In[10]:


df_white.nunique()


# In[11]:


df_red.describe()


# In[12]:


df_white.describe()


# In[20]:


df_red.hist(figsize=( 8,8));
plt.tight_layout()


# In[14]:


df_white.hist(figsize=( 8,8));
plt.tight_layout()


# ### 2.2 Verify Data Quality
# 
# >Is the data complete (does it cover all the cases required)?
#   Is it correct, or does it contain errors and, if there are errors, how common are they?
#   Are there missing values in the data? If so, how are they represented, where do they occur, and how common are they?
# 

# ### Missing Data

# In[21]:


def missing_values_table(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        return mis_val_table_ren_columns


# In[22]:


df_red.isnull().sum()


# In[17]:


df_white.isnull().sum()


# In[23]:


missing_values_table(df_red)


# In[24]:


missing_values_table(df_white)


# ### Duplicates

# In[25]:


sum(df_red.duplicated())


# In[26]:


sum(df_white.duplicated())


# >Decision We may want to remove duplicate rows entirely from the dataset. To do so we would run the following

# In[27]:


df_red.drop_duplicates(inplace=True)


# In[28]:


df_white.drop_duplicates(inplace = True)


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# >### Research Question 1 (How many samples of red wine are there?)

# In[29]:


df_red.shape


# >### Research Question 2 (How many samples of white wine are there?)

# In[30]:


df_white.shape


# >### Research Question 3 (How many columns are in each dataset?)
# 

# In[31]:


df_red.shape[1]


# In[32]:


df_white.shape[1]


# 
# >### Research Question 4 (How many unique values of quality are in the red wine and white dataset?)
# 

# In[33]:


df_red.nunique()


# In[34]:


df_white.nunique()


# 
# 
# >### Research Question 5 (What is the mean density in the red wine dataset?)
#     
# 
# 

# In[35]:


df_red.density.mean()


# >### Research Question 6 (Is a certain type of wine (red or white) associated with higher quality?)

# ###### let's combine two datasets

# In[36]:


df_red.columns


# In[37]:


df_white.columns


# In[38]:


(df_red.columns == df_white.columns).all()


# In[39]:


df_red.rename(columns = {'total sulfur dioxide' : 'total_sulfur_dioxide'},inplace = True)


# In[40]:


df_white.rename(columns = {'total sulfur dioxide' : 'total_sulfur_dioxide'},inplace = True)


# In[41]:


# create color array for red dataframe
color_red = np.repeat('red', df_red.shape[0])
# create color array for white dataframe
color_white = np.repeat('white', df_white.shape[0]) #Repeat of element (white , number of repeat)


# In[42]:


df_red['color'] = color_red
df_white['color'] = color_white


# In[43]:


wine_df = df_red.append(df_white)


# In[64]:


wine_df=wine_df.rename(columns = {'residual sugar':'residual_sugar'})


# In[66]:


wine_df.groupby('color').quality.mean()


# ###### The quality of white wine is better than red wine 

# >### Research Question 7 (What level of acidity (pH value) receives the highest average rating?)

# In[68]:


wine_df['pH'].describe()


# In[93]:


# Bin edges that will be used to "cut" the data into groups
bin_edges = [2.72, 3.11, 3.21, 3.32, 4.01]
# Labels for the four acidity level groups
bin_names = ['high', 'mod_high', 'medium', 'low']

# Creates acidity_levels column
wine_df['acidity_levels'] = pd.cut(wine_df['pH'], bin_edges, labels=bin_names)

# Checks for successful creation of this column
wine_df.head()# Labels for the four acidity level groups


# In[94]:


wine_df.groupby('acidity_levels').mean().quality


# >##### Do wines with higher alcoholic content receive better ratings?

# In[66]:


wine_df['alcohol'].plot(kind = 'hist' , figsize = (5,5));


# ###### the art shows me that the data is not symmetrical,so wel will use median in this case

# In[70]:


median = wine_df['alcohol'].median()
low  = wine_df.query('alcohol < {}'.format(median))
high = wine_df.query('alcohol >= {}'.format(median))
#low  = wine_df[wine_df['alcohol'] < median] -->dataset
#high = wine_df[wine_df['alcohol'] > median] -->dataset
mean_quality_low = low['quality'].mean()
mean_quality_high = high['quality'].mean()


# In[72]:


locations = [1, 2]
heights = [mean_quality_low, mean_quality_high]
labels = ['Low', 'High']
plt.bar(locations, heights, tick_label=labels)
plt.title('Average Quality Ratings by Alcohol Content')
plt.xlabel('Alcohol Content')
plt.ylabel('Average Quality Rating');


# 
# 
# > #### Do sweeter wines (more residual sugar) receive better ratings
# 
# 

# In[79]:


median = wine_df['residual_sugar'].median()
low    = wine_df[wine_df['residual_sugar'] < median]
high   = wine_df[wine_df['residual_sugar'] > median]

mean_quality_low = low['quality'].mean() ## -->return the quality from dataset which residual_sugar is low
mean_quality_high =high['quality'].mean()


# In[85]:


#Use query to select each group and get its mean quality
median = wine_df['residual_sugar'].median()
low = wine_df.query('residual_sugar < {}'.format(median))
high = wine_df.query('residual_sugar >= {}'.format(median))

mean_quality_low = low['quality'].mean()
mean_quality_high = high['quality'].mean()


# In[86]:



# Create a bar chart with proper labels
locations = [1, 2]
heights = [mean_quality_low, mean_quality_high]
labels = ['Low', 'High']
plt.bar(locations, heights, tick_label=labels)
plt.title('Average Quality Ratings by Residual Sugar')
plt.xlabel('Residual Sugar')
plt.ylabel('Average Quality Rating');


# In[100]:


acidity_level_quality_means = wine_df.groupby('acidity_levels').mean().quality
acidity_level_quality_means


# In[101]:


locations = [4, 1, 2, 3]  # reorder values above to go from low to high
heights = acidity_level_quality_means

# labels = ['Low', 'Medium', 'Moderately High', 'High']
labels = acidity_level_quality_means.index.str.replace('_', ' ').str.title() # alternative to commented out line above

plt.bar(locations, heights, tick_label=labels)
plt.title('Average Quality Ratings by Acidity Level')
plt.xlabel('Acidity Level')
plt.ylabel('Average Quality Rating');


# 
# 
