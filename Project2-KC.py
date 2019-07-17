#!/usr/bin/env python
# coding: utf-8

# 
# # Project: Regional Growth with Gapminder World
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
# <br>
# 1) Which regions had higher rates of urban growth in 2017?<br>
# 2) Which indicator has a stronger relationship with the Urban Growth(UG)indicator? Is the relationship between urban growth and population groth stable?<br>
# 3) How have the indicators behaved in the last 10 years?<br>
# <br>
# The main objectives are find trends between the selected metrics and discover the regions that stood out in UG in 2017.<br>
# Dataframes from Gapminder World, with update from the original source:<br>
# <br>
# Urban population growth (annual %)<br>
# Primer source: World Bank<br>
# Category: Population<br>	
# Subcategory: Urbanization<br>
# <br>
# HDI (Human Development Index):<br>
# Primer source:	UNDP(United Nations Development Programme)+ Update(:2017)<br>
# Category: Society<br>	 
# HDI is an index used to rank countries by level of "human development".<br>
# It contains three dimensions: health level, educational level and living standard.(http://wikiprogress.org/articles/initiatives/human-development-index/)<br>
# <br>
# Inequality index (Gini)<br>
# Primer source: The World Bank<br>	
# Category: Economy<br>
# Subcategory: Poverty & inequality<br>
# "In economics, the Gini coefficient (/ˈdʒiːni/ JEE-nee), sometimes called Gini index, or Gini ratio, is a measure of statistical dispersion intended to represent the income or wealth distribution of a nation's residents, and is the most commonly used measurement of inequality." (https://en.wikipedia.org/wiki/Gini_coefficient_)<br>
# <br>
# Population growth (annual %)<br>	
# Primer source: The World Bank<br>
# Category: Population<br>	
# Subcategory: Population growth<br>
# The population growth's indicator is used to check it behaviour along its rates, in comparison with the Urban growth's indicator.<br>
# 

# In[263]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# ### General Properties

# In this first section, I import the dataframes,and make a first analysis of them checking the years covered and the null cells.<br>
# Since the HDI's dataset is outdated, I update its with data from a dataset updated.<br>
# The datasets with fews or some null cells are filled with their means.<br>
# The data with many null cells (more than 80%), are deleted.<br>

# In[264]:


#Importing the dataframes
urb_growth_orig = pd.read_csv('urban_population_growth_annual_percent.csv')
gini_orig  = pd.read_csv('gini.csv')
pop_growth_orig  = pd.read_csv('population_growth_annual_percent.csv')
hdi = pd.read_csv('hdi_human_development_index.csv')
hdi_update = pd.read_csv('HDI-Copy of 2018_all_indicators.csv')


# In[265]:


#Verifying the independent variable (urban_population_growth_annual_percent)
#urb_growth_orig.head()    ###1960-2017
urb_growth_orig.info()  ### some null cells


# UG:1960-2017, some null cells

# In[266]:


#Checking the dataset of hdi
#hdi.head()
hdi.info()


# In[267]:


#Checking the dataset of update for hdi
#hdi_update.head(60)
#print(hdi_update)
hdi_update.info()


# HDI: 1990-2015, many null cells<br>
# HDI update: 1990-2017 & 9999, many null cells<br>
# I found differences and repetitions between hdi and its update, hdi has many null cells.<br> 
# In these cases, I keep with the hdi's dataset.<br>

# In[268]:


#Checking the dataset of Gini
gini_orig.head()   ###1800-2040
#gini_orig.shape


# In[269]:


#checking null cells
gini_orig.info()   ###no null cells (it seems)


# In[270]:


#gini_orig.apply(lambda x: x.count(), axis=1)
gini_orig.isnull().sum(axis=1)


# Gini: 1800-2040, no null cells

# In[271]:


#Checking the dataset of pop Growth
#pop_growth_orig.head() ###1960-2017
pop_growth_orig.info()  ### some null cells


# pop_growth_orig: 1960-2017, some null cells

# ### Data Cleaning (Replace this with more specific notes!)

# In[272]:


# After discussing the structure of the data and any problems that need to be
#   cleaned, perform those cleaning steps in the second part of this section.


# In[273]:


#Cleaning/filling the independent variable (urban_growth)
#Filling the missing values with the mean and checking
urb_growth_orig.fillna(urb_growth_orig.mean(), inplace=True)
urb_growth_orig.info()   


# checking: no null cells

# In[275]:


#Cleaning hdi_update (filtering and merging dataframes)

#Renaming the main column to be the same of the main dataset hdi.
hdi_update = hdi_update.rename(columns={'country_name': 'country'})
hdi_update.head()


# In[274]:


#Selecting data of updates to the indicator hdi and checking
#Filter Pandas Dataframe By Values of Column
hdi_update.drop(hdi_update[hdi_update['indicator_id'] != 137506].index, inplace=True)
hdi_update
#hdi_update.info()


# In[276]:


#Removing unnecessary columns and checking,  keeping with the 10 last years
hdi_update.drop(hdi_update.iloc[:, 0:4], axis=1, inplace=True) #first 4
hdi_update.head()


# In[277]:


#Removing column '9999' (the last)
hdi_update.drop(hdi_update.iloc[:, -1:], axis=1, inplace=True)
hdi_update.head()


# In[279]:


#Choosing just the countries and the years that are missing in hdi
hdi_update.drop(hdi_update.iloc[:, 1:-2], axis=1, inplace=True) #years before 2016
hdi_update.head()


# In[278]:


# merging hdi and hdi_updates and assigning the name hdi_new
hdi_new = pd.merge(hdi,hdi_update, on ='country')
print(hdi_new)


# In[299]:


# Droping columns with + or = to 20% of values missing
hdi_new.dropna(thresh=0.8*len(hdi_new), axis=1, inplace=True)
hdi_new.head()


# In[282]:


#Filling the missing values with the mean and checking
hdi_new.fillna(hdi_new.mean(), inplace=True)
hdi_new.info()
#hdi_new 


# In[281]:


#Checking after cleaning
#hdi_new.head()   ###1990-2017
hdi_new.info()   ### no null cells


# hdi_new: 1990-2017, no null cells

# In[283]:


#Cleaning pop growth
#Filling the missing values with the mean and checking
pop_growth_orig.fillna(pop_growth_orig.mean(), inplace=True)
pop_growth_orig.info()
#pop_growth_orig


# <a id='eda'></a>
# ## Exploratory Data Analysis

# ###  1) Analysis of Urban Growth comparing countries
# Parameters: Bar grath, 50 samples of countries, Urban Growth, 2017.<br>
# With the bar grath I could have an overview of this indicator in some countries, in 2017.<br>
# 

# In[284]:


# Drawing a bar grath from 50 sample of countries, considering the indeoendent variable (urban growth)
# To get 50 random rows sorting 
urb_growth_orig_sample = urb_growth_orig.sample(n = 50).sort_values(by='country')
urb_growth_orig_sample


# In[285]:


#printing a bar graph
urb_growth_orig_sample.plot(kind='bar',x='country',y='2017', figsize=(20,6), title='urban growth in 2017: top 50 countries')


# ### Findings: 
# With the bar grath I could have an overview of this indicator in some countries, in 2017.<br>
# Analysing the countries with UG higher than 3.5 twice, I found that most of the countries with higher UG are in Africa, mainly in East Africa. <br>
# Run 1:<br>
# Bahrain: in the Persian Gul<br>
# Congo: Central Africa<br>
# Mozambique: Southern African<br>
# Niger:West Africa<br>
# Somalia: East Africa<br>
# Tanzania: East Africa<br>
# Uganda: East Africa<br>
# Run 2:<br>
# Cameroon: Central Africa<br>
# Etiophia:East Africa<br>
# Gambia: West Africa<br>
# Mauritania: Northwest Africa<br>
# Senegal: West Africa <br>
# South Sudan: North Africa <br>
# Tanzania: East Africa<br>
# Uganda: East Africa<br>
# Zambia:  East Africa<br>
# (Source: Wikipedia)<br>

# ###  2) Comparison of two indicators

# Parameters: 3 Scatters: Urban growth vs hdi,Urban growth vs Gini, and Urban growth vs Population growth, 2017. <br>

# In[286]:


#Comparing the relationship between variables (independent vs dependent)

#urban growth vs hdi
#Create a table to scatter ,
#Filtering and renaming to urb_growth(2017)
urb_growth_orig.drop(urb_growth_orig.iloc[:, 1:-1], axis=1, inplace=True)
urb_growth_orig=urb_growth_orig.rename(columns={'2017': 'urb_growth(2017)'})
urb_growth_orig.head()
#urb_growth_orig.shape


# In[287]:


#Filtering and renaming to hdi(2017)
hdi_new.drop(hdi_new.iloc[:, 1:-1], axis=1, inplace=True)


# In[288]:


hdi_new=hdi_new.rename(columns={'2017': 'hdi(2017)'})
hdi_new


# In[289]:


#merging dfs originals urban and hdi
ug_hdi = pd.merge(urb_growth_orig,hdi_new, on ='country')
print(ug_hdi)


# In[290]:


#Graph urban growth vs hdi

x = ug_hdi['urb_growth(2017)']
y = ug_hdi['hdi(2017)']

plt.scatter(x,y, color='y')
plt.xlabel('urb_growth')
plt.ylabel('hdi)')
plt.title('urb_growth vs hdi')
plt.show()


# In[291]:


# UG vs gini
gini_orig.drop(gini_orig.iloc[:,1:-24], axis=1, inplace=True)
gini_orig.drop(gini_orig.iloc[:, -23:], axis=1, inplace=True)


# In[292]:


gini_orig=gini_orig.rename(columns={'2017': 'gini(2017)'})
#gini_orig


# In[293]:


#merging dfs originals urban and gini
ug_gini = pd.merge(urb_growth_orig,gini_orig, on ='country')
print(ug_gini)


# In[294]:


#Graph urban growth vs gini

x = ug_gini['urb_growth(2017)']
y = ug_gini['gini(2017)']

plt.scatter(x,y, color='y')
plt.xlabel('urb_growth')
plt.ylabel('gini')
plt.title('urb_growth vs gini')
plt.show()


# In[295]:


# UG vs PG
pop_growth_orig.drop(pop_growth_orig.iloc[:,1:-1], axis=1, inplace=True)
pop_growth_orig=pop_growth_orig.rename(columns={'2017': 'pop_growth(2017)'})
pop_growth_orig.head()


# In[296]:


#merging dfs originals urban and pop growth
ug_pop = pd.merge(urb_growth_orig,pop_growth_orig, on ='country')
ug_pop.head()


# In[298]:


#Graph urban growth vs pop growth

x = ug_pop['urb_growth(2017)']
y = ug_pop['pop_growth(2017)']

plt.scatter(x,y, color='y')
plt.xlabel('urb_growth')
plt.ylabel('gini')
plt.title('urb_growth vs pop growth')
plt.show()


# urb_growth and pop_growth are checked just if there are many outliers, and their behavious along their rates.<br>
# Since they are bases on population growth, they cannot be compared.<br>

# ### Findings: 
# Urban growth vs HDI- moderate, linear and negative relationship (inversely proportional).<br>
# Urban growth vs Gini- weak, linear and positive relationship (directly proportional).<br>
# Urban growth vs pop growth- strong, linear and positive relationship (directly proportional).<br> 
# Therefore:<br>
#     There is some relationship between urban growth(UG) and HDI. The higher the UG, the lower the HDI.<br>
#     The strong relationship between UG and population growth(PG) was comproved. This was expected since the UG is based on the PG.<br>
# However:<br>
# Urban growth vs pop growth, the higher the indexes, the weaker is the relationship between them; on the other hand, the rarer are the occurrences of outliers.<br>    

# ### 3) Comparison the evolution of four indicators 
# (Evolution of metrics)
# 

# Parameters: a line graph with the 4 indicators, 2007-2017.<br>

# In[297]:


#Lines grath from the last 10 years # parameters manually selected
 
x = [2008,2009,2010,2011,2012,2013,2014,2015,2016,2017] 
# line 1 points
y1 = [5.466,5.584,5.716,5.956,6.092,6.048,5.932,5.776,5.626,5.474] 
# line 2 points  
y2 = [0.4646,0.4784,0.4880,0.4924,0.4970,0.5022,0.5052,0.5074,0.5178,0.5208]
y3 = [39.033333,38.766667,38.483333,38.200000,37.983333,37.833333,37.750000,37.783333,37.816667,37.866667]
y4 = [3.315000,3.433330,3.556667,3.660000,3.713333,3.705000,3.635000,3.535000,3.438333,3.346667]
plt.plot(x, y1, label = "urb_growth") 
plt.plot(x, y2, label = "hdi") 
plt.plot(x, y3, label = "gini") 
plt.plot(x, y4, label = "pop_growth") 

# naming the x axis 
plt.xlabel('years') 
# naming the y axis 
plt.ylabel('indicators') 
# giving a title to my graph 
plt.title('Evolution of the indicators (2007-2017)') 

# show a legend on the plot 
plt.legend() 
  
# function to show the plot 
plt.show() 


# ### Findings:
# In the last 10 years, the evolution of the UG was ver similar to the PG, as expected, and also similar with the HDI.<br>
# The Gini has a less stable than other indicators, tending to decline.<br>

# <a id='conclusions'></a>
# ## Conclusions
# 
# The higher urban growth (UG) is in the African continent, manly in East Africa.<br>
# HDI is a variable inversely proportional to the UG, but with a weak relationship. <br> Considering the comparison between two indicators (2017) and their evolution (2017-2017), HDI is still better than the Gini.<br>
# The scatter graphs (to 2017) and the linear graph (to 2017) show the expected similarity in the evolution of UG and PG.<br>
# Even the indicators Urban growth and Pop growth are based on population growth, it is shown that the higher their indexes, the weaker is the relationship between them.<br>
# This study uses small samples to make a first, simplified analysis. As well as, it has limitations by lack of statistical tests and do not imply any statistical conclusions.<br> 
# 
