#!/usr/bin/env python
# coding: utf-8

# # ETL Script for Electric Vehicle Data 

# In[1]:


# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# In[2]:


# Step 2: Extract Data
url = "https://data.wa.gov/api/views/f6w7-q2d2/rows.csv"  
df = pd.read_csv(url)
df.head()


# ## Step 3: Explore Data - Summary Statistics

# In[3]:


#Data size
df.shape


# In[4]:


# Columns and datatypes 
for column in df.columns:
    print(f"{column}: {df[column].dtype}")


# In[5]:


# Columns and overview 
df.describe(include='all')


# In[6]:


# Step 4: Explore Key Feature Distributions
features = [ 'Electric Range', 'Base MSRP']


for feature in features:
    df[feature].hist()
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()


# In[7]:


# Step 4: Explore 7 Key categorical Feature dispersion

#Function to turn back counts by subcategory under a column
def plot(df,c):
    plt.figure(figsize=(20, 6))
    ax = sns.countplot(x=c, 
                       data=df)
#                        order = df[c].value_counts().index)
    plt.title(f'Count by {c}')
    plt.xlabel(f'{c}')
    plt.ylabel('Count')
    
    for p in ax.patches:
        txt = str(p.get_height()) 
        txt_x = p.get_x() 
        txt_y = p.get_height()
        ax.text(txt_x,txt_y,txt) 
        plt.xticks(rotation=90)

    return plt.show()

features=[ 'Model Year', 'Make', 'Model', 'Electric Vehicle Type',
       'Clean Alternative Fuel Vehicle (CAFV) Eligibility',  'Electric Utility']

for feature in features:
    plot(df, feature)


# ## Step 5: Data Cleaning: missing data & data type converstion 

# In[8]:


# Show only columns that have missing values
summary = pd.DataFrame({
    'Missing Values': df.isnull().sum(),
    'Data Type': df.dtypes
})

summary = summary[summary['Missing Values'] > 0]

print(summary)

# Show only columns that have missing values
num_rows_with_missing = df.isnull().any(axis=1).sum()
print(f"Number of rows with missing values: {num_rows_with_missing}")


# In[9]:


# Fill missing 'Electric Range' and 'Base MSRP' using matching 'Model', 'Model Year', and 'Make' group 
cols_to_fill = ['Electric Range', 'Base MSRP']
for col in cols_to_fill:
    df[col] = df.groupby(['Model', 'Model Year', 'Make'])[col].transform(lambda x: x.fillna(x.dropna().iloc[0]) if not x.dropna().empty else x)

# fill missing values in all columns with object (i.e., string) data type using "Unknown" 
df.loc[:, df.select_dtypes(include='object').columns] = df.select_dtypes(include='object').fillna("Unknown")
num_rows_with_missing = df.isnull().any(axis=1).sum()
print(f"Number of rows with missing values: {num_rows_with_missing}")


# In[146]:


# Remaining missing values can be left as-is for now and flagged for business review. 
# Confirm with the business team whether there are specific rules or defaults to apply in the future.
# During the data loading process, consider implementing alerting rules to detect missing values 
# and loop back to the business team for resolution before final load into the database. 


# In[12]:


#Additional cleaning:  Convert specified columns to integer type to optimize databse storage 
df['Postal Code']= df['Postal Code'].astype('Int64')
df['Electric Range'] = df['Electric Range'].astype('Int64')
df['Base MSRP'] = df['Base MSRP'].astype('Int64')
df['Legislative District'] = df['Legislative District'].astype('Int64')


# In[13]:


df.head()


# ## Step 6: Data Transformation: encode categorical variable

# In[14]:


#Overiew of unique values by columns
summary = pd.DataFrame({
    'Data Type': df.dtypes,
    'Unique Values': df.nunique()
})

print(summary)


# In[15]:


# Encode the 'Electric Vehicle Type' and 'Clean Alternative Fuel Vehicle (CAFV) Eligibility' columns  
# as they contain long categorical strings with a limited number of unique values
# The approach can be generalized to other categorical columns needed 

encoders = {}
columns_to_encode = ['Clean Alternative Fuel Vehicle (CAFV) Eligibility', 'Electric Vehicle Type']

# 3. Apply encoding and store label mappings
dim_tables = {}

for col in columns_to_encode:
    le = LabelEncoder()
    df[col + '_Encoded'] = le.fit_transform(df[col].astype(str))
    
    # Store encoder and create dimension tables 
    encoders[col] = le
    dim_tables[col] = pd.DataFrame({
        'Encoded_Value': range(len(le.classes_)),
        col: le.classes_
    })

# 4. View dimension tables
for col, dim_df in dim_tables.items():
    print(f"\nDimension Table for '{col}':")
    print(dim_df)


# In[16]:


df.head()


# In[ ]:





# ## Step 7: Data Model Design and Preparation for Load (Star Schema)

# In[17]:


# Rename the columns to more concise and consistent snake_case format
df.rename(columns={
    'VIN (1-10)': 'vin',
    'County': 'county',
    'City': 'city',
    'State': 'state',
    'Postal Code': 'postal_code',
    'Model Year': 'model_year',
    'Make': 'make',
    'Model': 'model',
    'Electric Vehicle Type': 'ev_type',
    'Clean Alternative Fuel Vehicle (CAFV) Eligibility': 'cafv_eligibility',
    'Electric Range': 'electric_range',
    'Base MSRP': 'base_msrp',
    'Legislative District': 'legislative_district',
    'DOL Vehicle ID': 'dol_vehicle_id',
    'Vehicle Location': 'vehicle_location',
    'Electric Utility': 'electric_utility',
    '2020 Census Tract': 'census_tract',
    'Electric Vehicle Type_Encoded': 'ev_type_code',
    'Clean Alternative Fuel Vehicle (CAFV) Eligibility_Encoded': 'cafv_eligibility_code'
}, inplace=True)


# In[18]:


# Dimension Tables:

dim_EVtype = dim_tables['Electric Vehicle Type'].rename(columns={
    'Electric Vehicle Type': 'ev_type',
    'Electric Vehicle Type_Encoded': 'ev_type_code',
    }, inplace=True)

dim_CAFV=dim_tables['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].rename(columns={
    'Clean Alternative Fuel Vehicle (CAFV) Eligibility': 'cafv_eligibility',
    'Clean Alternative Fuel Vehicle (CAFV) Eligibility_Encoded': 'cafv_eligibility_code'
    }, inplace=True)

# Retain only Electric Vehicle Type_Encoded and Clean Alternative Fuel Vehicle (CAFV) Eligibility_Encoded 
# as foreign keys from the above dimension tables to dim_vehicle table 
# The dim_vehicle table can be further optimized by encoding the Model and Make columns. 
# However, encoding was not implemented in this case, as the current dim_vehicle table size remains manageable and we just bring VIN (1-10) to facts table  
dim_vehicle = df[['vin', 'model_year', 'make', 'model', 
                  'ev_type_code', 'cafv_eligibility_code', 
                  'electric_range', 'base_msrp']].drop_duplicates()
dim_vehicle


# In[19]:


# dim_location table
# Drop duplicates from selected columns and reset the index to create a unique ID
dim_location = df[['postal_code', 'city', 'county', 'state']].drop_duplicates().reset_index(drop=True)
dim_location.insert(0, 'location_id', dim_location.index + 1)

# Merge back to the original df to add location_id
df = df.merge(dim_location, on=['postal_code', 'city', 'county', 'state'], how='left')

dim_location


# In[21]:


# Encode 'Electric_Utility' and create a dimension table, refence electric_utility_encoded as foreign key in fact table 
le = LabelEncoder()
df['electric_utility_encoded'] = le.fit_transform(df['electric_utility'].astype(str))

# Create dimension table for Electric_Utility
dim_electric_utility = pd.DataFrame({
    'electric_utility_encoded': range(len(le.classes_)),
    'electric_utility': le.classes_
})
dim_electric_utility


# In[22]:


# Fact table: Retain only VIN  and location_id as foreign keys from the dimension tables, 
# and remove duplicate columns already represented in the dimension tables to avoid redundancy 

# Define columns already represented in dimension tables (excluding foreign keys)
redundant_columns = [
    'model_year', 'make', 'model', 'ev_type','ev_type',
    'cafv_eligibility','cafv_eligibility_code','electric_utility','ev_type_code',
    'electric_range', 'base_msrp', 'postal_code','city', 'county', 'state'
]

# Drop redundant columns and keep unique rows for the fact table
fact_ev = df.drop(columns=redundant_columns).drop_duplicates()

# Extract longitude and latitude from 'vehicle_location' column for storage optimiazation 
fact_ev[['longitude', 'latitude']] = fact_ev['vehicle_location'].str.extract(r'POINT \(([-\d\.]+) ([-\d\.]+)\)')
fact_ev[['longitude', 'latitude']] = fact_ev[['longitude', 'latitude']].astype(float)
fact_ev.drop(columns=['vehicle_location'], inplace=True)

#use index as primary Key in prepartion to database load 
fact_ev.reset_index(drop=True)
fact_ev.insert(0, 'event_id', fact_ev.index + 1)

# Preview result
fact_ev.head()


# ## Step 8:  Ready Load 

# In[188]:


# These dataframes can now be loaded to a DWH
# Steps: connect to database, create tables and define data types, relations, laod files 
# sample code: load data to Microsoft database

import pymssql
import keyring
import getpass
import pandas as pd

# --- Securely retrieve SQL Server credentials ---
server = 'your_server_url_or_ip'
user = 'your_username'
password = keyring.get_password('mssql', user) or getpass.getpass("Enter SQL Server password: ")
database = 'your_database_name'

# --- Establish database connection ---
conn = pymssql.connect(server=server, user=user, password=password, database=database)
cursor = conn.cursor()


# --- Function to load DataFrame into SQL Server ---
def load_to_sql(df, table_name):
    cols = ', '.join(df.columns)
    placeholders = ', '.join(['%s'] * len(df.columns))
    insert_query = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})"
    
    for _, row in df.iterrows():
        cursor.execute(insert_query, tuple(row))

    conn.commit()
    print(f"{table_name} loaded successfully.")
    
# Assumes tables already exist in the database. If not, you'll need to issue CREATE TABLE commands.

load_to_sql(dim_EVtype, 'dim_EVtype')
load_to_sql(dim_CAFV, 'dim_CAFV')
load_to_sql(dim_vehicle, 'dim_vehicle')
load_to_sql(dim_location, 'dim_location')
load_to_sql(dim_electric_utility, 'dim_electric_utility')
load_to_sql(fact_ev, 'fact_ev')

# --- Close the connection ---
cursor.close()
conn.close()  


# **End of Script**
