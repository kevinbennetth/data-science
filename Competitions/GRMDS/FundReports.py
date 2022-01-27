# Importing libraries

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sns.set()

# Datasets

funds = pd.read_csv('datasets/funds.csv')
prices = pd.read_csv('datasets/prices_per_day.csv', index_col=[0], parse_dates=True)

#################################################################

st.title('Responsible Investing Dashboard')
st.caption('by Team CrossEntropy')

# Sidebar
st.sidebar.subheader('Fund Selection')

category = st.sidebar.selectbox(
    'Morningstar Category',
    funds['Morningstar Category'].unique()
)

fund = st.sidebar.selectbox(
    'Fund Name',
    funds['Name'][funds['Morningstar Category']==category]
)

# Select Data
ticker = funds['Ticker'][funds['Name']==fund].values[0]
selectedfund = funds[funds['Name']==fund]

st.header('Fund Summary')
# Ratings, Yield & Credit Quality
filter1 = ['Morningstar Sustainability Rating','Portfolio Sustainability Score','Yield (%)','Average Credit Quality']
selections1 = selectedfund[filter1].values.flatten()
rating, score, yields, quality = st.columns(4)

rating.metric('Sustainability Rating',selections1[0],None)
score.metric('Sustainability Score',selections1[1],None)
yields.metric('Yield',str(selections1[2])+' %',None)
quality.metric(filter1[3],selections1[3], None)

# Fund Size & Market Cap
filter2 = ['Fund Size (Mil)','Average Market Cap (Mil)']
selections2 = selectedfund[filter2].values.flatten()
fundsize, marketcap = st.columns(2)

fundsize.metric('Fund Size','$' + str(selections2[0]) +'M',None)
marketcap.metric('Average Market Cap','$' + str(selections2[1]) +'M',None)

# % Returns
timeframes = ['YTD Return (%)','1 Year Annualized (%)','3 Years Annualized (%)','5 Years Annualized (%)','10 Years Annualized (%)']
returns = selectedfund[timeframes].values.flatten()
ytd, y1, y3, y5, y10 = st.columns(5)
colnames = ['YTD','1Y','3Y','5Y','10Y']
cols = [ytd, y1, y3, y5, y10]

for i in range(5):
    if(i==0):
        cols[i].metric("Percentage Returns",colnames[i],str(returns[i])+' %')
    else:
        cols[i].metric("",colnames[i],str(returns[i])+' %')

# Stock Price Trend
st.header('Stock Price Trend')
selectedstock = prices[prices['stock_name']==ticker]
fig1, ax1 = plt.subplots(figsize=(12,6))
ax1.plot(selectedstock['stock_price'])
ax1.set_xlabel('Time')
ax1.set_ylabel('Price')
ax1.set_title(fund + ' ({})'.format(ticker))
st.pyplot(fig1)

fundalloc, portfolio = st.columns(2)

# Fund Allocations
fig2, ax2 = plt.subplots(figsize=(8,6))
allocs = ['% Alcohol','% Fossil Fuels','% Small Arms','% Thermal Coal','% Tobacco']
sns.barplot(x=[1,2,3,4,5],y=selectedfund[allocs].values.flatten(), ax=ax2)
ax2.set_xticklabels(allocs)
ax2.set_xlabel('% Allocation Fund')
ax2.set_ylabel('Percentage')
ax2.set_title(fund + ' Allocation Percentages')

fundalloc.header('Fund Allocation')
fundalloc.pyplot(fig2)
fundalloc.write('Fund allocations for 5 different fields: Alcohol, Fossil fuels, Small arms, Thermal coal, and Tobacco')

# Portfolio Score Distributions
fig3, ax3 = plt.subplots(figsize=(8,6))
scores = ['Portfolio Environmental Score','Portfolio Social Score','Portfolio Governance Score']
sns.barplot(x=[1,2,3],y=selectedfund[scores].values.flatten(), ax=ax3)
ax3.set_xticklabels(scores)
ax3.set_xlabel('Score Type')
ax3.set_ylabel('Score')
ax3.set_title(fund + ' Portfolio Scores')

portfolio.header('Portfolio Scores')
portfolio.pyplot(fig3)
portfolio.write('Portfolio Scores assigned by Morningstar in 3 differrent types')