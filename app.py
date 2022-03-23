import streamlit as st
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime  
from datetime import timedelta
import plotly.express as px
import requests
from bs4 import BeautifulSoup 
from multipage import MultiPage
from pages import stock_eval
from pages import portfolio

#https://towardsdatascience.com/creating-multipage-applications-using-streamlit-efficiently-b58a58134030

#Set page config
st.set_page_config(
     page_title="Stock Evaluator",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://www.extremelycoolapp.com/help',
         'Report a bug': "https://www.extremelycoolapp.com/bug",
         'About': "# This is a header. This is an *extremely* cool app!"
     }
 )

# Create app
app = MultiPage()

st.title("Stocks Evaluator Page")
st.write("Stonks only go up")


app.add_page("Stock Evaluation", stock_eval.app)
app.add_page("Portfolio", portfolio.app)

app.run()