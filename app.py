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
from pages import portfolio_pg, stock_eval, portfolio_creation, crypto_research
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


app.add_page("Stock Evaluation", stock_eval.app)
app.add_page("Crypto Research", crypto_research.app)
app.add_page("Portfolio Evaluation", portfolio_pg.app)
app.add_page("Portfolio Edit/Create", portfolio_creation.app)

app.run()