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

