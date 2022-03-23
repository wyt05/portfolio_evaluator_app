from tkinter import E
from soupsieve import select
import streamlit as st
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from streamlit_tags import st_tags
import yfinance as yf
import portfolio_func as port

def app():

    start_date_obj = datetime.strptime("01-01-2020", '%d-%m-%y')
    end_date_obj = datetime.strptime("31-12-2021", '%d-%m-%y')

    with st.form(key='stock_selection'):
        symbols = st_tags(label="Choose stocks to visualize",
                              text='Press enter to add more')
        date_cols = st.columns((1, 1))
        start_date = date_cols[0].date_input('Start Date', value=start_date_obj)
        end_date = date_cols[1].date_input('End Date', value=end_date_obj)
        submitted = st.form_submit_button('Submit')

    if submitted:

        ##Portfolio Creation
        N_PORTFOLIOS = 10 ** 5
        N_DAYS = 252
        RISKY_ASSETS = symbols
        RISKY_ASSETS.sort()
        START_DATE = start_date
        END_DATE = end_date

        portfolio_obj = port.Portfolio(RISKY_ASSETS, START_DATE, END_DATE, N_DAYS, N_PORTFOLIOS)

        








