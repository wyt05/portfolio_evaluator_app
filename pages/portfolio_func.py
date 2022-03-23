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

class Portfolio:
    def __init__(self, symbols, start_date, end_date, n_days):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.n_days = n_days
        self.panel_info = yf.download(self.symbols, start=start_date, end=end_date)

    def get_full_info(self):

        return self.panel_info

    def get_close_dataframe(self):

        return self.panel_info['Close']
    
    def get_adj_close_dataframe(self):

        return self.panel_info['Adj Close']
    
    def get_return_series(self):
        
        return self.panel_info['Adj Close'].pct_change().dropna()
