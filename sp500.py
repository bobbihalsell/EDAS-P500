import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import base64
import matplotlib.pyplot as plt
import yfinance as yf

st.title('S&P500 Stock Price')

st.markdown("""
            This app retrieves the list of the **S&P500** (from wikipedia) and its corrosponding **stock closing price** (year-to-date)
            **Python Libraries: streamlit, pandas, requests, bs4, base64, matplotlib, yfinance
            **Data Source**: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
""")

st.sidebar.header('User Input Features')

@st.cache_data
#webscrape data
def load_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all tables on the page
    tables = soup.find_all('table', {'class': 'wikitable'})
    
    # The S&P 500 companies table is typically the first table on the page
    if tables:
        df = pd.read_html(str(tables[0]))[0]
        return df
    else:
        return None
    

# Load the data
df = load_data()

if df is not None:
    # Display the first few rows of the dataframe
    print(df.head())
    print(df.columns)
else:
    print("Failed to find the table.")


sector = df.groupby('GICS Sector')

#make selections in side bar
sorted_sectors = sorted(df['GICS Sector'].unique())
selected_sector = st.sidebar.multiselect('Sector', sorted_sectors)

# Download S&P500 data
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="SP500.csv">Download CSV File</a>'
    return href

# Plot Closing Price of Query Symbol
def plot_price(symbol):
    # Ensure we're working with a fresh plot
    plt.clf()
    plt.close('all')
    
    df = pd.DataFrame(data[symbol].Close)
    df['Date'] = df.index

    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot the data
    ax.fill_between(df['Date'], df['Close'], color='blue', alpha=0.3)
    ax.plot(df['Date'], df['Close'], color='blue')
    
    # Set the labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Close')
    ax.set_title(symbol, fontweight = 'bold')
    
    # Rotate and align the tick labels so they look better
    plt.xticks(rotation=90)
    
    # Use tight layout to prevent the label from being cut off
    plt.tight_layout()
    
    # Return the figure object
    return fig


# Check if any sectors are selected
if selected_sector:
    # Filter data based on selected sectors
    df_sector = df[df['GICS Sector'].isin(selected_sector)]

    st.header('Display Companies in Selected Sector')
    st.write(f'Data Dimension: {df_sector.shape[0]} rows, {df_sector.shape[1]} columns')
    st.write(f'Selected Sectors: {', '.join(selected_sector)}')

    st.dataframe(df_sector)

    # Download S&P500 stock data if companies are available
    if not df_sector.empty:
        st.markdown(filedownload(df_sector), unsafe_allow_html=True)

        # Download data from Yahoo Finance
        data = yf.download(
            tickers=list(df_sector.Symbol),
            period="ytd",
            interval="1d",
            group_by='ticker',
            auto_adjust=True,
            prepost=True,
            threads=True,
            proxy=None
        )

        # Plotting section
        num_company = st.sidebar.slider('Number of Companies', 1, df_sector.shape[0])
        if st.button('Show Plots'):
            st.header('Stock Closing Price')
            for i in list(df_sector.Symbol)[:num_company]:
                fig = plot_price(i)
                st.pyplot(fig)
    else:
        st.warning("No data available for the selected sector.")
else:
    # Show a message when no sectors are selected
    st.warning("No sectors selected to show.")







