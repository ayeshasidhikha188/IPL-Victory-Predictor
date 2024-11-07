import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load models and scalers
one_hotencoder = pickle.load(open('one_hotencoder.pkl', 'rb'))
rf_model = pickle.load(open('randomforest.pkl', 'rb'))
scaler = pickle.load(open('standardscaler.pkl', 'rb'))

# Load datasets
matches_data = pd.read_csv('matches(1).csv')
ballbyball_data = pd.read_csv('ballbyball(1).csv')
auction_data = pd.read_csv('auction.csv')
ipl_data = pd.read_csv('ipl_data.csv')

# Set the background image function
def set_background_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Function for IPL Data Analysis
def page_ipl_data_analysis():
    set_background_image("https://wallpapercave.com/wp/wp9765076.jpg")
    
    # Set the page title in white and bold
    st.markdown("<h1 style='color:white; font-weight:bold; font-size:30px;'>IPL Data Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:white; font-weight:bold; font-size:24px;'>Analyzing IPL Team Performance</h2>", unsafe_allow_html=True)
    
    top_30_ipl_data = ipl_data.sort_values(by='total_runs_x', ascending=False).head(30)
    
    # Custom chart titles in cream color (#fffdd0)
    st.plotly_chart(px.bar(top_30_ipl_data, x='batting_team', y='total_runs_x', color='bowling_team', title="Top 30 Runs by Teams", template="plotly_dark")
                    .update_layout(title_font=dict(size=20, color='#fffdd0')), key="ipl_runs_chart")
    st.plotly_chart(px.scatter(top_30_ipl_data, x='crr', y='rrr', color='city', title="Top 30 Current vs Required Run Rate", template="plotly_dark")
                    .update_layout(title_font=dict(size=20, color='#fffdd0')), key="ipl_run_rate_chart")
    st.plotly_chart(px.histogram(top_30_ipl_data, x='city', color='result', title="Top 30 Results by City", template="plotly_dark")
                    .update_layout(title_font=dict(size=20, color='#fffdd0')), key="ipl_results_chart")
    st.plotly_chart(px.box(top_30_ipl_data, x='batting_team', y='wickets_left', title="Top 30 Wickets Left by Batting Team", template="plotly_dark")
                    .update_layout(title_font=dict(size=20, color='#fffdd0')), key="ipl_wickets_chart")

# Function for Ball by Ball Data Analysis
def page_ball_by_ball_analysis():
    set_background_image("https://img.freepik.com/premium-photo/red-ball-hitting-wicket-stumps-with-bat-black-abstract-splash-background-cricket-fever_1023984-17218.jpg")
    
    # Set the page title in white and bold
    st.markdown("<h1 style='color:white; font-weight:bold; font-size:30px;'>Ball by Ball Data Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:white; font-weight:bold; font-size:24px;'>Detailed Analysis of Ball-by-Ball Performance</h2>", unsafe_allow_html=True)
    
    top_30_ballbyball_data = ballbyball_data.sort_values(by='total_run', ascending=False).head(30)
    
    # Custom chart titles in cream color (#fffdd0)
    st.plotly_chart(px.line(top_30_ballbyball_data, x='overs', y='total_run', color='BattingTeam', title="Runs Over Time", template="plotly_dark")
                    .update_layout(title_font=dict(size=20, color='#fffdd0')), key="ball_by_ball_runs_chart")
    st.plotly_chart(px.bar(top_30_ballbyball_data, x='batter', y='batsman_run', color='bowler', title="Runs by Batsmen vs Bowlers", template="plotly_dark")
                    .update_layout(title_font=dict(size=20, color='#fffdd0')), key="ball_by_ball_batsmen_vs_bowlers_chart")
    st.plotly_chart(px.histogram(top_30_ballbyball_data, x='isWicketDelivery', title="Wickets by Delivery", template="plotly_dark")
                    .update_layout(title_font=dict(size=20, color='#fffdd0')), key="ball_by_ball_wickets_chart")
    st.plotly_chart(px.scatter(top_30_ballbyball_data, x='overs', y='batsman_run', color='batter', title="Runs by Overs", template="plotly_dark")
                    .update_layout(title_font=dict(size=20, color='#fffdd0')), key="ball_by_ball_runs_by_overs_chart")

# Function for Auction Data Analysis
def page_auction_data_analysis():
    set_background_image("https://img.jagranjosh.com/images/2023/December/19122023/ipl-auction-2024-sold-unsold-players-list.webp")
    
    # Set the page title in white and bold
    st.markdown("<h1 style='color:white; font-weight:bold; font-size:30px;'>Auction Data Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:white; font-weight:bold; font-size:24px;'>Examining Auction Trends and Player Bids</h2>", unsafe_allow_html=True)
    
    top_30_auction_data = auction_data.sort_values(by='Winning bid', ascending=False).head(30)
    
    # Custom chart titles in cream color (#fffdd0)
    st.plotly_chart(px.bar(top_30_auction_data, x='Team', y='Winning bid', color='Player', title="Bids by Team", template="plotly_dark")
                    .update_layout(title_font=dict(size=20, color='#fffdd0')), key="auction_bids_by_team_chart")
    st.plotly_chart(px.box(top_30_auction_data, x='Country', y='Winning bid', title="Auction Bids by Country", template="plotly_dark")
                    .update_layout(title_font=dict(size=20, color='#fffdd0')), key="auction_bids_by_country_chart")
    st.plotly_chart(px.pie(top_30_auction_data, names='Year', values='Winning bid', title="Auction Distribution by Year", template="plotly_dark")
                    .update_layout(title_font=dict(size=20, color='#fffdd0')), key="auction_distribution_by_year_chart")
    st.plotly_chart(px.scatter(top_30_auction_data, x='Player', y='Winning bid', color='Country', title="Player Winning Bids", template="plotly_dark")
                    .update_layout(title_font=dict(size=20, color='#fffdd0')), key="player_winning_bids_chart")

# Prediction function
def predict_win_probability(batting_team, bowling_team, city, target, current_score, overs_remaining):
    input_data = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'target_score': [target],
        'current_score': [current_score],
        'overs_remaining': [overs_remaining]
    })
    
    # One-hot encode categorical features
    input_data_encoded = one_hotencoder.transform(input_data)
    
    # Scale numerical features
    input_data_scaled = scaler.transform(input_data_encoded)
    
    # Predict the win probability
    win_probability = rf_model.predict_proba(input_data_scaled)[:, 1]
    
    return win_probability[0]

# Set the page config
st.set_page_config(page_title="IPL Victory Predictor", layout="wide")

# Background image URL
background_image_url = "https://i.ytimg.com/vi/tI04mAMwQfc/maxresdefault.jpg?sqp=-oaymwEmCIAKENAF8quKqQMa8AEB-AH-CYAC0AWKAgwIABABGFEgZSgbMA8=&rs=AOn4CLAp86bTrdXS-g-uoVmFasKwD3nSTw"

# Inject CSS to set the background image and styles
st.markdown(
    f"""
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: rgba(0, 0, 0, 0.8); 
            color: white; 
            padding: 0;
            margin: 0;
            text-align: center;
            position: relative; 
        }}
        .background-image {{
            position: absolute; 
            top: 0; 
            left: 0; 
            width: 100%; 
            height: 100%; 
            background-image: url('{background_image_url}');
            background-size: cover; 
            background-position: center; 
            z-index: -1; 
            opacity: 0.2; 
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Add a title
st.title("IPL Victory Predictor")
st.write("Choose an option to analyze IPL data or predict match outcomes.")

# Sidebar for navigation
page = st.sidebar.selectbox("Select Page", ("IPL Data Analysis", "Ball by Ball Analysis", "Auction Data Analysis", "Win Probability Predictor"))

# Show the selected page
if page == "IPL Data Analysis":
    page_ipl_data_analysis()
elif page == "Ball by Ball Analysis":
    page_ball_by_ball_analysis()
elif page == "Auction Data Analysis":
    page_auction_data_analysis()
elif page == "Win Probability Predictor":
    set_background_image("https://images.hdqwalls.com/wallpapers/cricket-sport-wallpaper-4k-8b.jpg")
    
    # Set the page title in white and bold
    st.markdown("<h1 style='color:white; font-weight:bold; font-size:30px;'>Win Probability Predictor</h1>", unsafe_allow_html=True)
    
    # User inputs for win probability prediction
    batting_team = st.selectbox("Select Batting Team", sorted(ipl_data['batting_team'].unique()))
    bowling_team = st.selectbox("Select Bowling Team", sorted(ipl_data['bowling_team'].unique()))
    city = st.selectbox("Select Match City", sorted(ipl_data['city'].unique()))
    target_score = st.number_input("Enter Target Score", min_value=1)
    current_score = st.number_input("Enter Current Score", min_value=0)
    overs_remaining = st.number_input("Enter Overs Remaining", min_value=0.0, format="%.1f")
    
    if st.button("Predict Win Probability"):
        win_probability = predict_win_probability(batting_team, bowling_team, city, target_score, current_score, overs_remaining)
        st.write(f"The predicted win probability for {batting_team} is: {win_probability:.2%}")