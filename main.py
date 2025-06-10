import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
from vnstock import Vnstock
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import random
import matplotlib.dates as mdates

# --- Configuration ---
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
pd.options.display.float_format = '{:,.0f}'.format

MODEL_FILENAME = 'lstm_stock_model_vhc.h5'
SEQUENCE_LENGTH = 60
DEFAULT_SOURCE = 'VCI'

FEATURES = [ # Original 12 features
    'close', 'open', 'high', 'low', 'volume',
    'MA5', 'MA20', 'RSI', 'MACD', 'Signal_Line',
    'Price_Change', 'Volatility'
]

# Parameters for randomized H/L estimation
MIN_HL_BASELINE_PCT = 0.01  # Min 1%
MAX_HL_BASELINE_PCT = 0.03  # Max 3%
MIN_VOL_MULTIPLIER = 0.8
MAX_VOL_MULTIPLIER = 2.2

try:
    FEATURE_INDEX_CLOSE = FEATURES.index('close')
    # ... (các FEATURE_INDEX_ khác không cần thiết cho logic chính của predict_with_monte_carlo nữa
    # vì chúng ta xây dựng new_day_ohlcv qua dictionary)
except ValueError:
    st.error("CRITICAL ERROR: 'close' feature not in FEATURES list.")
    st.stop()

# --- Utility functions (calculate_rsi, calculate_macd, add_all_features_to_df, load_lstm_model) ---
# (Giữ nguyên các hàm này như phiên bản trước đã dùng 12 features)
def calculate_rsi(data_series, window=14):
    delta = data_series.diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.replace([np.inf, -np.inf], 100).fillna(50)
    return rsi

def calculate_macd(data_series, fast=12, slow=26, signal=9):
    exp1 = data_series.ewm(span=fast, adjust=False, min_periods=1).mean()
    exp2 = data_series.ewm(span=slow, adjust=False, min_periods=1).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False, min_periods=1).mean()
    return macd, signal_line

def add_all_features_to_df(df_ohlcv): # Adapted for 12 features
    df = df_ohlcv.copy()
    if len(df) < 2:
        for col in ['MA5', 'MA20', 'RSI', 'MACD', 'Signal_Line', 'Price_Change', 'Volatility']:
            df[col] = np.nan
        return df

    df['MA5'] = df['close'].rolling(window=5, min_periods=1).mean()
    df['MA20'] = df['close'].rolling(window=20, min_periods=1).mean()
    df['RSI'] = calculate_rsi(df['close'])
    df['MACD'], df['Signal_Line'] = calculate_macd(df['close'])
    df['Price_Change'] = df['close'].pct_change()
    df['Volatility'] = df['close'].rolling(window=5, min_periods=1).std()
    return df

@st.cache_data
def load_lstm_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Error: Model file not found at '{model_path}'.")
        return None
    try:
        _model = load_model(model_path)
        return _model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def format_currency(value):
    return f"{value:,.0f}"

def predict_with_monte_carlo(model_mc, df_historical_with_features, scaler_obj, sequence_length_val, features_list_model, num_days_to_predict, num_simulations=20):
    essential_cols_for_feature_calc = ['open', 'high', 'low', 'close', 'volume']
    
    historical_close_pct_change = df_historical_with_features['close'].pct_change().dropna()
    if len(historical_close_pct_change) >= 20:
        recent_volatility_metric = historical_close_pct_change.rolling(window=20).std().iloc[-1]
        avg_daily_volume = df_historical_with_features['volume'].iloc[-20:].mean()
        avg_hl_range_pct = ((df_historical_with_features['high'].iloc[-20:] - df_historical_with_features['low'].iloc[-20:]) / df_historical_with_features['close'].iloc[-20:]).mean()
    elif len(historical_close_pct_change) > 0: # Dùng toàn bộ lịch sử nếu ít hơn 20 ngày
        recent_volatility_metric = historical_close_pct_change.std()
        avg_daily_volume = df_historical_with_features['volume'].mean()
        avg_hl_range_pct = ((df_historical_with_features['high'] - df_historical_with_features['low']) / df_historical_with_features['close']).mean()

    else:
        recent_volatility_metric, avg_daily_volume, avg_hl_range_pct = 0.015, 100000, 0.025 # Defaults for very short history
        
    recent_volatility_metric = max(0.005, 0.015 if pd.isna(recent_volatility_metric) else recent_volatility_metric) # Min 0.5% vol
    avg_daily_volume = 100000 if pd.isna(avg_daily_volume) or avg_daily_volume == 0 else avg_daily_volume
    avg_hl_range_pct = max(0.01, 0.025 if pd.isna(avg_hl_range_pct) else avg_hl_range_pct) # Min 1% range

    all_simulations_prices = []
    base_predicted_dates = []
    last_known_date = df_historical_with_features.index[-1]
    
    for i in range(num_days_to_predict):
        base_predicted_dates.append(pd.to_datetime(last_known_date) + pd.offsets.BDay(i + 1))
    
    for sim_idx in range(num_simulations):
        df_sim_ohlcv = df_historical_with_features[essential_cols_for_feature_calc].copy()
        current_sequence_unscaled_np = df_historical_with_features[features_list_model].values[-sequence_length_val:].copy()
        sim_prices_run = []
        
        current_day_volatility_multiplier = random.uniform(MIN_VOL_MULTIPLIER, MAX_VOL_MULTIPLIER)
        current_day_hl_baseline_pct = random.uniform(MIN_HL_BASELINE_PCT, MAX_HL_BASELINE_PCT)

        for day_idx in range(num_days_to_predict):
            current_sequence_scaled = scaler_obj.transform(current_sequence_unscaled_np)
            X_next = np.array([current_sequence_scaled])
            pred_scaled_close = model_mc.predict(X_next, verbose=0)[0,0]
            
            temp_arr_scaled = np.zeros((1, len(features_list_model)))
            temp_arr_scaled[0, FEATURE_INDEX_CLOSE] = pred_scaled_close
            pred_actual_close_no_noise = scaler_obj.inverse_transform(temp_arr_scaled)[0, FEATURE_INDEX_CLOSE]
            
            # Apply Monte Carlo noise to close price
            # Noise magnitude decreases slightly for further days but has a strong base
            noise_magnitude_for_day = recent_volatility_metric * (1.0 - 0.02 * day_idx) * random.uniform(0.8, 1.2)
            noise_magnitude_for_day = max(noise_magnitude_for_day, 0.003) # Min 0.3% noise on close
            
            pred_actual_close_w_noise = pred_actual_close_no_noise * (1.0 + random.uniform(-noise_magnitude_for_day, noise_magnitude_for_day))
            sim_prices_run.append(pred_actual_close_w_noise)
            
            if day_idx < num_days_to_predict - 1:
                curr_pred_date = base_predicted_dates[day_idx]
                prev_close = df_sim_ohlcv['close'].iloc[-1]
                
                # Estimate Open: around previous close with some noise
                open_noise = random.uniform(-noise_magnitude_for_day * 0.5, noise_magnitude_for_day * 0.3)
                est_open = prev_close * (1 + open_noise)

                # Estimate High/Low with more randomness
                # Base range on historical average daily range percentage + current volatility
                dynamic_range_pct = avg_hl_range_pct * (1 + recent_volatility_metric * current_day_volatility_multiplier) * random.uniform(0.7, 1.5)
                dynamic_range_pct = max(current_day_hl_baseline_pct, min(dynamic_range_pct, 0.1)) # Cap range pct (1% to 10%)
                
                estimated_day_abs_range = pred_actual_close_w_noise * dynamic_range_pct
                
                # Determine if it's more likely an up-day or down-day for H/L placement
                price_direction_factor = random.uniform(-1, 1) # -1 (down), 0 (center), 1 (up)
                
                # High is above open and close, Low is below open and close
                est_high = max(est_open, pred_actual_close_w_noise) + estimated_day_abs_range * (0.5 + 0.5 * price_direction_factor * random.random())
                est_low = min(est_open, pred_actual_close_w_noise) - estimated_day_abs_range * (0.5 - 0.5 * price_direction_factor * random.random())

                # Ensure OHLC consistency
                est_high = max(est_high, est_open, pred_actual_close_w_noise)
                est_low = min(est_low, est_open, pred_actual_close_w_noise)
                if est_low >= est_high: # If low is not lower than high
                    if est_low > pred_actual_close_w_noise: est_low = pred_actual_close_w_noise * (1 - noise_magnitude_for_day * 0.1)
                    if est_high < pred_actual_close_w_noise: est_high = pred_actual_close_w_noise * (1 + noise_magnitude_for_day * 0.1)
                    if est_low >= est_high: # Final check
                        est_low = est_high * (1 - 0.002) # Ensure low is at least slightly lower

                # Estimate Volume with more randomness
                price_chg_abs_from_prev = abs((pred_actual_close_w_noise / prev_close) - 1) if prev_close != 0 else 0
                vol_mult_price = 1 + price_chg_abs_from_prev * random.uniform(0.2, 2.5) # Wider range
                est_vol = max(1000, avg_daily_volume * vol_mult_price * random.uniform(0.5, 1.8)) # Wider random factor, min volume

                new_day_ohlcv = {'open': est_open, 'high': est_high, 'low': est_low, 
                                 'close': pred_actual_close_w_noise, 'volume': est_vol}
                new_day_series = pd.Series(new_day_ohlcv, name=curr_pred_date)
                
                df_sim_ohlcv = pd.concat([df_sim_ohlcv, new_day_series.to_frame().T])
                df_sim_ohlcv.index = pd.to_datetime(df_sim_ohlcv.index)
                
                df_sim_featured = add_all_features_to_df(df_sim_ohlcv)
                df_sim_featured_cleaned = df_sim_featured.fillna(method='bfill').fillna(method='ffill')
                current_sequence_unscaled_np = df_sim_featured_cleaned[features_list_model].values[-sequence_length_val:]
        
        all_simulations_prices.append(sim_prices_run)
    
    sim_array = np.array(all_simulations_prices)
    prediction_results = {
        'dates': base_predicted_dates, 
        'mean': np.mean(sim_array, axis=0),
        'lower_80': np.percentile(sim_array, 10, axis=0), 
        'upper_80': np.percentile(sim_array, 90, axis=0),
        'lower_95': np.percentile(sim_array, 2.5, axis=0), 
        'upper_95': np.percentile(sim_array, 97.5, axis=0),
        'all_simulations': sim_array
    }
    return prediction_results

# --- generate_technical_analysis (giữ nguyên như phiên bản trước đã dùng 12 features) ---
def generate_technical_analysis(df_hist, latest_price_val, predicted_mean_vals):
    analysis = {}
    if df_hist.empty or not isinstance(df_hist, pd.DataFrame): return analysis
    last_r = df_hist.iloc[-1]
    ma5 = last_r.get('MA5', np.nan)
    ma20 = last_r.get('MA20', np.nan)
    rsi_val = last_r.get('RSI', 50)
    macd_val = last_r.get('MACD', 0)
    signal_val = last_r.get('Signal_Line', 0)
    
    trend, trend_c = "Sideways", "gray"
    if not any(pd.isna([ma5, ma20])):
        if ma5 > ma20: trend, trend_c = "Uptrend", "green"
        elif ma5 < ma20: trend, trend_c = "Downtrend", "red"

    rsi_cond, rsi_c = "Neutral", "gray"
    if not pd.isna(rsi_val):
        if rsi_val > 70: rsi_cond, rsi_c = "Overbought", "red"
        elif rsi_val < 30: rsi_cond, rsi_c = "Oversold", "green"

    macd_sig, macd_c = "Neutral", "gray"
    if not any(pd.isna([macd_val, signal_val])):
        if macd_val > signal_val and macd_val > 0: macd_sig, macd_c = "Strong Buy", "darkgreen"
        elif macd_val > signal_val: macd_sig, macd_c = "Buy", "green"
        elif macd_val < signal_val and macd_val < 0: macd_sig, macd_c = "Strong Sell", "darkred"
        elif macd_val < signal_val: macd_sig, macd_c = "Sell", "red"
    
    chg_1d, chg_nd = 0, 0
    if predicted_mean_vals is not None and len(predicted_mean_vals) > 0 and not pd.isna(latest_price_val) and latest_price_val != 0:
        chg_1d = ((predicted_mean_vals[0] / latest_price_val) - 1) * 100
        chg_nd = ((predicted_mean_vals[-1] / latest_price_val) - 1) * 100

    analysis = {'trend': {'value': trend, 'color': trend_c}, 
                'rsi': {'value': rsi_cond, 'color': rsi_c, 'number': rsi_val}, 
                'macd': {'value': macd_sig, 'color': macd_c},
                'ma5': ma5, 'ma20': ma20, 
                'price_change_1d': chg_1d, 
                f'price_change_{len(predicted_mean_vals) if predicted_mean_vals is not None else "N"}d': chg_nd}
    return analysis


# --- Streamlit Interface (giữ nguyên như phiên bản trước) ---
st.set_page_config(page_title="LSTM Stock Price Prediction", layout="wide", initial_sidebar_state="collapsed")
# ... (Toàn bộ phần UI Streamlit từ st.title(...) đến st.markdown("---") giữ nguyên) ...
# Dán phần UI Streamlit từ phiên bản trước vào đây
st.title("📈 Stock Price Prediction App (LSTM & Monte Carlo)")

tabs_main = st.tabs(["**🚀 Predictor**", "**ℹ️ About & Methodology**"])

with tabs_main[0]:
    st.markdown("Enter a stock symbol and the number of future days to predict. The app uses an LSTM model (trained on 12 features) enhanced with Monte Carlo simulations.")
    input_cols = st.columns([2,1,1])
    stock_symbol_input = input_cols[0].text_input("Enter stock symbol (e.g., FPT, VCB, HPG):", "FPT", key="stock_symbol_main").upper()
    num_days_to_predict_input = input_cols[1].number_input("Days to predict:", min_value=1, max_value=100, value=3, step=1, key="num_days_main") # Max 100 for performance
    num_simulations_mc_input = input_cols[2].slider("Monte Carlo simulations:", min_value=10, max_value=50, value=20, step=5, key="num_sim_main", help="More simulations can improve accuracy but take longer.")
    
    predict_button = st.button(f"🔮 Predict {num_days_to_predict_input} days for {stock_symbol_input}", use_container_width=True, type="primary")

with tabs_main[1]:
    st.subheader("About This Application")
    st.markdown("""
    This app forecasts Vietnamese stock prices using a hybrid of Long Short-Term Memory (LSTM) neural networks and Monte Carlo simulations. 
    - The **LSTM model** is trained on historical price data and 12 key technical indicators (MAs, RSI, MACD, etc.).
    - **Monte Carlo simulations** generate multiple future price paths by introducing calculated randomness, reflecting market volatility and providing a range of outcomes with confidence intervals.
    """)
    st.subheader("Methodology in Brief")
    st.markdown("""
    1.  **Data Collection**: Historical OHLCV data is fetched using `vnstock`.
    2.  **Feature Engineering**: 12 technical indicators are computed.
    3.  **Data Scaling**: Features are scaled using `MinMaxScaler`.
    4.  **LSTM Prediction**: A pre-trained LSTM model predicts the *next day's closing price*.
    5.  **Iterative Monte Carlo**: For multi-day forecasts, the predicted close price + estimated OHLCV + re-calculated technical indicators form the input for the next day's prediction, repeated for multiple simulations.
    6.  **Results**: Mean prediction and confidence intervals are displayed.
    """)
    st.warning("⚠️ **Disclaimer**: Predictions are for informational purposes only, not financial advice.")


if predict_button:
    if not stock_symbol_input:
        st.error("Please enter a stock symbol.")
    elif not os.path.exists(MODEL_FILENAME):
        st.error(f"Error: Model file '{MODEL_FILENAME}' not found.")
    else:
        with st.spinner(f"Processing for {stock_symbol_input} ({num_simulations_mc_input} simulations)..."):
            try:
                end_date_dt = datetime.today()
                days_buffer_val = SEQUENCE_LENGTH + 20 + 60 
                
                start_date_dt = end_date_dt - timedelta(days=days_buffer_val * 2) 
                end_date_str_fetch, start_date_str_fetch = end_date_dt.strftime('%Y-%m-%d'), start_date_dt.strftime('%Y-%m-%d')

                stock_loader_obj = Vnstock().stock(symbol=stock_symbol_input, source=DEFAULT_SOURCE)
                df_raw_data = stock_loader_obj.quote.history(start=start_date_str_fetch, end=end_date_str_fetch, interval='1D')
                
                if df_raw_data.empty or len(df_raw_data) < (SEQUENCE_LENGTH + 20):
                    st.error(f"Insufficient historical data for {stock_symbol_input} (got {len(df_raw_data)} days, "
                             f"need at least {SEQUENCE_LENGTH + 20} for features).")
                    st.stop()

                df_clean_data = df_raw_data[['time', 'open', 'high', 'low', 'close', 'volume']].copy()
                df_clean_data['time'] = pd.to_datetime(df_clean_data['time'])
                df_clean_data.set_index('time', inplace=True)
                df_clean_data.sort_index(inplace=True)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df_clean_data[col] = pd.to_numeric(df_clean_data[col], errors='coerce')
                df_clean_data.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)

                if len(df_clean_data) < (SEQUENCE_LENGTH + 20):
                    st.error(f"Insufficient clean data for {stock_symbol_input} after cleaning.")
                    st.stop()
                
                df_hist_feat_raw = add_all_features_to_df(df_clean_data)
                df_hist_feat = df_hist_feat_raw.fillna(method='bfill').fillna(method='ffill')
                
                final_seq_candidate = df_hist_feat.iloc[-SEQUENCE_LENGTH:][FEATURES]
                if final_seq_candidate.isnull().any().any():
                    st.error("Data cleaning issue: NaN values in features for LSTM input.")
                    st.stop()
                if df_hist_feat.shape[0] < SEQUENCE_LENGTH:
                    st.error(f"Not enough data ({df_hist_feat.shape[0]} days) for LSTM sequence.")
                    st.stop()

                data_to_scale_hist = df_hist_feat[FEATURES].copy()
                scaler_instance = MinMaxScaler(feature_range=(0, 1))
                scaler_instance.fit(data_to_scale_hist.dropna())

                model_loaded = load_lstm_model(MODEL_FILENAME)
                if model_loaded is None: st.stop()

                prediction_output_data = predict_with_monte_carlo(
                    model_loaded, df_hist_feat, scaler_instance, 
                    SEQUENCE_LENGTH, FEATURES,
                    num_days_to_predict_input, num_simulations_mc_input
                )

                if prediction_output_data['mean'].size > 0:
                    st.success(f"**Prediction for {stock_symbol_input} - Next {num_days_to_predict_input} Trading Days**")
                    
                    latest_price_val = df_hist_feat['close'].iloc[-1]
                    tech_analysis_data = generate_technical_analysis(df_hist_feat, latest_price_val, prediction_output_data['mean'])
                    
                    summary_cols_disp = st.columns(5)
                    summary_cols_disp[0].metric("Last Close", f"{format_currency(latest_price_val)} VND")
                    summary_cols_disp[1].metric(
                        f"Day 1 (Mean)", f"{format_currency(prediction_output_data['mean'][0])} VND",
                        delta=f"{tech_analysis_data['price_change_1d']:.2f}%"
                    )
                    summary_cols_disp[2].markdown(f"**Trend**: <span style='color:{tech_analysis_data['trend']['color']};font-weight:bold;'>{tech_analysis_data['trend']['value']}</span>", unsafe_allow_html=True)
                    summary_cols_disp[3].markdown(f"**RSI ({tech_analysis_data['rsi']['number']:.1f})**: <span style='color:{tech_analysis_data['rsi']['color']};font-weight:bold;'>{tech_analysis_data['rsi']['value']}</span>", unsafe_allow_html=True)
                    summary_cols_disp[4].markdown(f"**MACD**: <span style='color:{tech_analysis_data['macd']['color']};font-weight:bold;'>{tech_analysis_data['macd']['value']}</span>", unsafe_allow_html=True)
                    
                    st.divider()
                    
                    pred_df_display = pd.DataFrame({
                        'Date': [d.strftime('%d/%m (%a)') for d in prediction_output_data['dates']],
                        'Mean (VND)': prediction_output_data['mean'],
                        '80% CI Low': prediction_output_data['lower_80'], '80% High': prediction_output_data['upper_80'],
                        '95% CI Low': prediction_output_data['lower_95'], '95% CI High': prediction_output_data['upper_95'],
                        'Change (%)': ((prediction_output_data['mean'] / latest_price_val) - 1) * 100 if latest_price_val != 0 else 0
                    })
                    st.dataframe(pred_df_display.style.format({
                        'Mean (VND)': "{:,.0f}", '80% CI Low': "{:,.0f}", '80% CI High': "{:,.0f}",
                        '95% CI Low': "{:,.0f}", '95% CI High': "{:,.0f}", 'Change (%)': "{:.2f}%"
                    }).background_gradient(cmap='RdYlGn_r', subset=['Change (%)'], vmin=-5, vmax=5), 
                    use_container_width=True)

                    st.subheader("Price Prediction Chart with Confidence Intervals")
                    history_plot_data = df_hist_feat['close'].iloc[-60:] # Dữ liệu lịch sử để vẽ
                    
                    # Nối điểm lịch sử cuối cùng vào đầu chuỗi dự đoán để vẽ liền mạch
                    plot_pred_dates = [history_plot_data.index[-1]] + prediction_output_data['dates']
                    plot_mean_preds = np.insert(prediction_output_data['mean'], 0, history_plot_data.iloc[-1])
                    
                    fig_pred, ax_pred = plt.subplots(figsize=(15, 8))
                    ax_pred.plot(history_plot_data.index, history_plot_data.values, label='Historical Close', color='dodgerblue', lw=2)
                    ax_pred.plot(plot_pred_dates, plot_mean_preds, 'o-', color='red', ms=5, lw=2, label='Mean Prediction')
                    
                    # CI bắt đầu từ ngày dự đoán đầu tiên
                    ax_pred.fill_between(prediction_output_data['dates'], prediction_output_data['lower_95'], prediction_output_data['upper_95'], color='tomato', alpha=0.2, label='95% CI')
                    ax_pred.fill_between(prediction_output_data['dates'], prediction_output_data['lower_80'], prediction_output_data['upper_80'], color='orange', alpha=0.3, label='80% CI')

                    if prediction_output_data['mean'].size > 0:
                        last_p_dt_ann, last_p_val_ann = prediction_output_data['dates'][-1], prediction_output_data['mean'][-1]
                        ax_pred.annotate(f"{format_currency(last_p_val_ann)}", (mdates.date2num(last_p_dt_ann), last_p_val_ann),
                                         xytext=(5, 5), textcoords='offset points', ha='left', va='bottom',
                                         bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))

                    ax_pred.set_title(f'{stock_symbol_input} - {num_days_to_predict_input}-Day Price Forecast', fontsize=18, fontweight='bold')
                    ax_pred.set_xlabel('Date', fontsize=14)
                    ax_pred.set_ylabel('Price (VND)', fontsize=14)
                    ax_pred.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
                    ax_pred.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=20)) # Tăng maxticks
                    plt.xticks(rotation=30, ha="right")
                    ax_pred.legend(fontsize=10, loc='upper left')
                    ax_pred.grid(True, linestyle=':', alpha=0.6)
                    plt.tight_layout()
                    st.pyplot(fig_pred)
                else:
                    st.warning("Could not generate predictions for the specified parameters.")
            except Exception as e_main:
                st.error(f"A critical error occurred: {str(e_main)}")
                st.exception(e_main)
                st.error("Please try again or check the stock symbol and data availability.")

st.markdown("---")
