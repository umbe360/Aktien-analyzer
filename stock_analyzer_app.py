import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Seitenkonfiguration
st.set_page_config(
    page_title="Aktien & Krypto Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titel und Beschreibung
st.title("Aktien & Krypto Analyzer")
st.markdown("Technische Analyse für Aktien und Kryptowährungen mit Kauf- und Verkaufssignalen")

# Sidebar für Eingaben
st.sidebar.header("Einstellungen")

# Analysetyp auswählen
analysis_type = st.sidebar.radio(
    "Analysetyp auswählen:",
    ["Einzelanalyse", "Mehrfachanalyse"]
)

# Zeitraum und Intervall
period_options = {
    "1 Monat": "1mo",
    "3 Monate": "3mo",
    "6 Monate": "6mo",
    "1 Jahr": "1y",
    "2 Jahre": "2y",
    "5 Jahre": "5y"
}
period = st.sidebar.selectbox("Zeitraum:", list(period_options.keys()))

interval_options = {
    "Täglich": "1d",
    "Wöchentlich": "1wk",
    "Monatlich": "1mo"
}
interval = st.sidebar.selectbox("Intervall:", list(interval_options.keys()))

# Standardlisten
st.sidebar.header("Standardlisten")
if st.sidebar.button("Tech-Aktien laden"):
    symbols_text = "AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA, AMD, NFLX, ADBE, ABNB, TEAM, SQ, BKNG, AVGO, COIN, EBAY, EA, FISV, MSTR, ORCL, PLTR, PYPL, CRM, SHOP, SNOW, TTWO, NET, UBER, ZS"
    st.session_state.symbols = symbols_text

if st.sidebar.button("Weitere Aktien laden"):
    symbols_text = "AI, XPEV, ISRG, TDOC, ILMN, LUMN, DELL, BABA, MRVL, MU, CRWD, SPOT"
    st.session_state.symbols = symbols_text

if st.sidebar.button("Kryptowährungen laden"):
    symbols_text = "BTC-USD, ETH-USD, BNB-USD, XRP-USD, ADA-USD, SOL-USD, DOGE-USD, DOT-USD, AVAX-USD, MATIC-USD"
    st.session_state.symbols = symbols_text

# Klasse für die technische Analyse
class StockAnalyzer:
    def __init__(self, period="1mo", interval="1d"):
        self.period = period
        self.interval = interval
    
    def fetch_data(self, symbol):
        try:
            data = yf.download(symbol, period=self.period, interval=self.interval, progress=False)
            
            if data.empty or len(data) < 20:
                return None
                
            return data
        except Exception as e:
            st.error(f"Fehler beim Holen der Daten für {symbol}: {e}")
            return None
    
    def calculate_indicators(self, data):
        # Kopie der Daten erstellen
        df = data.copy()
        
        # Gleitende Durchschnitte
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        
        # Exponentiell gleitende Durchschnitte
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD (Moving Average Convergence Divergence)
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
        
        return df
    
    def generate_signals(self, df, symbol):
        # Letzte Zeile für aktuelle Werte
        current = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else current
        
        # Aktuelle Werte
        current_price = current['Close']
        current_date = df.index[-1].strftime('%Y-%m-%d')
        
        # Signale initialisieren
        signals = {
            'symbol': symbol,
            'price': current_price,
            'date': current_date,
            'signals': {
                'sma_crossover': 0,
                'macd': 0,
                'rsi': 0,
                'bollinger': 0
            },
            'support_resistance': {
                'support': 0,
                'resistance': 0
            },
            'indicators': {
                'sma20': current['SMA20'] if not np.isnan(current['SMA20']) else 0,
                'sma50': current['SMA50'] if not np.isnan(current['SMA50']) else 0,
                'sma200': current['SMA200'] if not np.isnan(current['SMA200']) else 0,
                'rsi': current['RSI'] if not np.isnan(current['RSI']) else 0,
                'macd': current['MACD'] if not np.isnan(current['MACD']) else 0,
                'macd_signal': current['MACD_Signal'] if not np.isnan(current['MACD_Signal']) else 0,
                'bb_upper': current['BB_Upper'] if not np.isnan(current['BB_Upper']) else 0,
                'bb_lower': current['BB_Lower'] if not np.isnan(current['BB_Lower']) else 0
            }
        }
        
        # SMA Crossover Signal
        if not np.isnan(current['SMA20']) and not np.isnan(current['SMA50']):
            if current['SMA20'] > current['SMA50'] and prev['SMA20'] <= prev['SMA50']:
                signals['signals']['sma_crossover'] = 1  # Kaufsignal
            elif current['SMA20'] < current['SMA50'] and prev['SMA20'] >= prev['SMA50']:
                signals['signals']['sma_crossover'] = -1  # Verkaufssignal
            
        # MACD Signal
        if not np.isnan(current['MACD']) and not np.isnan(current['MACD_Signal']):
            if current['MACD'] > current['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
                signals['signals']['macd'] = 1  # Kaufsignal
            elif current['MACD'] < current['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
                signals['signals']['macd'] = -1  # Verkaufssignal
            
        # RSI Signal
        if not np.isnan(current['RSI']):
            if current['RSI'] < 30:
                signals['signals']['rsi'] = 1  # Überverkauft - Kaufsignal
            elif current['RSI'] > 70:
                signals['signals']['rsi'] = -1  # Überkauft - Verkaufssignal
            
        # Bollinger Bands Signal
        if not np.isnan(current['BB_Lower']) and not np.isnan(current['BB_Upper']):
            if current['Close'] < current['BB_Lower']:
                signals['signals']['bollinger'] = 1  # Kaufsignal
            elif current['Close'] > current['BB_Upper']:
                signals['signals']['bollinger'] = -1  # Verkaufssignal
            
        # Support und Widerstand berechnen (einfache Methode)
        recent_data = df.iloc[-min(100, len(df)):].copy()
        signals['support_resistance']['support'] = recent_data['Low'].min()
        signals['support_resistance']['resistance'] = recent_data['High'].max()
        
        # Gesamtsignal berechnen
        total_signal = sum(signals['signals'].values())
        
        # Empfehlung basierend auf Gesamtsignal
        if total_signal >= 2:
            recommendation = "STRONG BUY"
        elif total_signal == 1:
            recommendation = "BUY"
        elif total_signal == 0:
            recommendation = "HOLD"
        elif total_signal == -1:
            recommendation = "SELL"
        else:  # total_signal <= -2
            recommendation = "STRONG SELL"
            
        signals['total_signal'] = total_signal
        signals['recommendation'] = recommendation
        
        return signals
    
    def create_plotly_chart(self, df, symbol, signals):
        # Nur die letzten 90 Tage für das Chart verwenden
        df_plot = df.iloc[-min(90, len(df)):].copy()
        
        # Erstelle Subplots
        fig = make_subplots(
            rows=3, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Preis", "MACD", "RSI"),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Hauptchart: Preis mit SMA und Bollinger Bands
        fig.add_trace(
            go.Candlestick(
                x=df_plot.index,
                open=df_plot['Open'],
                high=df_plot['High'],
                low=df_plot['Low'],
                close=df_plot['Close'],
                name="Preis"
            ),
            row=1, col=1
        )
        
        # SMA Linien
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot['SMA20'],
                name="SMA20",
                line=dict(color='red', width=1)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot['SMA50'],
                name="SMA50",
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        # Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot['BB_Upper'],
                name="BB Oberes Band",
                line=dict(color='gray', width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot['BB_Lower'],
                name="BB Unteres Band",
                line=dict(color='gray', width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        # Support und Widerstand
        fig.add_trace(
            go.Scatter(
                x=[df_plot.index[0], df_plot.index[-1]],
                y=[signals['support_resistance']['support'], signals['support_resistance']['support']],
                name="Support",
                line=dict(color='green', width=1)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[df_plot.index[0], df_plot.index[-1]],
                y=[signals['support_resistance']['resistance'], signals['support_resistance']['resistance']],
                name="Widerstand",
                line=dict(color='red', width=1)
            ),
            row=1, col=1
        )
        
        # MACD
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot['MACD'],
                name="MACD",
                line=dict(color='blue', width=1)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot['MACD_Signal'],
                name="Signal Line",
                line=dict(color='red', width=1)
            ),
            row=2, col=1
        )
        
        # MACD Histogram
        colors = ['green' if val >= 0 else 'red' for val in df_plot['MACD_Hist']]
        fig.add_trace(
            go.Bar(
                x=df_plot.index,
                y=df_plot['MACD_Hist'],
                name="Histogram",
                marker_color=colors
            ),
            row=2, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot['RSI'],
                name="RSI",
                line=dict(color='orange', width=1)
            ),
            row=3, col=1
        )
        
        # RSI Linien bei 30 und 70
        fig.add_trace(
            go.Scatter(
                x=[df_plot.index[0], df_plot.index[-1]],
                y=[30, 30],
                name="Überverkauft",
                line=dict(color='green', width=1, dash='dash')
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[df_plot.index[0], df_plot.index[-1]],
                y=[70, 70],
                name="Überkauft",
                line=dict(color='red', width=1, dash='dash')
            ),
            row=3, col=1
        )
        
        # Layout anpassen
        fig.update_layout(
            title=f"{symbol} - Technische Analyse ({datetime.now().strftime('%d.%m.%Y')})",
            xaxis_rangeslider_visible=False,
            height=800,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Y-Achsen anpassen
        fig.update_yaxes(title_text="Preis", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
        
        return fig
    
    def analyze_symbol(self, symbol):
        try:
            # Daten holen
            data = self.fetch_data(symbol)
            if data is None:
                return {
                    'error': f"Keine Daten für {symbol} gefunden",
                    'symbol': symbol
                }
                
            # Indikatoren berechnen
            df = self.calculate_indicators(data)
            
            # Signale generieren
            signals = self.generate_signals(df, symbol)
            
            # Chart erstellen
            fig = self.create_plotly_chart(df, symbol, signals)
            signals['chart'] = fig
            signals['data'] = df
            
            return signals
        except Exception as e:
            st.error(f"Fehler bei der Analyse von {symbol}: {str(e)}")
            return {
                'error': f"Fehler bei der Analyse von {symbol}: {str(e)}",
                'symbol': symbol
            }

# Funktion zum Anzeigen der Signale
def display_signal_indicator(signal_value):
    if signal_value == 1:
        return "🟢 KAUFEN"
    elif signal_value == -1:
        return "🔴 VERKAUFEN"
    else:
        return "⚪ NEUTRAL"

# Funktion zum Anzeigen der Empfehlung
def display_recommendation(recommendation):
    if recommendation == "STRONG BUY":
        return "🟢 STRONG BUY"
    elif recommendation == "BUY":
        return "🟢 BUY"
    elif recommendation == "HOLD":
        return "🟡 HOLD"
    elif recommendation == "SELL":
        return "🔴 SELL"
    elif recommendation == "STRONG SELL":
        return "🔴 STRONG SELL"
    else:
        return recommendation

# Funktion zum Anzeigen der Analyseergebnisse
def display_analysis_results(results):
    if 'error' in results:
        st.error(f"Fehler: {results['error']}")
        return
    
    # Chart anzeigen
    st.plotly_chart(results['chart'], use_container_width=True)
    
    # Zwei Spalten für Indikatoren und Signale
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Technische Indikatoren")
        indicators = results['indicators']
        
        # Tabelle für Indikatoren
        indicator_data = {
            "Indikator": ["Aktueller Preis", "RSI", "SMA20", "SMA50", "MACD", "Support", "Widerstand"],
            "Wert": [
                f"{results['price']:.2f}",
                f"{indicators['rsi']:.2f}",
                f"{indicators['sma20']:.2f}",
                f"{indicators['sma50']:.2f}",
                f"{indicators['macd']:.2f}",
                f"{results['support_resistance']['support']:.2f}",
                f"{results['support_resistance']['resistance']:.2f}"
            ]
        }
        st.table(pd.DataFrame(indicator_data))
    
    with col2:
        st.subheader("Handelssignale")
        signals = results['signals']
        
        # Tabelle für Signale
        signal_data = {
            "Signal": ["SMA Crossover", "MACD", "RSI", "Bollinger Bands"],
            "Empfehlung": [
                display_signal_indicator(signals['sma_crossover']),
                display_signal_indicator(signals['macd']),
                display_signal_indicator(signals['rsi']),
                display_signal_indicator(signals['bollinger'])
            ]
        }
        st.table(pd.DataFrame(signal_data))
    
    # Gesamtempfehlung
    st.subheader("Gesamtempfehlung")
    recommendation_color = "green" if results['recommendation'] in ["BUY", "STRONG BUY"] else ("red" if results['recommendation'] in ["SELL", "STRONG SELL"] else "orange")
    st.markdown(f"<h2 style='color: {recommendation_color};'>{display_recommendation(results['recommendation'])}</h2>", unsafe_allow_html=True)
    
    # Handelsempfehlung
    if results['recommendation'] in ["BUY", "STRONG BUY"]:
        st.markdown(f"**Kaufsignal bei {results['price']:.2f}**. Beobachten Sie den Widerstand bei {results['support_resistance']['resistance']:.2f}.")
        st.markdown(f"Setzen Sie einen Stop-Loss bei {(results['support_resistance']['support'] * 0.98):.2f}.")
    elif results['recommendation'] in ["SELL", "STRONG SELL"]:
        st.markdown(f"**Verkaufssignal bei {results['price']:.2f}**. Der RSI zeigt überkaufte Bedingungen.")
        st.markdown(f"Beobachten Sie den Support bei {results['support_resistance']['support']:.2f}.")
    else:
        st.markdown(f"**Halten bei {results['price']:.2f}**. Keine klaren Kauf- oder Verkaufssignale.")

# Hauptfunktion für die Einzelanalyse
def single_analysis():
    st.header("Einzelanalyse")
    
    # Symbol-Eingabe
    symbol = st.text_input("Symbol eingeben (z.B. AAPL, BTC-USD):", "AAPL")
    
    # Schnellzugriff-Buttons
    st.markdown("**Schnellzugriff:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Apple (AAPL)"):
            symbol = "AAPL"
    with col2:
        if st.button("Tesla (TSLA)"):
            symbol = "TSLA"
    with col3:
        if st.button("NVIDIA (NVDA)"):
            symbol = "NVDA"
    with col4:
        if st.button("Bitcoin (BTC-USD)"):
            symbol = "BTC-USD"
    
    # Analyse-Button
    if st.button("Analysieren", key="single_analyze_button"):
        if symbol:
            with st.spinner(f"Analysiere {symbol}..."):
                analyzer = StockAnalyzer(period=period_options[period], interval=interval_options[interval])
                results = analyzer.analyze_symbol(symbol)
                display_analysis_results(results)
        else:
            st.warning("Bitte geben Sie ein Symbol ein.")

# Hauptfunktion für die Mehrfachanalyse
def multiple_analysis():
    st.header("Mehrfachanalyse")
    
    # Symbole-Eingabe
    if 'symbols' not in st.session_state:
        st.session_state.symbols = "AAPL, MSFT, GOOGL, AMZN, TSLA"
    
    symbols_text = st.text_area("Symbole eingeben (durch Komma getrennt):", st.session_state.symbols, height=100)
    
    # Analyse-Button
    if st.button("Alle analysieren", key="multiple_analyze_button"):
        if symbols_text:
            symbols = [s.strip() for s in symbols_text.split(",") if s.strip()]
            
            if symbols:
                analyzer = StockAnalyzer(period=period_options[period], interval=interval_options[interval])
                
                # Fortschrittsbalken
                progress_bar = st.progress(0)
                
                # Ergebnisse nach Empfehlung gruppieren
                buy_results = []
                sell_results = []
                hold_results = []
                error_results = []
                
                for i, symbol in enumerate(symbols):
                    # Fortschritt aktualisieren
                    progress = (i + 1) / len(symbols)
                    progress_bar.progress(progress)
                    
                    # Status anzeigen
                    st.write(f"Analysiere {symbol}... ({i+1}/{len(symbols)})")
                    
                    # Symbol analysieren
                    results = analyzer.analyze_symbol(symbol)
                    
                    if 'erro
