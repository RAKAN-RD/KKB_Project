from flask import Flask, render_template
import yfinance as yf
import plotly.graph_objects as go
import plotly.io as pio
import pickle
import numpy as np
from datetime import datetime, timedelta
import pandas as pd


class MLR:
  def __init__(self, data):
      X = [list(i[:-1]) for i in data]
      for xi in X:
        xi.insert(0, 1)
      X = np.array(X)
      Y = np.array([[i[-1]] for i in data])
      y_bar = sum(Y)/len(Y)
      XT = X.T
      XTX = np.dot(XT, X)
      XTX_inv = np.linalg.inv(XTX)
      XTY = np.dot(XT, Y)
      self.B = np.dot(XTX_inv, XTY)
      self.SST = sum([(Y[i][0] - y_bar)**2 for i in range(len(X))])
      y_pred = []
      for i in range(len(X)):
        t_pred = self.B[0][0]
        for b in range(1, len(self.B)):
          px = X[i][b-1]*self.B[b][0]
          t_pred += px
        y_pred.append(t_pred)
      self.SSE = sum([(Y[i][0] - y_pred[i])**2 for i in range(len(X))])
      self.R2 = 1 - (self.SSE/self.SST)

  def predict(self, X):
    prediction = []
    for i in range(len(X)):
      t_pred = self.B[0][0]
      for b in range(1, len(self.B)):
        px = X[i][b-1]*self.B[b][0]
        t_pred += px
      prediction.append(t_pred)
    return prediction


app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/about')
def detail_project():
    return render_template('detail_project.html')

@app.route('/kkb')
def main_menu():
    return render_template('main_menu.html')

@app.route('/detail/<string:ticker>')
def detail(ticker):
    # Ambil data saham menggunakan yfinance
    stock = yf.Ticker(ticker)
    
    # Mendapatkan informasi perusahaan
    info = stock.info  # Ini mengembalikan dictionary dengan data perusahaan

    # Ambil beberapa data yang diperlukan, misalnya:
    company_name = info.get('longName', 'Nama perusahaan tidak tersedia')
    sector = info.get('sector', 'Sektor tidak tersedia')
    industry = info.get('industry', 'Industri tidak tersedia')
    market_cap = info.get('marketCap', 'Market cap tidak tersedia')
    previous_close = info.get('previousClose', 'Harga penutupan sebelumnya tidak tersedia')

    # Kirim data ke template untuk ditampilkan
    return render_template('detail_perusahaan.html', ticker=ticker, company_name=company_name,
                           sector=sector, industry=industry, market_cap=market_cap, previous_close=previous_close)


@app.route('/keuangan/<string:ticker>')
def fundamental(ticker):
    saham = yf.Ticker(ticker)
    data_harga_saham = saham.history(period='3mo')
    data_neraca = saham.quarterly_balance_sheet
    data_laba = saham.quarterly_income_stmt
    data_arus = saham.quarterly_cash_flow
    
    # Mengambil data keuangan
    saham_beredar = data_neraca.loc['Ordinary Shares Number'].iloc[0]
    laba_bersih = data_laba.loc['Net Income'].iloc[0]
    aset = data_neraca.loc['Total Assets'].iloc[0]
    liabilitas = data_neraca.loc['Total Liabilities Net Minority Interest'].iloc[0]
    ekuitas = data_neraca.loc['Total Equity Gross Minority Interest'].iloc[0]
    revenue = data_laba.loc['Operating Revenue'].iloc[0]
    cash = data_neraca.loc['Cash And Cash Equivalents'].iloc[0]
    capex = data_arus.loc['Capital Expenditure'].iloc[0]
    
    # Menghitung perubahan (diff) dalam persen
    diff_laba_bersih = round(((laba_bersih - data_laba.loc['Net Income'].iloc[1]) / data_laba.loc['Net Income'].iloc[1])*100, 2)
    diff_aset = round(((aset - data_neraca.loc['Total Assets'].iloc[1]) / data_neraca.loc['Total Assets'].iloc[1])*100, 2)
    diff_liabilitas = round(((liabilitas - data_neraca.loc['Total Liabilities Net Minority Interest'].iloc[1]) / data_neraca.loc['Total Liabilities Net Minority Interest'].iloc[1])*100, 2)
    diff_ekuitas = round(((ekuitas - data_neraca.loc['Total Equity Gross Minority Interest'].iloc[1]) / data_neraca.loc['Total Equity Gross Minority Interest'].iloc[1])*100, 2)
    diff_revenue = round(((revenue - data_laba.loc['Operating Revenue'].iloc[1]) / data_laba.loc['Operating Revenue'].iloc[1])*100, 2)
    diff_cash = round(((cash - data_neraca.loc['Cash And Cash Equivalents'].iloc[1]) / data_neraca.loc['Cash And Cash Equivalents'].iloc[1])*100, 2)
    diff_capex = round(((capex - data_arus.loc['Capital Expenditure'].iloc[1]) / data_arus.loc['Capital Expenditure'].iloc[1])*100, 2)

    # Membuat line chart harga saham
    dates = data_harga_saham.index.strftime('%Y-%m-%d').tolist()
    closing_prices = data_harga_saham['Close'].tolist()

    # Membuat chart interaktif menggunakan Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=closing_prices, mode='lines', name='Harga Penutupan'))
    fig.update_layout(
        title=f'Harga Saham {ticker} (3 Bulan Terakhir)',
        xaxis_title='Tanggal',
        yaxis_title='Harga (IDR)',
        template='plotly_white'
    )

    # Konversi chart ke HTML
    chart_html = pio.to_html(fig, full_html=False)

    # Mengirimkan data ke template
    return render_template('keuangan.html', ticker=ticker, data_harga_saham=data_harga_saham,
                           laba_bersih=laba_bersih, aset=aset, liabilitas=liabilitas,
                           ekuitas=ekuitas, revenue=revenue, cash=cash, capex=capex,
                           diff_laba_bersih=diff_laba_bersih, diff_aset=diff_aset,
                           diff_liabilitas=diff_liabilitas, diff_ekuitas=diff_ekuitas,
                           diff_revenue=diff_revenue, diff_cash=diff_cash, diff_capex=diff_capex,
                           chart=chart_html, saham_beredar=saham_beredar)

@app.route('/prediksi/<string:ticker>')
def prediksi(ticker):

    my_model = pickle.load(open('./model/my_MLR.pickle', 'rb'))

    saham = yf.Ticker(ticker)
    daftar_harga = saham.history(start='2024-07-01', end='2024-11-01', interval='1mo')
    daftar_harga.reset_index(inplace=True)
    q_harga = []
    for i in range(0, len(daftar_harga), 3):
        q_harga.append(daftar_harga['Close'][i])
    q_harga.reverse()

    neraca = saham.quarterly_balance_sheet
    income = saham.quarterly_income_stmt

    net_income = income.loc['Net Income'].iloc[0:2]
    saham_beredar = neraca.loc['Ordinary Shares Number'].iloc[0:2]
    ekuitas = neraca.loc['Total Equity Gross Minority Interest'].iloc[:2]
    aset = neraca.loc['Total Assets'].iloc[:2]
    eps = net_income/saham_beredar
    bpvs = ekuitas/saham_beredar
    pb = q_harga/bpvs
    roa = net_income/aset

    diff_net_income = (net_income.iloc[0] - net_income.iloc[1])/net_income.iloc[1]
    diff_eps = (eps.iloc[0] - eps.iloc[1])/eps.iloc[1]
    diff_pb = (pb.iloc[0] - pb.iloc[1])/pb.iloc[1]
    diff_roa = (roa.iloc[0] - roa.iloc[1])/roa.iloc[1]

    q_prediction = my_model.predict([[diff_net_income, diff_eps, diff_pb, diff_roa]])


    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)
    harga_saham = saham.history(start=start_date, end=end_date)
    harga_saham['SMA_10'] = harga_saham['Close'].rolling(window=10).mean()
    harga_saham['SMA_50'] = harga_saham['Close'].rolling(window=50).mean()
    delta = harga_saham['Close'].diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    harga_saham['RSI'] = 100 - (100 / (1 + rs))
    harga_saham['SMA_20'] = harga_saham['Close'].rolling(window=20).mean()
    harga_saham['BB_Upper'] = harga_saham['SMA_20'] + 2 * harga_saham['Close'].rolling(window=20).std()
    harga_saham['BB_Lower'] = harga_saham['SMA_20'] - 2 * harga_saham['Close'].rolling(window=20).std()
    high_low = harga_saham['High'] - harga_saham['Low']
    high_close_prev = np.abs(harga_saham['High'] - harga_saham['Close'].shift(1))
    low_close_prev = np.abs(harga_saham['Low'] - harga_saham['Close'].shift(1))
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    harga_saham['ATR'] = true_range.rolling(window=14).mean()
    harga_saham['Log_Return'] = np.log(harga_saham['Close'] / harga_saham['Close'].shift(1))
    
    model_teknikal = pickle.load(open('./model/weekly_model.pickle', 'rb'))

    data_harga = harga_saham[['Close']].iloc[:]
    hasil_prediksi = model_teknikal.predict([harga_saham.iloc[-1][['SMA_10', 'SMA_50', 'Volume', 'RSI', 'BB_Upper', 'BB_Lower', 'ATR', 'Log_Return']].to_list()])
    selisih = (float(hasil_prediksi[0]) - harga_saham['Close'].iloc[-1])/5
    l_pred = []
    for s in range(1, 6):
      l_pred.append(harga_saham['Close'].iloc[-1] + (s*selisih))
    prediksi_tanggal = [end_date + timedelta(days=i) for i in range(1, 6)]
    data_prediksi = pd.DataFrame({
      'Date':prediksi_tanggal,
      'Close': l_pred
    })

    data_prediksi['Date'] = pd.to_datetime(data_prediksi['Date'])  # Pastikan format datetime
    data_prediksi.set_index('Date', inplace=True)  # Set kolom 'Date' sebagai indeks
    data_harga.index = data_harga.index.tz_localize(None)
    data_final = pd.concat([data_harga, data_prediksi])

    split_date = data_harga.index[-1]  # Tanggal terakhir dari data historis

    fig = go.Figure()

    # Tambahkan garis data historis
    fig.add_trace(go.Scatter(
        x=data_harga.index, 
        y=data_harga['Close'], 
        mode='lines', 
        name='Data Historis'
    ))

    # Tambahkan garis data prediksi
    fig.add_trace(go.Scatter(
        x=data_prediksi.index, 
        y=data_prediksi['Close'], 
        mode='lines', 
        name='Data Prediksi',
        line=dict(dash='dot', color='orange')  # Gaya garis putus-putus untuk prediksi
    ))

    # Tambahkan garis pemisah
    fig.add_shape(
        type='line',
        x0=split_date,
        y0=data_final['Close'].min(),  # Titik Y terendah
        x1=split_date,
        y1=data_final['Close'].max(),  # Titik Y tertinggi
        line=dict(color='red', dash='dash'),  # Gaya garis pemisah
        xref='x',
        yref='y'
    )

    fig.update_layout(
        title=f'Harga Saham {ticker} (3 Bulan Terakhir)',
        xaxis_title='Tanggal',
        yaxis_title='Harga (IDR)',
        template='plotly_white'
    )

    # Konversi chart ke HTML
    chart_html = pio.to_html(fig, full_html=False)

    return render_template('prediksi.html', ticker=ticker, q_prediction=round(float(q_prediction[0]), 2), chart_pred =chart_html)




if __name__ == '__main__':
    app.run(debug=True)
