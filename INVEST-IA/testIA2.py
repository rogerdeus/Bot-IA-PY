import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import pandas as pd
import requests
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score
import optuna
from optuna.pruners import MedianPruner
import os
from datetime import datetime
import ta

# =========================
# Coleta e Features
# =========================
def obter_dados_binance_periodo(symbol, interval, start_str, end_str):
    """Obtém dados históricos da Binance com tratamento de erros."""
    try:
        start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_str).timestamp() * 1000)
        if start_ts >= end_ts:
            raise ValueError("Data inicial deve ser anterior à data final")
    except ValueError as e:
        raise ValueError(f"Erro nas datas: {e}")

    url = 'https://api.binance.com/api/v3/klines'
    df_total = pd.DataFrame()

    while start_ts < end_ts:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_ts,
            'endTime': end_ts,
            'limit': 1000
        }
        try:
            response = requests.get(url, params=params)
            if response.status_code != 200:
                raise Exception(f"Erro na API: {response.text}")
            data = response.json()
            if not data or (isinstance(data, dict) and 'code' in data):
                break
            df = pd.DataFrame(data, columns=[
                'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close_time', 'Quote_asset_volume', 'Number_of_trades',
                'Taker_buy_base_volume', 'Taker_buy_quote_volume', 'Ignore'
            ])
            df_total = pd.concat([df_total, df], ignore_index=True)
            start_ts = int(df['timestamp'].iloc[-1]) + 1
            if len(data) < 1000:
                break
        except requests.RequestException as e:
            raise Exception(f"Erro de conexão com a API: {e}")

    if df_total.empty:
        raise ValueError(f"Nenhum dado retornado para {symbol} no intervalo {interval}")

    df_total['timestamp'] = pd.to_datetime(df_total['timestamp'], unit='ms')
    df_total.set_index('timestamp', inplace=True)
    df_total = df_total[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    return df_total

def preparar_dados_ml(df, price_threshold=0.003):
    """Prepara os dados para o modelo de ML com novos indicadores."""
    df = df.copy()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    macd_indicator = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd_indicator.macd()
    df['MACD_Signal'] = macd_indicator.macd_signal()
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    df['Stoch'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3).stoch()
    
    features = ['RSI', 'MACD', 'MACD_Signal', 'ATR', 'ADX', 'SMA50', 'SMA200', 'Volume', 'BB_Width', 'Stoch']
    df = df.dropna()
    
    df['Return'] = df['Close'].pct_change().shift(-1)
    df['Target'] = (df['Return'] > price_threshold).astype(int)
    df.dropna(inplace=True)

    X = df[features]
    y = df['Target']
    return X, y, df

# =========================
# Treino e Simulação
# =========================
def treinar_modelo_walk_forward(X, y, n_splits=5):
    """Treina um modelo Random Forest usando análise walk-forward."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    precisions, recalls, f1_scores = [], [], []
    models = []
    feature_importances = []

    initial_features = X.columns.tolist()
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced_subsample",
            max_depth=None,
            min_samples_split=2
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
        models.append(model)
        feature_importances.append(model.feature_importances_)

    mean_importances = np.mean(feature_importances, axis=0)
    importance_df = pd.DataFrame({'Feature': initial_features, 'Importance': mean_importances})
    print("Importância das Features:\n", importance_df.sort_values(by='Importance', ascending=False))
    
    selected_features = importance_df[importance_df['Importance'] > 0.05]['Feature'].tolist()
    if selected_features:
        X = X[selected_features]
        print(f"Features selecionadas: {selected_features}")
        models = []
        precisions, recalls, f1_scores = [], [], []
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model = RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced_subsample",
                max_depth=None,
                min_samples_split=2
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            precisions.append(precision_score(y_test, y_pred, zero_division=0))
            recalls.append(recall_score(y_test, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
            models.append(model)

    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1_scores)

    return models[-1], avg_precision, avg_recall, avg_f1, models, tscv, selected_features

def simular_estrategia_ml_walk_forward(df, models, tscv, capital_inicial, risco_por_trade, taxa=0.001, atr_multiplier=2, prob_threshold=0.7, selected_features=None):
    """Simula a estratégia de trading usando os modelos treinados."""
    capital = capital_inicial
    capital_evolucao = [capital]
    timestamps = [df.index[0]]
    posicao = 0
    vitorias = 0
    total = 0
    perdas_consecutivas = 0
    retornos = []

    for fold, (train_index, test_index) in enumerate(tscv.split(df)):
        df_test = df.iloc[test_index]
        # Usar as features selecionadas em vez de feature_names_in_
        if selected_features is not None:
            X_test = df_test[selected_features]
        else:
            X_test = df_test[models[fold].feature_names_in_]  # Fallback para versões recentes
        for i in range(len(df_test)):
            idx = df_test.index[i]
            sma200 = df['SMA200'].loc[idx]
            adx = df['ADX'].loc[idx]
            close_anterior = df['Close'].shift(1).loc[idx]
            candles_em_bear = (df['Close'].shift(1) < df['Close'].shift(2)).rolling(window=5).sum().loc[idx]

            if not np.isfinite(sma200) or not np.isfinite(adx) or not np.isfinite(close_anterior):
                continue

            regime_bull = df['Close'].loc[idx] > sma200 and adx > 25
            regime_bear = df['Close'].loc[idx] < sma200 or candles_em_bear >= 3

            prob = models[fold].predict_proba(X_test.iloc[[i]])[0, 1]

            if posicao == 0 and prob >= prob_threshold and regime_bull and not regime_bear:
                atr = df['ATR'].loc[idx]
                stop_loss = df['Close'].loc[idx] * (1 - atr_multiplier * atr / df['Close'].loc[idx])
                take_profit = df['Close'].loc[idx] * (1 + atr_multiplier * atr / df['Close'].loc[idx])
                risco_trade_ajustado = risco_por_trade * (0.8 ** perdas_consecutivas)
                tamanho_posicao = capital * risco_trade_ajustado / (df['Close'].loc[idx] - stop_loss)
                posicao = tamanho_posicao
                preco_entrada = df['Close'].loc[idx] * (1 + taxa)

            if posicao > 0:
                if df['Low'].loc[idx] <= stop_loss:
                    preco_saida = stop_loss * (1 - taxa)
                    retorno = (preco_saida - preco_entrada) * posicao
                    capital += retorno
                    retornos.append(retorno / capital_inicial)
                    posicao = 0
                    total += 1
                    perdas_consecutivas += 1
                elif df['High'].loc[idx] >= take_profit:
                    preco_saida = take_profit * (1 - taxa)
                    retorno = (preco_saida - preco_entrada) * posicao
                    capital += retorno
                    retornos.append(retorno / capital_inicial)
                    posicao = 0
                    total += 1
                    vitorias += 1
                    perdas_consecutivas = 0

            capital_evolucao.append(capital)
            timestamps.append(idx)

    taxa_acerto = vitorias / total if total > 0 else 0.0
    retornos = np.array(retornos)
    sharpe_ratio = np.mean(retornos) / np.std(retornos) * np.sqrt(365 * 6) if len(retornos) > 0 and np.std(retornos) != 0 else 0.0
    drawdown = np.min(np.cumsum(retornos)) if len(retornos) > 0 else 0.0

    return capital_evolucao, capital, sharpe_ratio, drawdown, taxa_acerto, timestamps, total

def otimizar_com_optuna(df, X, y, capital_inicial):
    """Otimiza os parâmetros usando Optuna."""
    def objective(trial):
        n_splits = trial.suggest_int('n_splits', 3, 10)
        prob_threshold = trial.suggest_float('prob_threshold', 0.5, 0.9)
        atr_multiplier = trial.suggest_float('atr_multiplier', 1.0, 5.0)
        risco_por_trade = trial.suggest_float('risco_por_trade', 0.01, 0.2)

        # Ajustar desempacotação para 7 valores
        model, precision, recall, f1, models, tscv, selected_features = treinar_modelo_walk_forward(X, y, n_splits)
        capital_evolucao, capital, sharpe_ratio, drawdown, taxa_acerto, _, total = simular_estrategia_ml_walk_forward(
            df, models, tscv, capital_inicial, risco_por_trade,
            atr_multiplier=atr_multiplier, prob_threshold=prob_threshold, selected_features=selected_features
        )
        lucro_bot = (capital / capital_inicial - 1) * 100
        score = (lucro_bot / 100) * sharpe_ratio / (1 + abs(drawdown))
        if total < 10:  # Usar total real da simulação
            score *= 0.1
        return score

    study = optuna.create_study(direction='maximize', pruner=MedianPruner())
    study.optimize(objective, n_trials=60)
    return study.best_params

def executar_simulacao():
    """Executa a simulação de trading."""
    text_resultado.delete(1.0, tk.END)
    label_status.config(text="Simulação em andamento...")
    janela.update()

    symbols = [s.strip() for s in entry_symbol.get().split(',')]
    intervals = [i.strip() for i in entry_interval.get().split(',')]
    start = entry_start.get()
    end = entry_end.get()
    try:
        capital_inicial = float(entry_capital.get())
        risco_por_trade = float(entry_risco.get())
        prob_threshold = float(entry_prob.get())
    except ValueError:
        messagebox.showerror("Erro", "Por favor, insira valores numéricos válidos para Capital, Risco e Probabilidade.")
        return

    for symbol in symbols:
        for interval in intervals:
            try:
                df = obter_dados_binance_periodo(symbol, interval, start, end)
                X, y, df_processed = preparar_dados_ml(df, price_threshold=0.003)
                if modo_otimizado.get():
                    best_params = otimizar_com_optuna(df_processed, X, y, capital_inicial)
                    n_splits = best_params['n_splits']
                    prob_threshold = best_params['prob_threshold']
                    atr_multiplier = best_params['atr_multiplier']
                    risco_por_trade = best_params['risco_por_trade']
                else:
                    n_splits = 5

                model, precision, recall, f1, models, tscv, selected_features = treinar_modelo_walk_forward(X, y, n_splits)
                capital_evolucao, capital, sharpe_ratio, drawdown, taxa_acerto, timestamps, total = simular_estrategia_ml_walk_forward(
                    df_processed, models, tscv, capital_inicial, risco_por_trade,
                    prob_threshold=prob_threshold, atr_multiplier=2, selected_features=selected_features
                )
                capital_hold = buy_and_hold(df_processed['Close'], capital_inicial)
                lucro_bot = (capital / capital_inicial - 1) * 100
                lucro_hold = (capital_hold / capital_inicial - 1) * 100

                resultado = f"\n📊 Resultados para {symbol} ({interval}):\n"
                resultado += f"💰 Capital Final do Bot: ${capital:.2f}\n"
                resultado += f"📈 Capital Final do Hold: ${capital_hold:.2f}\n"
                resultado += f"📉 Lucro do Bot: {lucro_bot:.2f}%\n"
                resultado += f"📊 Lucro do Hold: {lucro_hold:.2f}%\n"
                resultado += f"📈 Sharpe Ratio: {sharpe_ratio:.2f}\n"
                resultado += f"📉 Drawdown Máximo: {drawdown * 100:.2f}%\n"
                resultado += f"✅ Taxa de Acerto: {taxa_acerto * 100:.2f}%\n"
                resultado += f"🔍 Precisão Média: {precision * 100:.2f}%\n"
                resultado += f"🔍 Recall Médio: {recall * 100:.2f}%\n"
                resultado += f"🔍 F1-Score Médio: {f1 * 100:.2f}%\n"
                resultado += f"Total de Trades: {total}\n"
                resultado += f"{'-' * 50}\n"
                text_resultado.insert(tk.END, resultado)

            except Exception as e:
                text_resultado.insert(tk.END, f"Erro ao simular {symbol} ({interval}): {str(e)}\n{'-' * 50}\n")

    label_status.config(text="Simulação concluída!")

def buy_and_hold(close_series, capital_inicial):
    """Calcula o capital final de uma estratégia buy-and-hold."""
    return capital_inicial * (close_series.iloc[-1] / close_series.iloc[0]) * (1 - 0.001)

# =========================
# GUI
# =========================
janela = tk.Tk()
janela.title("Simulador de Trading com IA")
janela.geometry("800x700")
janela.configure(bg="#f0f0f0")

fonte_padrao = ("Segoe UI", 10)
janela.option_add("*Font", fonte_padrao)

estilo = ttk.Style()
estilo.configure("TLabel", background="#f0f0f0", font=("Segoe UI", 10))
estilo.configure("TEntry", font=("Segoe UI", 10))
estilo.configure("TButton", font=("Segoe UI", 10, "bold"), padding=5)
estilo.configure("TCheckbutton", background="#f0f0f0", font=("Segoe UI", 10))

frame_config = ttk.LabelFrame(janela, text="Configurações do Ativo")
frame_config.pack(padx=15, pady=10, fill="x")

def add_row(frame, label_text, entry_widget, row):
    ttk.Label(frame, text=label_text).grid(row=row, column=0, sticky='e', padx=5, pady=5)
    entry_widget.grid(row=row, column=1, padx=5, pady=5)

entry_symbol = ttk.Entry(frame_config, width=30)
entry_symbol.insert(0, "BTCUSDT")
add_row(frame_config, "Ativos (ex: BTCUSDT,ETHUSDT):", entry_symbol, 0)

entry_interval = ttk.Entry(frame_config, width=30)
entry_interval.insert(0, "4h")
add_row(frame_config, "Intervalos (ex: 1h,4h):", entry_interval, 1)

entry_start = ttk.Entry(frame_config, width=30)
entry_start.insert(0, "2024-01-01")
add_row(frame_config, "Data Inicial (AAAA-MM-DD):", entry_start, 2)

entry_end = ttk.Entry(frame_config, width=30)
entry_end.insert(0, "2024-12-31")
add_row(frame_config, "Data Final (AAAA-MM-DD):", entry_end, 3)

frame_estrategia = ttk.LabelFrame(janela, text="Parâmetros da Estratégia")
frame_estrategia.pack(padx=15, pady=10, fill="x")

entry_capital = ttk.Entry(frame_estrategia, width=30)
entry_capital.insert(0, "100")
add_row(frame_estrategia, "Capital Inicial:", entry_capital, 0)

entry_risco = ttk.Entry(frame_estrategia, width=30)
entry_risco.insert(0, "0.1")
add_row(frame_estrategia, "Risco por Trade (0.1 = 10%):", entry_risco, 1)

entry_prob = ttk.Entry(frame_estrategia, width=30)
entry_prob.insert(0, "0.7")
add_row(frame_estrategia, "Probabilidade Threshold (0 a 1):", entry_prob, 2)

modo_otimizado = tk.BooleanVar()
check_otimizado = ttk.Checkbutton(frame_estrategia, text="Usar parâmetros otimizados (Optuna)", variable=modo_otimizado)
check_otimizado.grid(row=3, column=0, columnspan=2, pady=10)

frame_execucao = ttk.Frame(janela)
frame_execucao.pack(pady=5)

btn_simular = ttk.Button(frame_execucao, text="▶ Executar Simulação", command=executar_simulacao)
btn_simular.grid(row=0, column=0, padx=10, pady=10)

label_status = ttk.Label(frame_execucao, text="", foreground="#333")
label_status.grid(row=0, column=1)

frame_resultado = ttk.LabelFrame(janela, text="Resultados da Simulação")
frame_resultado.pack(padx=15, pady=10, fill="both", expand=True)

text_resultado = tk.Text(frame_resultado, height=20, wrap="word", font=("Consolas", 10))
text_resultado.pack(fill="both", expand=True, padx=5, pady=5)

janela.mainloop()