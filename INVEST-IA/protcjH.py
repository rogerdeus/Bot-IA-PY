import tkinter as tk
from tkinter import messagebox
import pandas as pd
import requests
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score
import itertools
import optuna

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
            if isinstance(data, dict) and 'code' in data:
                raise Exception(f"Erro da Binance: {data['msg']}")
            if not data:
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


def calcular_rsi(df, period=14):
    """Calcula o RSI."""
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calcular_macd(df, fast=12, slow=26, signal=9):
    """Calcula o MACD e a linha de sinal."""
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def calcular_atr(df, period=14):
    """Calcula o ATR."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calcular_adx(df, period=14):
    """Calcula o ADX."""
    plus_dm = df['High'].diff()
    minus_dm = -df['Low'].diff()
    plus_di = 100 * (plus_dm.clip(lower=0).rolling(window=period).mean() / df['ATR'])
    minus_di = 100 * (minus_dm.clip(lower=0).rolling(window=period).mean() / df['ATR'])
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    return adx


def preparar_dados_ml(df, price_threshold=0.0015):
    """Prepara os dados para o modelo de ML."""
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = calcular_rsi(df)
    df['MACD'], df['MACD_Signal'] = calcular_macd(df)
    df['ATR'] = calcular_atr(df)
    df['ADX'] = calcular_adx(df)

    # Criar target: 1 se o próximo candle subir mais que price_threshold, 0 caso contrário
    df['Return'] = df['Close'].pct_change().shift(-1)
    df['Target'] = (df['Return'] > price_threshold).astype(int)
    df.dropna(inplace=True)

    features = ['RSI', 'MACD', 'MACD_Signal', 'ATR', 'ADX', 'SMA50', 'SMA200', 'Volume']
    X = df[features]
    y = df['Target']
    return X, y, df

# =========================
# Treino e Simulação
# =========================
def treinar_modelo_walk_forward(X, y, n_splits=5, test_size=0.2):
    """Treina um modelo Random Forest usando análise walk-forward."""
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=int(len(X) * test_size))
    precisions, recalls, f1_scores = [], [], []
    models = []

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

    # Retorna o último modelo treinado e métricas médias
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1_scores)

    return models[-1], avg_precision, avg_recall, avg_f1, models, tscv


def simular_estrategia_ml_walk_forward(df, models, tscv, capital_inicial, risco_por_trade, taxa=0.001, atr_multiplier=2,
                                       prob_threshold=0.7):
    """Simula a estratégia usando modelos walk-forward com gestão por regime e custos realistas.
       Retorna também n_trades e time_in_market para otimização."""
    capital = capital_inicial
    perdas_consecutivas = 0
    capital_evolucao = []
    trades = []
    timestamps = []
    candles_em_bear = 0
    features = ['RSI', 'MACD', 'MACD_Signal', 'ATR', 'ADX', 'SMA50', 'SMA200', 'Volume']

    spread_pct = 0.0003  # custo realista ~0,03% por lado

    # métricas de exposição
    total_bars = 0
    bars_in_pos = 0
    posicao = 0.0
    for i, (train_index, test_index) in enumerate(tscv.split(df[features])):
        model = models[i]
        df_test = df.iloc[test_index].copy()

        entrada = 0.0
        maior_preco = 0.0
        stop_fixo = 0.0
        tp_hit = False
        target_price = 0.0

        for j in range(1, len(df_test)):
            row = df_test.iloc[j]
            timestamps.append(df_test.index[j])
            total_bars += 1
            if posicao > 0:
                bars_in_pos += 1

            # --- DETECÇÃO DE REGIME (mais inclusivo em bull) ---
            j_back = max(0, j - 5)
            sma200_slope = float(row['SMA200'] - df_test['SMA200'].iloc[j_back])

            bull_regime = (row['Close'] > row['SMA200']) and (sma200_slope >= 0) and (row['ADX'] >= 16)
            bear_regime = (row['Close'] < row['SMA200']) and (sma200_slope < 0) and (row['ADX'] >= 18)

            if bear_regime:
                candles_em_bear += 1
            else:
                candles_em_bear = max(0, candles_em_bear - 1)
            bear_prolongado = candles_em_bear >= 3

            # saída de segurança
            if (bear_regime or bear_prolongado) and posicao > 0:
                preco_atual = float(row['Close'])
                slippage = taxa + spread_pct
                capital += posicao * preco_atual * (1 - slippage)
                trades.append({'type': 'sell', 'price': preco_atual, 'time': df_test.index[j], 'reason': 'bear_exit'})
                posicao = 0.0
                tp_hit = False
                target_price = 0.0

            # em bear não entra
            if bear_regime or bear_prolongado:
                capital_evolucao.append(capital + (posicao * row['Close'] if posicao > 0 else 0))
                continue

            # --- PREVISÃO ---
            X_row = df_test[features].iloc[j:j + 1]
            prob = model.predict_proba(X_row)[0][1] if 1 in model.classes_ else 0.0

            threshold_base = prob_threshold
            risco_trade_ajustado = risco_por_trade * (0.8 ** perdas_consecutivas)

            # modos por regime: em bull entrar mais
            if bull_regime:
                threshold_dinamico = max(threshold_base - 0.08, 0.5)
                atr_mult_eff = max(1.0, float(atr_multiplier) * 0.9)
            else:
                threshold_dinamico = threshold_base
                atr_mult_eff = float(atr_multiplier)

            # ENTRADA
            if posicao == 0 and prob > threshold_dinamico:
                permitir_entrada = bull_regime or (row['ADX'] >= 20 and row['SMA50'] >= row['SMA200'])
                if not permitir_entrada:
                    capital_evolucao.append(capital)
                    continue

                entrada = float(row['Close'])
                base_stop = float(row['ATR']) * max(1.0, atr_mult_eff * 0.8)
                stop_fixo = entrada - base_stop
                stop_trailing = entrada - (float(row['ATR']) * atr_mult_eff)
                stop_total = max(stop_fixo, stop_trailing)

                # em bull: dá mais fôlego (2,5% abaixo da SMA50)
                if bull_regime:
                    stop_total = max(stop_total, float(row['SMA50']) * 0.975)

                risco_por_unit = entrada - stop_total
                if (not np.isfinite(risco_por_unit)) or risco_por_unit <= 0:
                    capital_evolucao.append(capital)
                    continue

                perda_maxima = capital * risco_trade_ajustado * prob
                edge = max(prob - threshold_dinamico, 0.0)
                perda_maxima *= (1.0 + 0.5 * edge)

                slippage = taxa + spread_pct
                qtd = perda_maxima / risco_por_unit
                qtd = min(qtd, (capital * 0.95) / (entrada * (1 + slippage)))
                if (not np.isfinite(qtd)) or qtd <= 0:
                    capital_evolucao.append(capital)
                    continue

                capital -= qtd * entrada * (1 + slippage)
                posicao = qtd
                maior_preco = entrada
                trades.append({'type': 'buy', 'price': entrada, 'time': df_test.index[j]})

                tp_hit = False
                # ainda deixamos sem parcial em bull pra “deixar correr”
                if not bull_regime:
                    target_price = entrada + 1.5 * (entrada - stop_total)
                else:
                    target_price = 0.0

            # GESTÃO / SAÍDA
            elif posicao > 0:
                preco_atual = float(row['Close'])
                maior_preco = max(maior_preco, preco_atual)

                if (not bull_regime) and (not tp_hit) and target_price > 0 and preco_atual >= target_price:
                    slippage = taxa + spread_pct
                    qtd_vender = posicao * 0.5
                    capital += qtd_vender * preco_atual * (1 - slippage)
                    posicao -= qtd_vender
                    trades.append({'type': 'sell', 'price': preco_atual, 'time': df_test.index[j], 'reason': 'partial_tp'})
                    tp_hit = True
                    stop_fixo = max(stop_fixo, entrada)

                if bull_regime:
                    atr_mult_eff = max(1.0, float(atr_multiplier) * 0.9)
                else:
                    atr_mult_eff = float(atr_multiplier)

                stop_trailing = maior_preco - (float(row['ATR']) * atr_mult_eff)
                stop_total = max(stop_fixo, stop_trailing)

                if bull_regime:
                    stop_total = max(stop_total, float(row['SMA50']) * 0.975)

                slippage = taxa + spread_pct
                if preco_atual < stop_total:
                    capital += posicao * preco_atual * (1 - slippage)
                    trades.append({'type': 'sell', 'price': preco_atual, 'time': df_test.index[j]})

                    if preco_atual < entrada:
                        perdas_consecutivas += 1
                    else:
                        perdas_consecutivas = 0
                    posicao = 0.0
                    tp_hit = False
                    target_price = 0.0

            capital_evolucao.append(capital + (posicao * row['Close'] if posicao > 0 else 0))

    if posicao > 0:
        slippage = taxa + spread_pct
        capital += posicao * df.iloc[-1]['Close'] * (1 - slippage)
        trades.append({'type': 'sell', 'price': df.iloc[-1]['Close'], 'time': df.index[-1]})
        timestamps.append(df.index[-1])

    # métricas
    retornos = pd.Series(capital_evolucao).pct_change().dropna()
    sharpe_ratio = (retornos.mean() / retornos.std() * (252 ** 0.5)) if retornos.std() != 0 else 0.0
    drawdown = (pd.Series(capital_evolucao) / pd.Series(capital_evolucao).cummax() - 1).min()

    # taxa de acerto e nº de trades (pares buy→sell)
    wins, total = 0, 0
    for k in range(0, len(trades) - 1, 2):
        if trades[k]['type'] == 'buy' and trades[k + 1]['type'] == 'sell':
            total += 1
            if trades[k + 1]['price'] > trades[k]['price']:
                wins += 1
    taxa_acerto = (wins / total) if total > 0 else 0.0
    n_trades = total

    time_in_market = (bars_in_pos / total_bars) if total_bars > 0 else 0.0

    return capital_evolucao, capital, sharpe_ratio, drawdown, taxa_acerto, timestamps, n_trades, time_in_market



def buy_and_hold(df, capital_inicial=100.0, taxa=0.001):
    """Calcula o desempenho de uma estratégia buy and hold."""
    preco_entrada = df['Close'].iloc[0]
    preco_saida = df['Close'].iloc[-1]
    capital = capital_inicial * (preco_saida / preco_entrada) * (1 - taxa)
    return capital


def validar_entradas(symbols, intervals, start, end, capital, risco, prob_threshold):
    """Valida as entradas do usuário."""
    try:
        pd.Timestamp(start)
        pd.Timestamp(end)
        capital = float(capital)
        risco = float(risco)
        prob_threshold = float(prob_threshold)
        if capital <= 0 or risco <= 0 or risco > 1:
            return False, "Capital e risco devem ser números positivos, e risco deve ser <= 1."
        if prob_threshold < 0 or prob_threshold > 1:
            return False, "Probabilidade threshold deve estar entre 0 e 1."
        if not symbols or not intervals:
            return False, "Símbolo(s) e intervalo(s) não podem estar vazios."
        return True, ""
    except ValueError:
        return False, "Datas inválidas ou valores numéricos incorretos. Use o formato AAAA-MM-DD para datas."

# =========================
# Optuna
# =========================
def otimizar_com_optuna(df, X, y, capital_inicial):
    """Optuna focado em retorno com DD controlado, mínimo de trades e exposição mínima."""
    from optuna.pruners import MedianPruner

    TARGET_TRADES = 40      # alvo de trades
    MIN_TRADES = 15         # mínimo aceitável
    MIN_TIM = 0.15          # time in market mínimo (15%)

    def objective(trial):
        # >>> estas 4 linhas substituem os "∈ [...]" que estavam inválidos
        n_splits = trial.suggest_int("n_splits", 3, 5)
        prob_threshold = trial.suggest_float("prob_threshold", 0.60, 0.72)
        atr_multiplier = trial.suggest_float("atr_multiplier", 1.20, 1.80)
        risco_por_trade = trial.suggest_float("risco_por_trade", 0.02, 0.04)

        try:
            _, _, _, _, models, tscv = treinar_modelo_walk_forward(X, y, n_splits=n_splits)
            _, capital_final, sharpe, drawdown, _, _, n_trades, tim = simular_estrategia_ml_walk_forward(
                df, models, tscv, capital_inicial, risco_por_trade,
                prob_threshold=prob_threshold, atr_multiplier=atr_multiplier
            )
            ret = (capital_final / capital_inicial - 1.0)
            dd_pen = max(0.0, -float(drawdown))  # profundidade de DD

            # filtros duros
            if n_trades < MIN_TRADES or tim < MIN_TIM:
                return -9999
            if dd_pen > 0.35 or ret < -0.03:
                return -9999

            # bônus por operar o suficiente
            trades_bonus = min(n_trades / TARGET_TRADES, 1.0)
            tim_bonus = min(tim / 0.35, 1.0)  # expõe até ~35% do tempo

            # score: retorno manda, DD penaliza, Sharpe ajuda, trades/tim estabilizam
            score = (ret * 0.60) + (max(sharpe, 0.0) * 0.10) - (dd_pen * 0.25) + (0.05 * trades_bonus) + (0.10 * tim_bonus)
            return float(score)
        except Exception:
            return -9999

    study = optuna.create_study(direction="maximize", pruner=MedianPruner(n_startup_trials=5))
    study.optimize(objective, n_trials=60, show_progress_bar=False)
    return study.best_params



# =========================
# Execução GUI
# =========================
def executar_simulacao():
    """Executa a simulação com ML para múltiplos ativos e intervalos, com suporte ao Optuna."""
    symbols = entry_symbol.get().upper().replace(" ", "").split(",")
    intervals = entry_interval.get().lower().replace(" ", "").split(",")
    start = entry_start.get()
    end = entry_end.get()
    capital_inicial = entry_capital.get()
    risco = entry_risco.get()
    prob_threshold_input = entry_prob.get()

    valid, erro = validar_entradas(symbols, intervals, start, end, capital_inicial, risco, prob_threshold_input)
    if not valid:
        messagebox.showerror("Erro de Entrada", erro)
        return

    capital_inicial = float(capital_inicial)
    risco = float(risco)
    prob_threshold_input = float(prob_threshold_input)
    resultado = ""
    chart_data = []

    # Parâmetros do grid search se Optuna não for usado
    param_grid = {
        'n_splits': [3, 4],
        'prob_threshold': [0.6, 0.7, 0.8],
        'atr_multiplier': [1.5, 2, 2.5]
    }
    param_combinations = list(itertools.product(param_grid['n_splits'], param_grid['prob_threshold'], param_grid['atr_multiplier']))

    for symbol in symbols:
        for interval in intervals:
            try:
                df = obter_dados_binance_periodo(symbol, interval, start, end)
                X, y, df = preparar_dados_ml(df)

                usar_optuna = False
                best_optuna_params = {}

                if modo_otimizado.get():
                    try:
                        resultado += f"⚙️ Otimizando parâmetros com Optuna...\n"
                        best_optuna_params = otimizar_com_optuna(df, X, y, capital_inicial)
                        usar_optuna = True
                        resultado += f"✅ Parâmetros encontrados: {best_optuna_params}\n"
                    except Exception as e:
                        messagebox.showerror("Erro", f"Falha ao executar Optuna: {e}")
                        return

                if usar_optuna:
                    try:
                        n_splits = best_optuna_params['n_splits']
                        prob_threshold = best_optuna_params['prob_threshold']
                        atr_multiplier = best_optuna_params['atr_multiplier']
                        risco_por_trade = best_optuna_params['risco_por_trade']

                        _, precision, recall, f1, models, tscv = treinar_modelo_walk_forward(X, y, n_splits=n_splits)
                        curva_bot, capital_final_bot, sharpe, drawdown, taxa_acerto, timestamps, n_trades, time_in_market = simular_estrategia_ml_walk_forward(
                            df, models, tscv, capital_inicial, risco_por_trade, prob_threshold=prob_threshold,
                            atr_multiplier=atr_multiplier
                        )

                        capital_hold = buy_and_hold(df, capital_inicial)
                        resultado += (
                            f"\n📊 Resultados com Optuna para {symbol} ({interval}):\n"
                            f"💰 Capital Final do Bot: ${capital_final_bot:.2f}\n"
                            f"📈 Capital Final do Hold: ${capital_hold:.2f}\n"
                            f"📉 Lucro do Bot: {((capital_final_bot / capital_inicial - 1) * 100):.2f}%\n"
                            f"📊 Lucro do Hold: {((capital_hold / capital_inicial - 1) * 100):.2f}%\n"
                            f"📈 Sharpe Ratio: {sharpe:.2f}\n"
                            f"📉 Drawdown Máximo: {drawdown * 100:.2f}%\n"
                            f"✅ Taxa de Acerto: {taxa_acerto * 100:.2f}%\n"
                            f"🔍 Precisão Média: {precision * 100:.2f}%\n"
                            f"🔍 Recall Médio: {recall * 100:.2f}%\n"
                            f"🔍 F1-Score Médio: {f1 * 100:.2f}%\n"
                            f"{'-' * 50}\n"
                            )

                        chart_data.append({
                            'symbol': symbol,
                            'interval': interval,
                            'labels': [str(t)[:10] for t in timestamps],
                            'bot_capital': curva_bot,
                            'hold_capital': [capital_inicial * (df['Close'].iloc[i] / df['Close'].iloc[0]) * (1 - 0.001) for i in range(len(timestamps))]
                        })

                    except Exception as e:
                        resultado += f"Erro ao simular com parâmetros do Optuna para {symbol} ({interval}): {e}\n{'-' * 50}\n"
                    continue

                # Caso não esteja usando Optuna → grid search
                best_sharpe = -float('inf')
                best_params = None
                best_result = None
                best_timestamps = None
                best_curva_bot = None

                for n_splits, prob_threshold, atr_multiplier in param_combinations:
                    try:
                        _, precision, recall, f1, models, tscv = treinar_modelo_walk_forward(X, y, n_splits=n_splits)
                        curva_bot, capital_final_bot, sharpe, drawdown, taxa_acerto, timestamps, n_trades, time_in_market = simular_estrategia_ml_walk_forward(
                            df, models, tscv, capital_inicial, risco, prob_threshold=prob_threshold,
                            atr_multiplier=atr_multiplier
                        )

                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_params = {'n_splits': n_splits, 'prob_threshold': prob_threshold,
                                           'atr_multiplier': atr_multiplier}
                            best_result = (capital_final_bot, precision, recall, f1, sharpe, drawdown, taxa_acerto)
                            best_timestamps = timestamps
                            best_curva_bot = curva_bot

                    except Exception as e:
                        resultado += f"Erro ao simular {symbol} ({interval}) com params {n_splits, prob_threshold, atr_multiplier}: {str(e)}\n"

                if best_params:
                    capital_final_bot, precision, recall, f1, sharpe, drawdown, taxa_acerto = best_result
                    capital_hold = buy_and_hold(df, capital_inicial)
                    resultado += (
                        f"\n📊 Resultados para {symbol} ({interval}) - Melhores parâmetros: {best_params}\n"
                        f"💰 Capital Final do Bot: ${capital_final_bot:.2f}\n"
                        f"📈 Capital Final do Hold: ${capital_hold:.2f}\n"
                        f"📉 Lucro do Bot: {((capital_final_bot / capital_inicial - 1) * 100):.2f}%\n"
                        f"📊 Lucro do Hold: {((capital_hold / capital_inicial - 1) * 100):.2f}%\n"
                        f"📈 Sharpe Ratio: {sharpe:.2f}\n"
                        f"📉 Drawdown Máximo: {drawdown * 100:.2f}%\n"
                        f"✅ Taxa de Acerto: {taxa_acerto * 100:.2f}%\n"
                        f"🔍 Precisão Média (Walk-Forward): {precision * 100:.2f}%\n"
                        f"🔍 Recall Médio (Walk-Forward): {recall * 100:.2f}%\n"
                        f"🔍 F1-Score Médio (Walk-Forward): {f1 * 100:.2f}%\n"
                        f"{'-' * 50}\n"
                    )
                    chart_data.append({
                        'symbol': symbol,
                        'interval': interval,
                        'labels': [str(t)[:10] for t in best_timestamps],
                        'bot_capital': best_curva_bot,
                        'hold_capital': [capital_inicial * (df['Close'].iloc[i] / df['Close'].iloc[0]) * (1 - 0.001) for i in range(len(best_timestamps))]
                    })
                else:
                    resultado += f"Nenhum resultado válido para {symbol} ({interval})\n{'-' * 50}\n"

            except Exception as e:
                resultado += f"Erro ao simular {symbol} ({interval}): {str(e)}\n{'-' * 50}\n"

    # Gráficos Chart.js
    for data in chart_data:
        resultado += f"\nGráfico para {data['symbol']} ({data['interval']}):\n"
        resultado += "```chartjs\n"
        resultado += str({
            "type": "line",
            "data": {
                "labels": data['labels'],
                "datasets": [
                    {
                        "label": f"Capital do Bot ({data['symbol']})",
                        "data": data['bot_capital'],
                        "borderColor": "#4CAF50",
                        "backgroundColor": "rgba(76, 175, 80, 0.1)",
                        "fill": False
                    },
                    {
                        "label": f"Buy and Hold ({data['symbol']})",
                        "data": data['hold_capital'],
                        "borderColor": "#2196F3",
                        "backgroundColor": "rgba(33, 150, 243, 0.1)",
                        "fill": False
                    }
                ]
            },
            "options": {
                "scales": {
                    "x": {"title": {"display": True, "text": "Data"}},
                    "y": {"title": {"display": True, "text": "Capital ($)"}}
                },
                "plugins": {
                    "title": {"display": True, "text": f"Curva de Capital - {data['symbol']} ({data['interval']})"}}
            }
        }) + "\n```"

    text_resultado.delete("1.0", tk.END)
    text_resultado.insert(tk.END, resultado)

# =========================
# GUI
# =========================
janela = tk.Tk()
janela.title("Simulador de Trading com IA")
janela.geometry("800x700")
janela.configure(bg="#f0f0f0")

# Fonte padrão
fonte_padrao = ("Segoe UI", 10)
janela.option_add("*Font", fonte_padrao)

from tkinter import ttk
estilo = ttk.Style()
estilo.configure("TLabel", background="#f0f0f0", font=("Segoe UI", 10))
estilo.configure("TEntry", font=("Segoe UI", 10))
estilo.configure("TButton", font=("Segoe UI", 10, "bold"), padding=5)
estilo.configure("TCheckbutton", background="#f0f0f0", font=("Segoe UI", 10))

# Frame 1 – Configurações
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

# Frame 2 – Estratégia
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

# Frame 3 – Botão e status
frame_execucao = ttk.Frame(janela)
frame_execucao.pack(pady=5)

btn_simular = ttk.Button(frame_execucao, text="▶ Executar Simulação", command=executar_simulacao)
btn_simular.grid(row=0, column=0, padx=10, pady=10)

label_status = ttk.Label(frame_execucao, text="", foreground="#333")
label_status.grid(row=0, column=1)

# Frame 4 – Resultados
frame_resultado = ttk.LabelFrame(janela, text="Resultados da Simulação")
frame_resultado.pack(padx=15, pady=10, fill="both", expand=True)

text_resultado = tk.Text(frame_resultado, height=20, wrap="word", font=("Consolas", 10))
text_resultado.pack(fill="both", expand=True, padx=5, pady=5)

# Rodar o loop da interface
janela.mainloop()
