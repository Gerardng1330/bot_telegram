import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from telegram import Bot
import asyncio
from scipy import stats
from dotenv import load_dotenv
import os
warnings.filterwarnings('ignore')

class AdvancedCryptoAnalyzer:
    def __init__(self, telegram_token=None, chat_id=None):
        self.symbols = ['ETHUSDT', 'SOLUSDT', 'LINKUSDT', 'BNBUSDT']
        self.base_url = 'https://api.binance.com/api/v3'
        self.telegram_token = telegram_token
        self.chat_id = chat_id
        self.bot = Bot(token=telegram_token) if telegram_token else None
        self.telegram_messages = []
        
    def get_klines(self, symbol, interval='1h', limit=500):
        """Obtiene datos históricos de Binance"""
        try:
            url = f"{self.base_url}/klines"
            params = {'symbol': symbol, 'interval': interval, 'limit': limit}
            response = requests.get(url, params=params)
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            return df
        except Exception as e:
            print(f"Error obteniendo datos de {symbol}: {e}")
            return None
    
    def calculate_rsi(self, data, period=14):
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, data):
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram
    
    def calculate_atr(self, data, period=14):
        high = data['high']
        low = data['low']
        close = data['close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def predict_next_move(self, data, window=48):
        """Predice el próximo movimiento usando análisis estadístico y regresión"""
        recent = data.tail(window)
        prices = recent['close'].values
        
        # 1. ANÁLISIS DE MOMENTUM Y ACELERACIÓN
        momentum = np.diff(prices)
        acceleration = np.diff(momentum)
        
        current_momentum = momentum[-1]
        avg_momentum = np.mean(momentum[-20:])
        momentum_trend = "acelerando" if acceleration[-1] > 0 else "desacelerando"
        momentum_strength = abs(current_momentum) / (np.std(momentum) + 1e-10)
        
        # 2. REGRESIÓN LINEAL PARA TENDENCIA
        x = np.arange(len(prices))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
        trend_direction = "alcista" if slope > 0 else "bajista"
        trend_strength = abs(r_value)  # 0-1, qué tan fuerte es la tendencia
        
        # 3. ANÁLISIS DE VOLATILIDAD
        returns = np.diff(prices) / prices[:-1]
        recent_vol = np.std(returns[-20:])
        historical_vol = np.std(returns)
        vol_ratio = recent_vol / (historical_vol + 1e-10)
        
        # 4. REVERSIÓN A LA MEDIA
        sma20 = np.mean(prices[-20:])
        current_price = prices[-1]
        distance_from_sma = (current_price - sma20) / sma20 * 100
        
        # 5. DETECCIÓN DE CANALES Y PATRONES
        recent_high = np.max(prices[-24:])
        recent_low = np.min(prices[-24:])
        price_position = (current_price - recent_low) / (recent_high - recent_low)
        
        # 6. ANÁLISIS DE PUNTOS DE PIVOTE
        pivot = (recent_high + recent_low + current_price) / 3
        
        prediction = {
            'momentum': current_momentum,
            'momentum_trend': momentum_trend,
            'momentum_strength': momentum_strength,
            'trend_direction': trend_direction,
            'trend_strength': trend_strength * 100,
            'volatility_ratio': vol_ratio,
            'distance_from_sma': distance_from_sma,
            'price_position': price_position * 100,
            'pivot': pivot,
            'recent_high': recent_high,
            'recent_low': recent_low,
            'slope': slope
        }
        
        return prediction
    
    def calculate_probability_of_move(self, data):
        """Calcula probabilidad de movimiento alcista usando historia reciente"""
        closes = data['close'].values[-100:]
        returns = np.diff(closes) / closes[:-1]
        
        # Probabilidad de subida
        up_moves = np.sum(returns > 0)
        up_probability = up_moves / len(returns) * 100
        
        # Magnitud promedio de movimientos
        up_magnitude = np.mean(returns[returns > 0]) * 100
        down_magnitude = np.mean(abs(returns[returns < 0])) * 100
        
        # Sesgo
        skewness = stats.skew(returns)
        
        return {
            'up_probability': up_probability,
            'up_magnitude': up_magnitude,
            'down_magnitude': down_magnitude,
            'skewness': skewness
        }
    
    def analyze_context(self, data):
        """Analiza el contexto de mercado para decisiones mejores"""
        closes = data['close'].values
        
        # Últimos 7 días vs 30 días
        change_7d = ((closes[-1] - closes[-168]) / closes[-168] * 100) if len(closes) > 168 else 0
        change_30d = ((closes[-1] - closes[-720]) / closes[-720] * 100) if len(closes) > 720 else 0
        
        # Volatilidad histórica vs reciente
        returns_all = np.diff(closes) / closes[:-1]
        vol_30d = np.std(returns_all[-720:]) * 100
        vol_7d = np.std(returns_all[-168:]) * 100
        
        # Identify regime
        if vol_7d > vol_30d * 1.5:
            volatility_regime = "EXPANSIÓN"
        elif vol_7d < vol_30d * 0.7:
            volatility_regime = "CONTRACCIÓN"
        else:
            volatility_regime = "NORMAL"
        
        return {
            'change_7d': change_7d,
            'change_30d': change_30d,
            'vol_7d': vol_7d,
            'vol_30d': vol_30d,
            'volatility_regime': volatility_regime
        }
    
    def calculate_smart_score(self, df):
        """Calcula score inteligente que PREDICE no solo confirma"""
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Indicadores básicos
        rsi = last['rsi']
        macd_val = last['macd']
        signal_val = last['signal']
        
        score = 0
        signals = []
        
        # PREDICCIÓN (30 puntos) - LO MÁS IMPORTANTE
        prediction = self.predict_next_move(df)
        prob = self.calculate_probability_of_move(df)
        
        if prediction['momentum_strength'] > 1.5 and prediction['trend_direction'] == "alcista":
            score += 30
            signals.append(f"🚀 Momentum fuerte alcista ({prediction['momentum_strength']:.1f}x) (+30)")
        elif prediction['trend_direction'] == "alcista" and prediction['trend_strength'] > 60:
            score += 20
            signals.append(f"📈 Tendencia alcista confirmada ({prediction['trend_strength']:.1f}%) (+20)")
        elif prediction['trend_direction'] == "bajista":
            score -= 15
            signals.append(f"📉 Tendencia bajista detectada (-15)")
        
        # OPORTUNIDADES ESCONDIDAS (25 puntos)
        if abs(prediction['distance_from_sma']) > 3:
            if prediction['distance_from_sma'] < -3:  # Precio muy bajo
                score += 15
                signals.append(f"💰 Precio {abs(prediction['distance_from_sma']):.1f}% bajo media (rebote probable) (+15)")
            elif prediction['distance_from_sma'] > 3:
                score -= 8
                signals.append(f"⚠️  Precio muy alto ({prediction['distance_from_sma']:.1f}%) sobre media (-8)")
        
        # VOLATILIDAD INTELIGENTE (20 puntos)
        if prediction['volatility_ratio'] > 1.3:
            score += 10
            signals.append(f"🌊 Volatilidad en expansión (+10)")
        elif prediction['volatility_ratio'] < 0.7 and prob['up_probability'] > 55:
            score += 8
            signals.append(f"⚡ Baja volatilidad + probabilidad alcista (+8)")
        
        # RSI CONTEXTUADO (15 puntos)
        if rsi < 30 and prediction['momentum_trend'] == "acelerando":
            score += 15
            signals.append(f"✅ RSI sobrevendido + momentum acelerando (+15)")
        elif 30 < rsi < 40:
            score += 12
            signals.append(f"✅ RSI óptimo en zona de compra (+12)")
        elif rsi > 70:
            score -= 8
            signals.append(f"⚠️  RSI sobrecomprado (-8)")
        
        # MACD CRUZANDO (15 puntos)
        if macd_val > signal_val and prev['macd'] <= prev['signal']:
            score += 15
            signals.append(f"✅ Cruce MACD alcista (+15)")
        elif macd_val > signal_val > 0:
            score += 10
            signals.append(f"✅ MACD alcista y positivo (+10)")
        elif macd_val < signal_val and prev['macd'] >= prev['signal']:
            score -= 10
            signals.append(f"⚠️  Cruce MACD bajista (-10)")
        
        # PROBABILIDAD ESTADÍSTICA (10 puntos)
        if prob['up_probability'] > 60:
            score += 10
            signals.append(f"📊 Probabilidad alcista {prob['up_probability']:.0f}% (+10)")
        
        # CONTEXTO DE MERCADO (5 puntos)
        context = self.analyze_context(df)
        if context['volatility_regime'] == "EXPANSIÓN" and context['change_7d'] > 2:
            score += 5
            signals.append(f"🔥 Mercado en expansión alcista (+5)")
        
        score_percentage = max(-100, min(100, score))
        
        return score_percentage, signals, prediction, prob, context
    
    def get_recommendation(self, score, prediction, prob, context):
        """Genera recomendación inteligente que mira al futuro"""
        
        if score >= 75:
            if prediction['momentum_strength'] > 2:
                rec = "🟢 COMPRAR AGRESIVO"
                detail = "Momentum muy fuerte + todas las señales alcistas"
            else:
                rec = "🟢 COMPRAR"
                detail = "Excelente oportunidad"
        elif score >= 60:
            if prob['up_probability'] > 65:
                rec = "🟢 COMPRAR (precaución)"
                detail = "Buena oportunidad + estadística alcista"
            else:
                rec = "🟢 COMPRAR"
                detail = "Señales positivas presentes"
        elif score >= 45:
            if context['distance_from_sma'] < -2 and prediction['trend_direction'] == "alcista":
                rec = "🟡 OBSERVAR - REBOTE PROBABLE"
                detail = "Precio bajo + tendencia alcista = entrada en rebote"
            else:
                rec = "🟡 MANTENER/OBSERVAR"
                detail = "Esperar confirmación"
        elif score >= 30:
            rec = "🟡 PRECAUCIÓN"
            detail = "Señales mixtas, evitar entrada"
        else:
            if prediction['trend_direction'] == "bajista" and prob['up_probability'] < 45:
                rec = "🔴 NO ENTRAR"
                detail = "Tendencia bajista + probabilidad baja"
            else:
                rec = "🔴 EVITAR"
                detail = "Señales desfavorables"
        
        return rec, detail
    
    def analyze_coin(self, symbol):
        """Análisis completo con predicción al futuro"""
        print(f"\n{'='*70}")
        print(f"🔍 ANÁLISIS: {symbol.replace('USDT', '')}/USDT")
        print(f"{'='*70}")
        
        df = self.get_klines(symbol, interval='1h', limit=500)
        if df is None:
            return None
        
        # Calcular indicadores
        df['rsi'] = self.calculate_rsi(df)
        df['macd'], df['signal'], df['histogram'] = self.calculate_macd(df)
        df['atr'] = self.calculate_atr(df)
        
        current_price = df.iloc[-1]['close']
        last = df.iloc[-1]
        
        # CALCULAR SCORE INTELIGENTE
        score, signals, prediction, prob, context = self.calculate_smart_score(df)
        
        # Generar recomendación
        rec, detail = self.get_recommendation(score, prediction, prob, context)
        
        # Crear mensaje
        tg_msg = []
        tg_msg.append(f"\n{'='*50}")
        tg_msg.append(f"🔍 {symbol.replace('USDT', '')}/USDT")
        tg_msg.append(f"{'='*50}")
        
        tg_msg.append(f"\n💰 Precio: ${current_price:,.4f}")
        
        # PREDICCIÓN
        tg_msg.append(f"\n🔮 PREDICCIÓN:")
        tg_msg.append(f"   Tendencia: {prediction['trend_direction'].upper()} ({prediction['trend_strength']:.1f}%)")
        tg_msg.append(f"   Momentum: {prediction['momentum_trend'].upper()} (fuerza: {prediction['momentum_strength']:.1f}x)")
        tg_msg.append(f"   Probabilidad alcista: {prob['up_probability']:.0f}%")
        
        print(f"\n💰 Precio: ${current_price:,.4f}")
        print(f"\n🔮 PREDICCIÓN DEL FUTURO:")
        print(f"   Tendencia: {prediction['trend_direction'].upper()} ({prediction['trend_strength']:.1f}%)")
        print(f"   Momentum: {prediction['momentum_trend'].upper()} (fuerza: {prediction['momentum_strength']:.1f}x)")
        print(f"   Probabilidad alcista: {prob['up_probability']:.0f}%")
        
        # CONTEXTO
        tg_msg.append(f"\n📊 CONTEXTO:")
        tg_msg.append(f"   Régimen: {context['volatility_regime']}")
        tg_msg.append(f"   Cambio 7d: {context['change_7d']:+.2f}%")
        tg_msg.append(f"   Vol 7d: {context['vol_7d']:.2f}% vs 30d: {context['vol_30d']:.2f}%")
        
        print(f"\n📊 CONTEXTO DE MERCADO:")
        print(f"   Régimen: {context['volatility_regime']}")
        print(f"   Cambio 7d: {context['change_7d']:+.2f}%")
        
        # SEÑALES CLAVE
        tg_msg.append(f"\n⚡ SEÑALES:")
        for signal in signals[:5]:  # Top 5
            tg_msg.append(f"   {signal}")
        
        print(f"\n⚡ SEÑALES DETECTADAS:")
        for signal in signals[:5]:
            print(f"   {signal}")
        
        # INDICADORES
        tg_msg.append(f"\n📈 INDICADORES:")
        tg_msg.append(f"   RSI: {last['rsi']:.1f}")
        tg_msg.append(f"   MACD: {last['macd']:.4f} | Signal: {last['signal']:.4f}")
        tg_msg.append(f"   ATR: {last['atr']:.4f}")
        
        print(f"\n📈 INDICADORES:")
        print(f"   RSI: {last['rsi']:.1f}")
        print(f"   MACD: {last['macd']:.4f}")
        
        # RECOMENDACIÓN (SIMPLIFICADA Y CLARA)
        tg_msg.append(f"\n{'='*50}")
        tg_msg.append(f"🎯 SCORE: {score:.0f}/100")
        tg_msg.append(f"\n{rec}")
        tg_msg.append(f"→ {detail}")
        tg_msg.append(f"{'='*50}")
        
        print(f"\n🎯 SCORE FINAL: {score:.0f}/100")
        print(f"{rec}")
        print(f"→ {detail}")
        
        # GESTIÓN DE RIESGO (simple)
        atr = last['atr']
        sl = current_price - (1.5 * atr)
        tp = current_price + (2.5 * atr)
        sl_pct = ((current_price - sl) / current_price) * 100
        tp_pct = ((tp - current_price) / current_price) * 100
        rr = tp_pct / sl_pct if sl_pct > 0 else 0
        
        tg_msg.append(f"\n💼 RIESGO:")
        tg_msg.append(f"   SL: ${sl:,.4f} (-{sl_pct:.2f}%)")
        tg_msg.append(f"   TP: ${tp:,.4f} (+{tp_pct:.2f}%)")
        tg_msg.append(f"   R/R: 1:{rr:.2f}")
        
        self.telegram_messages.append("\n".join(tg_msg))
        
        return {
            'symbol': symbol,
            'price': current_price,
            'score': score,
            'recommendation': rec,
            'stop_loss': sl,
            'take_profit': tp,
            'risk_reward': rr
        }
    
    def run_analysis(self):
        """Ejecuta análisis completo"""
        self.telegram_messages = []
        
        header = [
            "="*50,
            "🚀 CRYPTO ANALYZER - PREDICTIVO",
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "="*50
        ]
        
        print("\n" + "\n".join(header))
        self.telegram_messages.append("\n".join(header))
        
        results = []
        for symbol in self.symbols:
            result = self.analyze_coin(symbol)
            if result:
                results.append(result)
        
        # RANKING
        results.sort(key=lambda x: x['score'], reverse=True)
        
        rank_msg = ["\n" + "="*50, "🏆 RANKING", "="*50]
        for i, r in enumerate(results, 1):
            coin = r['symbol'].replace('USDT', '')
            rank_msg.append(f"{i}. {coin} - Score: {r['score']:.0f}/100 → {r['recommendation']}")
        
        print("\n" + "\n".join(rank_msg))
        self.telegram_messages.append("\n".join(rank_msg))
        
        # MEJOR OPORTUNIDAD
        if results:
            best = results[0]
            best_msg = [
                "\n" + "="*50,
                "⭐ MEJOR OPORTUNIDAD",
                "="*50,
                f"{best['symbol'].replace('USDT', '')}/USDT",
                f"Score: {best['score']:.0f}/100",
                f"Entry: ${best['price']:,.4f}",
                f"SL: ${best['stop_loss']:,.4f}",
                f"TP: ${best['take_profit']:,.4f}",
                f"R/R: 1:{best['risk_reward']:.2f}",
                f"\n{best['recommendation']}"
            ]
            
            print("\n" + "\n".join(best_msg))
            self.telegram_messages.append("\n".join(best_msg))
        
        return results
    
    async def send_telegram_message(self, message):
        if self.bot and self.chat_id:
            try:
                max_length = 4000
                if len(message) > max_length:
                    parts = [message[i:i+max_length] for i in range(0, len(message), max_length)]
                    for part in parts:
                        await self.bot.send_message(chat_id=self.chat_id, text=part)
                        await asyncio.sleep(1)
                else:
                    await self.bot.send_message(chat_id=self.chat_id, text=message)
                print("✅ Mensaje enviado a Telegram")
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def send_all_to_telegram(self):
        if self.bot and self.chat_id:
            print(f"\n📱 Enviando {len(self.telegram_messages)} mensajes...")
            try:
                for i, msg in enumerate(self.telegram_messages, 1):
                    await self.send_telegram_message(msg)
                    print(f"   ✅ Mensaje {i}/{len(self.telegram_messages)}")
                    await asyncio.sleep(2)
            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Cargar variables de entorno desde el archivo .env
    load_dotenv()
    
    # Obtener variables de entorno
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
    CHAT_ID = os.getenv('CHAT_ID')
    
    analyzer = AdvancedCryptoAnalyzer(telegram_token=TELEGRAM_TOKEN, chat_id=CHAT_ID)
    
    print("🚀 Iniciando análisis predictivo...")
    results = analyzer.run_analysis()
    
    print("\n" + "="*70)
    print("📱 ENVIANDO A TELEGRAM")
    print("="*70)
    
    try:
        asyncio.run(analyzer.send_all_to_telegram())
    except Exception as e:
        print(f"⚠️  Error: {e}")