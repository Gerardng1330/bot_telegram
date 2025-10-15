"""
Bot de Alertas de Trading v3.0 - Filtros Inteligentes
Incluye análisis de niveles de precio y contexto de mercado
Evita falsos positivos con scoring contextual
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import warnings
from telegram import Bot
warnings.filterwarnings('ignore')

class SmartTradingBot:
    def __init__(self, telegram_token=None, chat_id=None):
        self.all_symbols = []
        self.opportunities = []
        self.telegram_token = telegram_token
        self.chat_id = chat_id
        self.bot = Bot(token=telegram_token) if telegram_token else None
        
    def get_comprehensive_symbol_list(self):
        """Lista completa de símbolos"""
        sp500_top = [
            # Tech Giants
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA',
            'AVGO', 'ORCL', 'ADBE', 'CRM', 'CSCO', 'ACN', 'AMD', 'INTC',
            'TXN', 'QCOM', 'IBM', 'INTU', 'NOW', 'AMAT', 'MU', 'ADI', 'LRCX',
            
            # Finance
            'BRK-B', 'JPM', 'BAC', 'WFC', 'MS', 'GS', 'C', 'BLK', 'SCHW',
            'AXP', 'SPGI', 'USB', 'PNC', 'TFC', 'COF', 'BK', 'STT',
            
            # Healthcare
            'UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT', 'DHR',
            'CVS', 'MDT', 'BMY', 'AMGN', 'GILD', 'CI', 'ISRG', 'VRTX', 'REGN',
            
            # Consumer
            'WMT', 'HD', 'MCD', 'DIS', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX',
            'COST', 'PG', 'KO', 'PEP', 'PM', 'MO', 'CL', 'EL', 'MDLZ',
            
            # Industrial & Energy
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY',
            'CAT', 'BA', 'GE', 'RTX', 'HON', 'UPS', 'DE', 'LMT', 'MMM',
            
            # Communication & Media
            'NFLX', 'CMCSA', 'T', 'VZ', 'TMUS', 'CHTR', 'EA',
            
            # Other Leaders
            'V', 'MA', 'PYPL', 'ADP', 'ADSK', 'PAYX', 'FTNT', 'PANW', 'CRWD'
        ]
        
        growth_stocks = [
            # EV & Auto
            'RIVN', 'LCID', 'F', 'GM', 'MBLY',
            
            # FinTech & Crypto
            'COIN', 'HOOD', 'SOFI', 'AFRM', 'UPST', 'NU', 'MELI',
            
            # Cloud & SaaS
            'SNOW', 'DDOG', 'NET', 'CRWD', 'ZS', 'OKTA', 'DKNG', 'RBLX',
            'U', 'TWLO', 'ZM', 'DOCU', 'MDB', 'TEAM', 'WDAY', 'VEEV',
            
            # E-commerce & Retail
            'SHOP', 'ETSY', 'W', 'CHWY', 'BABA', 'JD', 'PDD', 'SE',
            
            # Streaming & Social
            'ROKU', 'SPOT', 'SNAP', 'PINS', 'MTCH', 'BMBL',
            
            # Semiconductors
            'ASML', 'TSM', 'KLAC', 'SNPS', 'CDNS', 'MRVL', 'ON', 'MPWR',
            
            # Biotech
            'MRNA', 'BNTX', 'CRSP', 'EDIT', 'BEAM', 'NTLA',
            
            # Green Energy
            'ENPH', 'FSLR', 'SEDG', 'RUN', 'PLUG', 'BE', 'CHPT', 'BLNK',
            
            # Cannabis
            'TLRY', 'CGC', 'CRON',
            
            # Gaming & Entertainment
            'TTWO', 'RBLX', 'DKNG', 'PENN', 'LYV',
            
            # Special Situations
            'SPCE', 'OPEN', 'ABNB', 'LYFT', 'UBER', 'DASH'
        ]
        
        sector_etfs = [
            'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLC',
            'VGT', 'VFH', 'VDE', 'VHT', 'VIS', 'VCR', 'VDC',
            'SMH', 'XBI', 'IBB', 'KRE', 'XME', 'XHB', 'XRT',
            'SPY', 'QQQ', 'DIA', 'IWM'
        ]
        
        crypto_stocks = ['MARA', 'RIOT', 'CLSK', 'HUT', 'COIN', 'HOOD', 'MSTR']
        
        all_symbols = set()
        all_symbols.update(sp500_top)
        all_symbols.update(growth_stocks)
        all_symbols.update(sector_etfs)
        all_symbols.update(crypto_stocks)
        
        # Remove known delisted
        invalid = {'WISH', 'BBBY', 'SQ', 'DFS', 'WWE', 'ATVI'}
        all_symbols = all_symbols - invalid
        
        return sorted(list(all_symbols))
    
    def load_market_symbols(self):
        """Carga símbolos del mercado"""
        print(f"\n{'='*70}")
        print(f"📊 CARGANDO UNIVERSO DE ACCIONES USA v3.0")
        print(f"{'='*70}\n")
        
        self.all_symbols = self.get_comprehensive_symbol_list()
        
        print(f"✅ {len(self.all_symbols)} símbolos cargados")
        print(f"   • Mega caps + Growth stocks + ETFs + Crypto")
        print(f"{'='*70}\n")
        
        return self.all_symbols
    
    def get_stock_data(self, symbol, period='6mo', interval='1d'):
        """Obtiene datos con retry"""
        max_retries = 2
        for attempt in range(max_retries):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)
                if df.empty or len(df) < 30:
                    return None
                return df
            except:
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    continue
                return None
    
    # ==================== ANÁLISIS DE NIVELES DE PRECIO ====================
    
    def analyze_price_levels(self, df, symbol):
        """
        Análisis profundo de niveles de precio
        - ATH/ATL de 52 semanas
        - Distancia a niveles clave
        - Soporte/Resistencia
        - Contexto del precio actual
        """
        if df is None or len(df) < 30:
            return None
        
        current_price = df['Close'].iloc[-1]
        
        # ATH y ATL de 52 semanas (o disponible)
        high_52w = df['High'].max()
        low_52w = df['Low'].min()
        
        # Distancias en %
        dist_from_high = ((current_price - high_52w) / high_52w) * 100
        dist_from_low = ((current_price - low_52w) / low_52w) * 100
        
        # Rango actual
        price_range_pct = ((high_52w - low_52w) / low_52w) * 100
        
        # Posición en el rango (0 = mínimo, 100 = máximo)
        position_in_range = ((current_price - low_52w) / (high_52w - low_52w)) * 100
        
        # Contexto de precio
        if position_in_range >= 95:
            context = "NEAR_ATH"
            context_emoji = "🔴"
            context_desc = "Cerca de máximos 52w"
        elif position_in_range >= 80:
            context = "UPPER_RANGE"
            context_emoji = "🟡"
            context_desc = "Rango alto"
        elif position_in_range >= 50:
            context = "MID_RANGE"
            context_emoji = "🟢"
            context_desc = "Rango medio"
        elif position_in_range >= 20:
            context = "LOWER_RANGE"
            context_emoji = "🟢"
            context_desc = "Rango bajo"
        else:
            context = "NEAR_LOW"
            context_emoji = "🟢"
            context_desc = "Cerca de mínimos 52w"
        
        # Identificar soportes y resistencias (swing points)
        supports = []
        resistances = []
        
        # Buscar pivots en últimos 60 días
        lookback = min(60, len(df))
        df_recent = df.tail(lookback)
        
        for i in range(2, len(df_recent) - 2):
            # Soporte: mínimo local
            if df_recent['Low'].iloc[i] < df_recent['Low'].iloc[i-1] and \
               df_recent['Low'].iloc[i] < df_recent['Low'].iloc[i-2] and \
               df_recent['Low'].iloc[i] < df_recent['Low'].iloc[i+1] and \
               df_recent['Low'].iloc[i] < df_recent['Low'].iloc[i+2]:
                supports.append(df_recent['Low'].iloc[i])
            
            # Resistencia: máximo local
            if df_recent['High'].iloc[i] > df_recent['High'].iloc[i-1] and \
               df_recent['High'].iloc[i] > df_recent['High'].iloc[i-2] and \
               df_recent['High'].iloc[i] > df_recent['High'].iloc[i+1] and \
               df_recent['High'].iloc[i] > df_recent['High'].iloc[i+2]:
                resistances.append(df_recent['High'].iloc[i])
        
        # Encontrar soporte y resistencia más cercanos
        supports = sorted(supports, reverse=True)
        resistances = sorted(resistances)
        
        nearest_support = None
        nearest_resistance = None
        
        for s in supports:
            if s < current_price:
                nearest_support = s
                break
        
        for r in resistances:
            if r > current_price:
                nearest_resistance = r
                break
        
        # Calcular risk/reward
        risk_reward_ratio = None
        if nearest_support and nearest_resistance:
            potential_gain = nearest_resistance - current_price
            potential_loss = current_price - nearest_support
            if potential_loss > 0:
                risk_reward_ratio = potential_gain / potential_loss
        
        return {
            'high_52w': round(high_52w, 2),
            'low_52w': round(low_52w, 2),
            'dist_from_high_pct': round(dist_from_high, 2),
            'dist_from_low_pct': round(dist_from_low, 2),
            'position_in_range': round(position_in_range, 1),
            'price_range_pct': round(price_range_pct, 1),
            'context': context,
            'context_emoji': context_emoji,
            'context_desc': context_desc,
            'nearest_support': round(nearest_support, 2) if nearest_support else None,
            'nearest_resistance': round(nearest_resistance, 2) if nearest_resistance else None,
            'risk_reward_ratio': round(risk_reward_ratio, 2) if risk_reward_ratio else None
        }
    
    # ==================== SCORING CONTEXTUAL INTELIGENTE ====================
    
    def calculate_context_score(self, price_levels):
        """
        Ajusta el score según el contexto de precio
        Penaliza cerca de ATH sin confirmación
        Premia rebotes desde mínimos
        """
        if not price_levels:
            return 0, []
        
        score = 0
        signals = []
        
        context = price_levels['context']
        position = price_levels['position_in_range']
        
        # BONIFICACIONES por contexto favorable
        if context == "NEAR_LOW":  # 0-20%
            score += 30
            signals.append(f"🟢 Cerca de mínimos 52w ({position:.0f}% rango)")
        elif context == "LOWER_RANGE":  # 20-50%
            score += 20
            signals.append(f"🟢 Rango bajo favorable ({position:.0f}%)")
        elif context == "MID_RANGE":  # 50-80%
            score += 10
            signals.append(f"🟡 Rango medio ({position:.0f}%)")
        
        # PENALIZACIONES por contexto desfavorable
        elif context == "UPPER_RANGE":  # 80-95%
            score -= 10
            signals.append(f"⚠️ Rango alto ({position:.0f}%) - Cuidado resistencia")
        elif context == "NEAR_ATH":  # 95-100%
            score -= 20
            signals.append(f"🔴 Cerca ATH ({position:.0f}%) - Alto riesgo")
        
        # Risk/Reward ratio
        if price_levels['risk_reward_ratio']:
            rr = price_levels['risk_reward_ratio']
            if rr >= 3:
                score += 20
                signals.append(f"💎 R/R excelente: {rr:.1f}:1")
            elif rr >= 2:
                score += 10
                signals.append(f"✅ R/R bueno: {rr:.1f}:1")
            elif rr < 1:
                score -= 15
                signals.append(f"❌ R/R malo: {rr:.1f}:1")
        
        # Bonus por estar cerca de soporte
        if price_levels['nearest_support']:
            support = price_levels['nearest_support']
            current = price_levels['high_52w'] * position / 100 + price_levels['low_52w'] * (100 - position) / 100
            dist_to_support = ((current - support) / support) * 100
            
            if dist_to_support <= 3:
                score += 15
                signals.append(f"📊 Muy cerca soporte ${support:.2f}")
            elif dist_to_support <= 5:
                score += 8
                signals.append(f"📊 Cerca soporte ${support:.2f}")
        
        return score, signals
    
    # ==================== FILTROS ANTI-FALSOS POSITIVOS ====================
    
    def apply_smart_filters(self, result):
        """
        Filtros inteligentes para eliminar falsos positivos
        """
        if not result or not result.get('price_levels'):
            return False, []
        
        warnings = []
        disqualify = False
        
        price_levels = result['price_levels']
        indicators = result.get('indicators', {})
        
        # FILTRO 1: Evitar comprar en ATH sin confirmación
        if price_levels['context'] == "NEAR_ATH":
            rsi = indicators.get('rsi', 50)
            if rsi > 70:
                warnings.append("⛔ RSI sobrecomprado en ATH - ALTO RIESGO")
                disqualify = True
            elif rsi > 65:
                warnings.append("⚠️ RSI alto cerca ATH - Precaución")
        
        # FILTRO 2: Volumen insuficiente (distribución)
        momentum = result.get('momentum', {})
        metrics = momentum.get('metrics', {})
        vol_ratio = metrics.get('vol_ratio', 1.0)
        
        if price_levels['position_in_range'] > 80 and vol_ratio < 0.8:
            warnings.append("⚠️ Volumen bajando en zona alta - Distribución posible")
        
        # FILTRO 3: Risk/Reward desfavorable
        if price_levels['risk_reward_ratio'] and price_levels['risk_reward_ratio'] < 0.8:
            warnings.append("⛔ Risk/Reward < 1:1 - NO recomendado")
            disqualify = True
        
        # FILTRO 4: Divergencia negativa (precio sube, indicadores bajan)
        if price_levels['position_in_range'] > 85:
            returns_5d = metrics.get('returns_5d', 0)
            if returns_5d > 5 and indicators.get('rsi', 50) < 55:
                warnings.append("⚠️ Posible divergencia negativa")
        
        # FILTRO 5: Falta de setup claro
        total_score = result.get('total_score', 0)
        if total_score < 45:
            warnings.append("⚠️ Setup débil - Score muy bajo")
        
        return not disqualify, warnings
    
    # ==================== ANÁLISIS TÉCNICO ====================
    
    def detect_accumulation(self, df):
        """Detecta acumulación"""
        if df is None or len(df) < 20:
            return {'score': 0, 'signals': []}
        
        signals = []
        score = 0
        
        # OBV
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        df['OBV'] = obv
        obv_recent = df['OBV'].tail(10)
        obv_slope = np.polyfit(range(len(obv_recent)), obv_recent, 1)[0]
        
        if obv_slope > 0:
            score += 25
            signals.append("OBV alcista")
        
        # Volumen en días verdes
        last_10 = df.tail(10)
        green_days = last_10[last_10['Close'] > last_10['Open']]
        
        if len(green_days) >= 6:
            avg_vol_green = green_days['Volume'].mean()
            avg_vol_total = last_10['Volume'].mean()
            if avg_vol_green > avg_vol_total * 1.05:
                score += 20
                signals.append(f"Vol verde ({len(green_days)}/10)")
        
        return {'score': score, 'signals': signals}
    
    def calculate_indicators(self, df):
        """Indicadores técnicos"""
        if df is None or len(df) < 50:
            return {'score': 0, 'signals': [], 'rsi': 50}
        
        signals = []
        score = 0
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        current_rsi = df['RSI'].iloc[-1]
        
        if 30 <= current_rsi <= 45:
            score += 25
            signals.append(f"RSI {current_rsi:.0f} (sobreventa)")
        elif 45 < current_rsi <= 55:
            score += 20
            signals.append(f"RSI {current_rsi:.0f} (neutral)")
        elif 55 < current_rsi <= 65:
            score += 12
            signals.append(f"RSI {current_rsi:.0f} (momentum)")
        elif current_rsi > 70:
            score -= 10
            signals.append(f"RSI {current_rsi:.0f} (sobrecomprado)")
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        if df['MACD'].iloc[-1] > df['Signal_Line'].iloc[-1] and \
           df['MACD'].iloc[-2] <= df['Signal_Line'].iloc[-2]:
            score += 30
            signals.append("MACD cruce alcista ⭐")
        elif df['MACD'].iloc[-1] > df['Signal_Line'].iloc[-1]:
            score += 15
            signals.append("MACD positivo")
        
        # Bollinger Bands
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['BB_std'] = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['SMA20'] + (df['BB_std'] * 2)
        df['BB_lower'] = df['SMA20'] - (df['BB_std'] * 2)
        
        current_price = df['Close'].iloc[-1]
        bb_lower = df['BB_lower'].iloc[-1]
        bb_upper = df['BB_upper'].iloc[-1]
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
        
        if bb_position <= 0.25:
            score += 20
            signals.append("En banda inferior BB")
        elif bb_position <= 0.4:
            score += 10
            signals.append("Cerca banda inferior")
        
        if current_price > df['SMA20'].iloc[-1]:
            score += 8
            signals.append("Sobre SMA20")
        
        return {'score': score, 'signals': signals, 'rsi': current_rsi}
    
    def analyze_momentum(self, df):
        """Análisis de momentum"""
        if df is None or len(df) < 20:
            return {'score': 0, 'signals': [], 'metrics': {}}
        
        signals = []
        score = 0
        
        try:
            ret_5d = ((df['Close'].iloc[-1] / df['Close'].iloc[-6]) - 1) * 100
            
            if ret_5d > 5:
                score += 20
                signals.append(f"+{ret_5d:.1f}% en 5d")
            elif ret_5d > 2:
                score += 10
                signals.append(f"+{ret_5d:.1f}% en 5d")
            elif ret_5d < -5:
                score -= 10
                signals.append(f"{ret_5d:.1f}% en 5d (caída)")
            
            # Volumen
            avg_vol = df['Volume'].tail(20).mean()
            current_vol = df['Volume'].tail(5).mean()
            vol_ratio = current_vol / avg_vol
            
            if vol_ratio > 1.4:
                score += 20
                signals.append(f"Vol {vol_ratio:.1f}x")
            elif vol_ratio > 1.15:
                score += 10
                signals.append(f"Vol {vol_ratio:.1f}x")
            
            return {'score': score, 'signals': signals, 'metrics': {
                'returns_5d': ret_5d,
                'vol_ratio': vol_ratio
            }}
        except:
            return {'score': 0, 'signals': [], 'metrics': {}}
    
    def calculate_potential(self, df):
        """Potencial de ganancia"""
        if df is None or len(df) < 20:
            return 5.0
        
        try:
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]
            
            current_price = df['Close'].iloc[-1]
            potential = (atr / current_price) * 100 * 2.5
            
            return round(potential, 2)
        except:
            return 5.0
    
    # ==================== ANÁLISIS COMPLETO ====================
    
    def analyze_single_stock(self, symbol):
        """Análisis completo con filtros inteligentes"""
        try:
            df = self.get_stock_data(symbol, period='6mo')
            if df is None:
                return None
            
            current_price = df['Close'].iloc[-1]
            change_1d = ((df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1) * 100
            
            # Análisis de niveles de precio
            price_levels = self.analyze_price_levels(df, symbol)
            if not price_levels:
                return None
            
            # Análisis técnico
            accumulation = self.detect_accumulation(df)
            indicators = self.calculate_indicators(df)
            momentum = self.analyze_momentum(df)
            
            # Score base
            base_score = (
                accumulation['score'] * 0.30 +
                indicators['score'] * 0.45 +
                momentum['score'] * 0.25
            )
            
            # Scoring contextual (ajusta según niveles)
            context_score, context_signals = self.calculate_context_score(price_levels)
            
            # Score total ajustado
            total_score = base_score + context_score
            total_score = max(0, min(100, total_score))  # Limitar 0-100
            
            potential = self.calculate_potential(df)
            
            result = {
                'symbol': symbol,
                'price': round(current_price, 2),
                'change_1d': round(change_1d, 2),
                'total_score': round(total_score, 2),
                'base_score': round(base_score, 2),
                'context_score': context_score,
                'potential_pct': potential,
                'price_levels': price_levels,
                'accumulation': accumulation,
                'indicators': indicators,
                'momentum': momentum,
                'context_signals': context_signals,
                'rsi': indicators.get('rsi', 50)
            }
            
            # Aplicar filtros anti-falsos positivos
            passed, warnings = self.apply_smart_filters(result)
            result['passed_filters'] = passed
            result['warnings'] = warnings
            
            return result
            
        except:
            return None
    
    # ==================== ESCANEO ====================
    
    def scan_market(self, min_score=50, min_potential=4, max_workers=10, 
                    filter_false_positives=True):
        """Escanea el mercado con filtros inteligentes"""
        print(f"\n{'='*70}")
        print(f"🚀 ESCANEANDO {len(self.all_symbols)} ACCIONES (Filtros Inteligentes v3.0)")
        print(f"{'='*70}")
        print(f"⚙️ Score ≥ {min_score} | Potencial ≥ {min_potential}%")
        print(f"⚙️ Anti-falsos positivos: {'✅ Activado' if filter_false_positives else '❌ Desactivado'}\n")
        
        opportunities = []
        filtered_out = []
        processed = 0
        errors = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.analyze_single_stock, symbol): symbol 
                for symbol in self.all_symbols
            }
            
            for future in as_completed(future_to_symbol):
                processed += 1
                
                if processed % 30 == 0:
                    print(f"📊 {processed}/{len(self.all_symbols)} | "
                          f"✅ {len(opportunities)} | 🚫 {len(filtered_out)} | ❌ {errors}")
                
                try:
                    result = future.result()
                    if result and result['total_score'] >= min_score and \
                       result['potential_pct'] >= min_potential:
                        
                        if filter_false_positives and not result['passed_filters']:
                            filtered_out.append(result)
                            print(f"🚫 {result['symbol']}: Filtrado - {result['warnings'][0] if result['warnings'] else 'No cumple criterios'}")
                        else:
                            opportunities.append(result)
                            pl = result['price_levels']
                            print(f"✅ {result['symbol']}: Score {result['total_score']:.0f} | "
                                  f"Pot +{result['potential_pct']:.1f}% | {pl['context_emoji']} {pl['context_desc']}")
                except:
                    errors += 1
        
        # Ordenar por score
        opportunities.sort(key=lambda x: x['total_score'], reverse=True)
        filtered_out.sort(key=lambda x: x['total_score'], reverse=True)
        
        print(f"\n{'='*70}")
        print(f"✅ ESCANEO COMPLETADO")
        print(f"{'='*70}")
        print(f"Procesadas: {processed} | ✅ Válidas: {len(opportunities)} | "
              f"🚫 Filtradas: {len(filtered_out)} | ❌ Errores: {errors}\n")
        
        self.opportunities = opportunities
        self.filtered_out = filtered_out
        return opportunities
    
    # ==================== VISUALIZACIÓN DE RESULTADOS ====================
    
    def print_opportunities(self, top_n=30):
        """Muestra oportunidades con análisis detallado"""
        if not self.opportunities:
            print("❌ No se encontraron oportunidades")
            print("💡 Sugerencia: Reduce min_score o desactiva filtros")
            return
            
        message = []
        message.append("🚀 ANÁLISIS DE MERCADO - OPORTUNIDADES")
        message.append(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        message.append("-" * 40)
        
        print(f"\n{'='*70}")
        print(f"🏆 TOP {min(top_n, len(self.opportunities))} OPORTUNIDADES VALIDADAS")
        print(f"{'='*70}\n")
        
        # Agrupar por contexto
        near_low = [o for o in self.opportunities if o['price_levels']['context'] in ['NEAR_LOW', 'LOWER_RANGE']]
        mid_range = [o for o in self.opportunities if o['price_levels']['context'] == 'MID_RANGE']
        breakouts = [o for o in self.opportunities if o['price_levels']['context'] in ['UPPER_RANGE', 'NEAR_ATH']]
        
        if near_low:
            header = [
                f"\n🟢 REBOTES DESDE MÍNIMOS ({len(near_low)})",
                f"Oportunidades con mejor R/R y menor riesgo"
            ]
            print("\n".join(header))
            message.extend(header)
            
            # Mostrar solo las 5 mejores
            for opp in near_low[:5]:
                opp_details = self._print_detailed_opportunity(opp)
                message.extend(opp_details)
        
        if mid_range:
            header = [
                f"\n🟡 ZONA MEDIA - MOMENTUM ({len(mid_range)})",
                f"Balance riesgo/oportunidad"
            ]
            print("\n".join(header))
            message.extend(header)
            
            # Mostrar solo las 3 mejores
            for opp in mid_range[:3]:
                opp_details = self._print_detailed_opportunity(opp)
                message.extend(opp_details)
        
        if breakouts:
            header = [
                f"\n🔴 BREAKOUTS CONFIRMADOS ({len(breakouts)})",
                f"Alto momentum - Mayor riesgo"
            ]
            print("\n".join(header))
            message.extend(header)
            
            # Mostrar solo las 2 mejores
            for opp in breakouts[:2]:
                opp_details = self._print_detailed_opportunity(opp)
                message.extend(opp_details)
                
        # Agregar resumen final
        if self.opportunities:
            footer = [
                f"\n📊 RESUMEN",
                f"✅ {len(self.opportunities)} oportunidades encontradas",
                f"🏆 Mejor: {self.opportunities[0]['symbol']} (Score: {self.opportunities[0]['total_score']:.0f})",
                f"⏰ Próxima actualización en 24h"
            ]
            message.extend(footer)
                
        return "\n".join(message)
    
    def _print_detailed_opportunity(self, opp):
        """Genera un resumen conciso de la oportunidad para Telegram"""
        pl = opp['price_levels']
        lines = []
        
        # Línea principal con información esencial
        main_line = (f"\n{pl['context_emoji']} {opp['symbol']:<6} ${opp['price']:>8.2f} ({opp['change_1d']:+.1f}%)")
        print(main_line)
        lines.append(main_line)
        
        # Resumen en una línea
        summary = []
        summary.append(f"Score: {opp['total_score']:.0f}")
        if opp['total_score'] >= 70:
            summary.append("⭐ FUERTE")
        summary.append(f"RSI: {opp['rsi']:.0f}")
        if pl['risk_reward_ratio'] and pl['risk_reward_ratio'] >= 2:
            summary.append(f"R/R: {pl['risk_reward_ratio']:.1f}")
        
        summary_line = f"   📊 {' | '.join(summary)}"
        print(summary_line)
        lines.append(summary_line)
        
        # Señales más importantes
        signals = []
        # Contexto
        signals.extend(opp['context_signals'][:1])
        # Mejor señal técnica
        if opp['indicators']['signals']:
            signals.extend(opp['indicators']['signals'][:1])
        # Momentum si es relevante
        if opp['momentum']['signals'] and "+" in opp['momentum']['signals'][0]:
            signals.extend(opp['momentum']['signals'][:1])
        
        if signals:
            signals_line = f"   ✨ {' • '.join(signals)}"
            print(signals_line)
            lines.append(signals_line)
        
        # Niveles clave para trading
        levels = []
        if pl['nearest_support']:
            levels.append(f"SL: ${pl['nearest_support']:.2f}")
        if pl['nearest_resistance']:
            levels.append(f"TP: ${pl['nearest_resistance']:.2f}")
        
        if levels:
            levels_line = f"   🎯 {' | '.join(levels)}"
            print(levels_line)
            lines.append(levels_line)
        
        # Advertencia principal si existe
        if opp.get('warnings'):
            warning_line = f"   ⚠️ {opp['warnings'][0]}"
            print(warning_line)
            lines.append(warning_line)
        
        return lines
    
    def print_filtered_out(self, top_n=10):
        """Muestra las que fueron filtradas y por qué"""
        if not hasattr(self, 'filtered_out') or not self.filtered_out:
            return
        
        print(f"\n{'='*70}")
        print(f"🚫 OPORTUNIDADES FILTRADAS (Falsos Positivos) - Top {top_n}")
        print(f"{'='*70}")
        print(f"Estas acciones cumplían score mínimo pero fueron descartadas:\n")
        
        for i, opp in enumerate(self.filtered_out[:top_n], 1):
            pl = opp['price_levels']
            print(f"{i}. {opp['symbol']:<6} ${opp['price']:>8.2f} | Score: {opp['total_score']:.0f} | "
                  f"{pl['context_emoji']} {pl['context_desc']}")
            for warning in opp['warnings'][:2]:
                print(f"   {warning}")
            print()
    
    async def send_telegram_message(self, message):
        """Envía un mensaje a Telegram"""
        if self.bot and self.chat_id:
            try:
                await self.bot.send_message(chat_id=self.chat_id, text=message)
                print("✅ Mensaje enviado a Telegram")
            except Exception as e:
                print(f"❌ Error enviando mensaje a Telegram: {e}")

    def export_to_csv(self, filename='oportunidades_smart.csv'):
        """Exporta a CSV con análisis completo"""
        if not self.opportunities:
            print("❌ No hay datos para exportar")
            return
        
        data = []
        for opp in self.opportunities:
            pl = opp['price_levels']
            data.append({
                'Symbol': opp['symbol'],
                'Price': opp['price'],
                'Change_1D_%': opp['change_1d'],
                'Score_Total': opp['total_score'],
                'Score_Base': opp['base_score'],
                'Score_Context': opp['context_score'],
                'Potential_%': opp['potential_pct'],
                'RSI': opp['rsi'],
                'Context': pl['context_desc'],
                'Position_Range_%': pl['position_in_range'],
                'ATH_52w': pl['high_52w'],
                'ATL_52w': pl['low_52w'],
                'Dist_ATH_%': pl['dist_from_high_pct'],
                'Dist_ATL_%': pl['dist_from_low_pct'],
                'Risk_Reward': pl['risk_reward_ratio'],
                'Support': pl['nearest_support'],
                'Resistance': pl['nearest_resistance'],
                'Top_Signals': ' | '.join(
                    opp['context_signals'][:1] +
                    opp['indicators']['signals'][:1] +
                    opp['momentum']['signals'][:1]
                )
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"\n✅ Datos exportados a: {filename}")
        print(f"   Total registros: {len(df)}")
        print(f"   Columnas: {len(df.columns)}")
    
    def get_summary_stats(self):
        """Estadísticas del escaneo"""
        if not self.opportunities:
            return
        
        print(f"\n{'='*70}")
        print(f"📊 ESTADÍSTICAS DEL ESCANEO")
        print(f"{'='*70}\n")
        
        # Por contexto
        contexts = {}
        for opp in self.opportunities:
            ctx = opp['price_levels']['context']
            contexts[ctx] = contexts.get(ctx, 0) + 1
        
        print("Distribución por contexto:")
        context_names = {
            'NEAR_LOW': '🟢 Cerca mínimos',
            'LOWER_RANGE': '🟢 Rango bajo',
            'MID_RANGE': '🟡 Rango medio',
            'UPPER_RANGE': '🔴 Rango alto',
            'NEAR_ATH': '🔴 Cerca ATH'
        }
        for ctx, count in sorted(contexts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {context_names.get(ctx, ctx)}: {count}")
        
        # Scores promedio
        avg_score = sum(o['total_score'] for o in self.opportunities) / len(self.opportunities)
        avg_potential = sum(o['potential_pct'] for o in self.opportunities) / len(self.opportunities)
        avg_rsi = sum(o['rsi'] for o in self.opportunities) / len(self.opportunities)
        
        print(f"\nPromedios:")
        print(f"  Score: {avg_score:.1f}/100")
        print(f"  Potencial: +{avg_potential:.1f}%")
        print(f"  RSI: {avg_rsi:.1f}")
        
        # Top 3
        print(f"\nTop 3 mejores scores:")
        for i, opp in enumerate(self.opportunities[:3], 1):
            pl = opp['price_levels']
            print(f"  {i}. {opp['symbol']}: {opp['total_score']:.0f} | {pl['context_emoji']} {pl['context_desc']}")


# ==================== EJEMPLO DE USO ====================

if __name__ == "__main__":
    import asyncio
    # Token y chat_id de Telegram
    TELEGRAM_TOKEN = '8246723060:AAEi4QcWlEdQtVrwMbEdIXDbolxiuTVrWmc'
    CHAT_ID = '1609734261'
    
    # Crear bot inteligente
    bot = SmartTradingBot(telegram_token=TELEGRAM_TOKEN, chat_id=CHAT_ID)
    
    # Cargar símbolos
    bot.load_market_symbols()
    
    # Escanear con filtros inteligentes
    opportunities = bot.scan_market(
        min_score=50,                    # Score mínimo
        min_potential=4,                 # Potencial mínimo %
        max_workers=12,                  # Procesamiento paralelo
        filter_false_positives=True      # Activar filtros anti-falsos positivos
    )
    
    # Mostrar resultados detallados
    bot.print_opportunities(top_n=25)
    
    # Ver qué fue filtrado (opcional)
    bot.print_filtered_out(top_n=10)
    
    # Estadísticas
    bot.get_summary_stats()
    
    # Mostrar resultados detallados y obtener mensaje para Telegram
    telegram_message = bot.print_opportunities(top_n=25)
    
    # Ver qué fue filtrado (opcional)
    bot.print_filtered_out(top_n=10)
    
    # Estadísticas
    bot.get_summary_stats()
    
    # Exportar
    bot.export_to_csv('oportunidades_smart.csv')
    
    # Enviar mensaje a Telegram
    try:
        asyncio.run(bot.send_telegram_message(telegram_message))
    except Exception as e:
        print(f"Error al enviar mensaje a Telegram: {e}")
        print("Verifica que el token y chat_id sean correctos")