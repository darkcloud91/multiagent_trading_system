import os
from typing import Optional
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.data.enums import DataFeed

class AlpacaAPI:
    def __init__(self):
        """Inicializa la clase con las claves de la API desde variables de entorno."""
        # ✅ Corregido a los nombres de variable estándar de la SDK de Alpaca
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.is_paper_trading = os.getenv('ALPACA_PAPER_TRADING', 'False').lower() == 'true'

        if not self.api_key or not self.secret_key:
            raise ValueError("Las variables de entorno ALPACA_API_KEY y ALPACA_SECRET_KEY deben estar configuradas.")

        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=self.is_paper_trading)
        self.stock_data_client = StockHistoricalDataClient(self.api_key, self.secret_key)

    def get_account_info(self):
        """Obtiene y devuelve la información de la cuenta."""
        try:
            account = self.trading_client.get_account()
            return account
        except Exception as e:
            print(f"Error al obtener la información de la cuenta: {e}")
            return None

    def submit_order(self, symbol: str, side: str, qty: Optional[float] = None, notional: Optional[float] = None):
        """
        Ejecuta una orden de mercado para un símbolo y cantidad/valor nocional dados.

        Args:
            symbol (str): El ticker de la acción (e.g., "AAPL").
            side (str): El tipo de operación ("buy" o "sell").
            qty (Optional[float]): El número de acciones a operar.
            notional (Optional[float]): La cantidad de dinero a invertir/vender.
        """
        if not (qty or notional) or (qty and notional):
            raise ValueError("Se debe proporcionar 'qty' o 'notional', pero no ambos.")

        if side.lower() == "buy":
            order_side = OrderSide.BUY
        elif side.lower() == "sell":
            order_side = OrderSide.SELL
        else:
            raise ValueError("Lado de la orden inválido. Debe ser 'buy' o 'sell'.")

        market_order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            notional=notional,
            side=order_side,
            time_in_force=TimeInForce.DAY
        )

        try:
            market_order = self.trading_client.submit_order(order_data=market_order_data)
            print(f"Orden de {side} para {symbol} enviada.")
            return market_order
        except Exception as e:
            print(f"Error al enviar la orden: {e}")
            return None

    def get_open_positions(self):
        """Obtiene y devuelve todas las posiciones de trading abiertas."""
        try:
            positions = self.trading_client.get_all_positions()
            return positions
        except Exception as e:
            print(f"Error al obtener las posiciones abiertas: {e}")
            return None

    def get_market_data(self, symbol: str):
        """Devuelve la última cotización para un símbolo."""
        try:
            request_params = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
            latest_quote_dict = self.stock_data_client.get_stock_latest_quote(request_params)
            
            if latest_quote_dict and latest_quote_dict.get(symbol):
                quote_data = latest_quote_dict[symbol]
                print(f"Última cotización para {symbol}: {quote_data}")
                return quote_data
            else:
                print(f"No se encontró una cotización para {symbol}.")
                return None
        except Exception as e:
            print(f"Error al obtener datos de mercado para {symbol}: {e}")
            return None
    
    def get_pending_orders(self):
        """Obtiene y devuelve todas las órdenes que no están cerradas (pendientes)."""
        try:
            request_params = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            orders = self.trading_client.get_orders(filter=request_params)
            return orders
        except Exception as e:
            print(f"Error al obtener las órdenes pendientes: {e}")
            return None

    def cancel_order(self, order_id: str):
        """Cancela una orden específica por su ID."""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            return {"status": "success", "message": f"Solicitud de cancelación para la orden {order_id} enviada."}
        except Exception as e:
            print(f"Error al cancelar la orden {order_id}: {e}")
            return None
    
    def get_available_cash(self) -> Optional[float]:
        """Obtiene y devuelve el saldo de efectivo disponible para operar (cash)."""
        try:
            account = self.trading_client.get_account()
            if hasattr(account, 'cash') and account.cash is not None:
                # Retorna el saldo como un flotante para su uso
                return float(account.cash)
            return None
        except Exception as e:
            print(f"Error al obtener el saldo de efectivo: {e}")
            return None