from datetime import datetime, timezone
from google.adk.tools.tool_context import ToolContext
from typing import Dict, Any, List, Optional
from trading_utils.alpaca_adapter import AlpacaAPI 
from alpaca.trading.models import Order, Position
from alpaca.data.models import Quote
from alpaca.trading.enums import OrderStatus


async def get_open_positions(tool_context: ToolContext) -> Dict[str, Any]:
    """
    This tool gets the user's open positions from their portfolio.
    
    Returns:
        A dictionary with all the open positions. The 'open_positions' key contains
        a list where each position includes the symbol, quantity, current price, cost basis,
        unrealized profit/loss, and the total market_value of the holding.
    """
    alpaca_api = AlpacaAPI()
    try:
        open_positions_raw = alpaca_api.get_open_positions()
        if not isinstance(open_positions_raw, list) or not all(isinstance(pos, Position) for pos in open_positions_raw):
            tool_context.state["open_positions"] = []
            return {"status": "error", "message": "No se pudieron obtener las posiciones abiertas correctamente."}
        open_positions: List[Position] = open_positions_raw

        if not open_positions:
            tool_context.state["open_positions"] = []
            return {"status": "success", "message": "No tienes posiciones abiertas."}

        positions_data = []
        for pos in open_positions:
            positions_data.append({
                "symbol": pos.symbol,
                "qty": float(pos.qty),
                "market_value": float(pos.market_value),
                "cost_basis": float(pos.cost_basis),
                "unrealized_pl": float(pos.unrealized_pl),
                "current_price": float(pos.current_price)
            })

        tool_context.state["open_positions"] = positions_data
        
        return {
            "status": "success",
            "message": f"Se han obtenido {len(positions_data)} posiciones abiertas.",
            "open_positions": positions_data
        }
    except Exception as e:
        return {"status": "error", "message": f"Error al obtener las posiciones abiertas: {e}"}
        
def sell_order(
    symbol: str, 
    entry_price: float,
    predicted_buy_price: float,
    predicted_sell_price: float,
    tool_context: ToolContext,
    qty: Optional[float] = None,
    notional: Optional[float] = None
) -> Dict[str, Any]:
    """
    Submit a stock sell order by specifying either a quantity of shares or a notional (dollar) amount.

    Args:
        symbol: The stock ticker, e.g., "AAPL", "GOOG".
        qty: The quantity of shares to sell. Use this OR notional.
        notional: The dollar amount to sell. Use this OR qty.
        entry_price: The actual price at which the position was opened.
        predicted_buy_price: The price predicted by the model at the time of purchase.
        predicted_sell_price: The price predicted by the model at the time of sale.
        tool_context: The tool context object, passed automatically by the agent.
    """
    alpaca_api = AlpacaAPI()
    try:
        api_response: Order = alpaca_api.submit_order(symbol=symbol, side="sell", qty=qty, notional=notional)
        if api_response is None:
            return {
                "status": "error",
                "message": "La llamada a la API de Alpaca falló. Por favor, revisa el log de la consola para más detalles."
            }
        
        exit_price = 0.0
        profit_loss = 0.0
        
        if api_response.status in [OrderStatus.FILLED, OrderStatus.ACCEPTED, OrderStatus.PENDING_NEW]:
            if api_response.filled_avg_price is not None:
                exit_price = float(api_response.filled_avg_price)
            else:
                market_data: Quote = alpaca_api.get_market_data(symbol)
                if market_data:
                    exit_price = market_data.bid_price # Use bid price for sells
            
            if exit_price > 0:
                order_qty = float(api_response.filled_qty) if api_response.filled_qty else qty
                if order_qty:
                    profit_loss = (exit_price - entry_price) * order_qty

            closed_position = {
                "symbol": symbol, "status": str(api_response.status.value), "order_id": str(api_response.id),
                "client_order_id": api_response.client_order_id, "qty": qty or notional, "entry_price": entry_price,
                "exit_price": exit_price, "profit_loss": profit_loss, "exit_date": datetime.now(timezone.utc).isoformat(),
                "predicted_buy_price": predicted_buy_price, "predicted_sell_price": predicted_sell_price,
            }

            closed_positions_list = tool_context.state.get("closed_positions", [])
            closed_positions_list.append(closed_position)
            tool_context.state["closed_positions"] = closed_positions_list
            message = f"La orden de venta para {symbol} ha sido completada. La posición fue cerrada con una ganancia/pérdida de {profit_loss:.2f}."
        else:
            message = f"La orden de venta para {symbol} ha sido enviada con el estado: {api_response.status.value}."

        return {
            "status": str(api_response.status.value), "order_id": str(api_response.id),
            "client_order_id": api_response.client_order_id, "message": message, "entry_price": entry_price,
            "exit_price": exit_price, "profit_loss": profit_loss, "exit_date": datetime.now(timezone.utc).isoformat(),
            "filled_avg_price": float(api_response.filled_avg_price) if api_response.filled_avg_price else None,
            "filled_qty": float(api_response.filled_qty) if api_response.filled_qty else None,
            "predicted_buy_price": predicted_buy_price, "predicted_sell_price": predicted_sell_price,
        }

    except Exception as e:
        return {"status": "error", "message": f"Hubo un problema al enviar la orden de venta: {e}"}


def buy_order(
    symbol: str, 
    predicted_buy_price: float,
    tool_context: ToolContext,
    qty: Optional[float] = None,
    notional: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Submit a stock buy order by specifying either a quantity of shares or a notional (dollar) amount.

    Args:
        symbol: The stock ticker, e.g., "AAPL", "GOOG".
        qty: The quantity of shares to buy. Use this OR notional.
        notional: The dollar amount to invest. Use this OR qty.
        predicted_buy_price: The predicted buy price for the stock.
        tool_context: The tool context object, passed automatically by the agent.
    """
    alpaca_api = AlpacaAPI()
    try:
        api_response: Order = alpaca_api.submit_order(symbol=symbol, side="buy", qty=qty, notional=notional)
        if api_response is None:
            return {"status": "error", "message": "La llamada a la API de Alpaca falló."}
        
        estimated_price = None
        if api_response.filled_avg_price is None:
            market_data: Quote = alpaca_api.get_market_data(symbol)
            if market_data:
                estimated_price = market_data.ask_price
        
        filled_price = float(api_response.filled_avg_price) if api_response.filled_avg_price else estimated_price
        
        new_order = {
            "status": str(api_response.status.value), "order_id": str(api_response.id),
            "message": f"La orden de compra para {qty or notional} de {symbol} ha sido enviada con estado: {api_response.status.value}.",
            "client_order_id": api_response.client_order_id, "created_at": api_response.created_at.isoformat() if api_response.created_at else None,
            "filled_avg_price": filled_price, "filled_qty": float(api_response.filled_qty) if api_response.filled_qty else None,
            "filled_at": api_response.filled_at.isoformat() if api_response.filled_at else None,
            "symbol": symbol, "qty": qty, "notional": notional, "predicted_buy_price": predicted_buy_price,
        }
        orders_list = tool_context.state.get("orders", [])
        orders_list.append(new_order)
        tool_context.state["orders"] = orders_list

        return {
            "status": str(api_response.status.value), "order_id": str(api_response.id),
            "message": f"La orden de compra para {qty or notional} de {symbol} ha sido enviada con estado: {api_response.status.value}.",
            "client_order_id": api_response.client_order_id, "created_at": api_response.created_at.isoformat() if api_response.created_at else None,
            "filled_avg_price": filled_price, "filled_qty": float(api_response.filled_qty) if api_response.filled_qty else None,
            "filled_at": api_response.filled_at.isoformat() if api_response.filled_at else None,
            "symbol": symbol, "qty": qty, "notional": notional, "predicted_buy_price": predicted_buy_price,
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Hubo un problema al enviar la orden de compra: {e}"}
        

def cancel_orders(order_ids: List[str]) -> Dict[str, Any]:
    """
    Cancels one or more pending trading orders.

    Args:
        order_ids: A list of order IDs to be canceled. For a single order,
                   it should be a list with one element (e.g., ["some-id"]).
    """
    # Ya no necesitamos comprobar el tipo, asumimos que siempre es una lista.
    ids_to_cancel = order_ids

    alpaca_api = AlpacaAPI()
    successful_cancellations = []
    failed_cancellations = []

    for order_id in ids_to_cancel:
        try:
            cancellation_response = alpaca_api.cancel_order(order_id)
            if cancellation_response and cancellation_response["status"] == "success":
                successful_cancellations.append(order_id)
            else:
                failed_cancellations.append(order_id)
        except Exception:
            failed_cancellations.append(order_id)

    # El resto de la lógica de resumen se mantiene igual
    total_requested = len(ids_to_cancel)
    total_success = len(successful_cancellations)
    total_failed = len(failed_cancellations)
    
    message = f"Resumen de la cancelación: {total_success} de {total_requested} órdenes canceladas con éxito."
    if total_failed > 0:
        message += f" {total_failed} fallaron."

    return {
        "status": "partial_success" if total_failed > 0 and total_success > 0 else "success" if total_failed == 0 else "error",
        "message": message,
        "successful_ids": successful_cancellations,
        "failed_ids": failed_cancellations
    }

def list_pending_orders() -> Dict[str, Any]:
    """
    Use this tool to get a list of all trading orders that are currently open
    (e.g., 'accepted', 'new') and have not yet been filled. These are the orders
    that can be canceled.
    """
    alpaca_api = AlpacaAPI()
    try:
        pending_orders: List[Order] = alpaca_api.get_pending_orders()

        if not pending_orders:
            return {"status": "success", "message": "No hay ninguna orden pendiente para cancelar."}

        orders_data = []
        for order in pending_orders:
            orders_data.append({
                "order_id": str(order.id),
                "symbol": order.symbol,
                "qty": float(order.qty),
                "side": order.side.value,
                "status": order.status.value,
                "submitted_at": order.submitted_at.isoformat()
            })
        
        return {
            "status": "success",
            "message": f"Se han encontrado {len(orders_data)} órdenes pendientes.",
            "pending_orders": orders_data
        }
    except Exception as e:
        return {"status": "error", "message": f"Error al obtener las órdenes pendientes: {e}"}







