def revenue_per_order(total_revenue: float,total_orders: int) -> float:
    if n_users <= 0:
        return 0.0
    return total_revenue / total_orders

def high_value(revenue: float,threshold: int) -> float:
    if revenue >= threshold:
        return 1
    else:
        return 0
        
def multiple_purchase(orders: float,threshold: int) -> float:
    if orders >= threshold:
        return 1
    else:
        return 0
