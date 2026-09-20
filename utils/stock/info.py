def is_bse_stock(stock_code: str) -> bool:
  """Return whether a source code belongs to the Beijing exchange."""
  bare_code = stock_code.split('.')[0]
  return bare_code.startswith(('43', '83', '87', '92'))

def is_b_stock(stock_code: str) -> bool:
  """Return whether a source code is a Shanghai/Shenzhen B share."""
  return stock_code.startswith('900') or stock_code.startswith('200')

def is_convertible_bond(stock_code: str) -> bool:
  """Return whether a broker code is a convertible bond."""
  return stock_code.startswith('11') or stock_code.startswith('12') or stock_code.startswith('13')
