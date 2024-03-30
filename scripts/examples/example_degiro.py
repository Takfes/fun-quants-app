import os

from degiro_connector.trading.api import API
from degiro_connector.trading.models.trading_pb2 import Credentials, TransactionsHistory
from dotenv import load_dotenv

load_dotenv()

# SETUP TRADING API
credentials = Credentials(
    username=os.getenv("DEGIRO_USERNAME"),
    password=os.getenv("DEGIRO_PASSWORD"),
    int_account=int(os.getenv("DEGIRO_INT_ACCOUNT")),
)
trading_api = API(credentials=credentials)

# ESTABLISH CONNECTION
trading_api.connect()

# EXPLORE METHODS
[x for x in dir(trading_api) if not x.startswith("_")]

trading_api.get_account_info()

trading_api.product_search()

# TRANSACTION HISTORY
from_date = TransactionsHistory.Request.Date(year=2024, month=11, day=1)
to_date = TransactionsHistory.Request.Date(year=2024, month=6, day=15)
request = TransactionsHistory.Request(from_date=from_date, to_date=to_date)
transactions_history = trading_api.get_transactions_history(request=request)


# PRODUCT_ISIN = "FR0000131906"
PRODUCT_ISIN = "US23804L1035"

company_profile = trading_api.get_company_profile(
    product_isin=PRODUCT_ISIN,
)

company_ratios = trading_api.get_company_ratios(
    product_isin=PRODUCT_ISIN,
)

estimates_summaries = trading_api.get_estimates_summaries(
    product_isin=PRODUCT_ISIN,
    raw=False,
)

# DESTROY CONNECTION
trading_api.logout()
