"""Input contract. Post-event outcomes are deliberately forbidden."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    transaction_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    event_time: float = Field(ge=0, le=4102444800)
    account_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    device_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    amount: float = Field(gt=0, le=100000)
    account_age_days: float = Field(ge=0, le=36500)
    email_age_days: float = Field(ge=0, le=36500)
    geo_distance_km: float = Field(ge=0, le=20040)
    failed_logins_24h: int = Field(ge=0, le=1000)
    cross_border: int = Field(ge=0, le=1)
    channel: Literal["web", "app"]
    product: Literal["retail", "digital", "travel", "subscription"]
    device_type: Literal["desktop", "mobile", "tablet"]

RAW_COLUMNS = list(Transaction.model_fields)
NUMERIC = ["log_amount", "account_age_days", "email_age_days", "geo_distance_km",
           "failed_logins_24h", "cross_border", "hour_sin", "hour_cos", "account_txn_1h",
           "account_txn_24h", "amount_ratio_7d", "device_accounts_24h", "new_device_7d",
           "account_txn_7d"]
CATEGORICAL = ["channel", "product", "device_type"]
FEATURES = NUMERIC + CATEGORICAL
FEATURE_VERSION = "pit-v1"

