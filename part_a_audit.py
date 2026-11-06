"""Part A task 2: audit the raw tables."""
import pandas as pd

RAW = "data/raw/"
cust = pd.read_csv(RAW + "customers.csv", dtype=str, keep_default_na=False)
orders = pd.read_csv(RAW + "orders.csv", dtype=str, keep_default_na=False)
prod = pd.read_csv(RAW + "products.csv")

# per-table shape / dtypes / missing (read normally for dtypes and NaN counts)
for name in ["customers", "products", "orders"]:
    df = pd.read_csv(RAW + f"{name}.csv")
    print(f"\n== {name}: shape={df.shape}")
    rep = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(),
                        "missing_pct": (df.isna().mean() * 100).round(2)})
    print(rep.to_string())

amt_num = pd.to_numeric(orders["order_amount"], errors="coerce")
dm = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
age = pd.to_numeric(cust["age"], errors="coerce")
audit = {
    "raw orders rows": len(orders),
    "exact duplicate rows": int(orders.duplicated().sum()),
    "pending rows": int((orders["status"] == "pending").sum()),
    "distinct order_id": orders["order_id"].nunique(),
    "non-numeric order_amount": int(amt_num.isna().sum()),
    "negative delivery_minutes": int((dm < 0).sum()),
    "blank delivery_minutes": int((orders["delivery_minutes"] == "").sum()),
    "blank age": int((cust["age"] == "").sum()),
    "age outside 10-100": int(((age < 10) | (age > 100)).sum()),
    "distinct raw city spellings": cust["city"].nunique(),
    "distinct non-blank gender spellings": cust.loc[cust["gender"] != "", "gender"].nunique(),
    "blank gender": int((cust["gender"] == "").sum()),
}
print("\n== audit figures")
for k, v in audit.items():
    print(f"{k}: {v}")
