"""Part A tasks 3-5: clean customers and orders, write CSVs and SQLite database."""
import os, sqlite3
import numpy as np
import pandas as pd

RAW, CLEAN = "data/raw/", "data/clean/"
os.makedirs(CLEAN, exist_ok=True)

# ---------------- customers ----------------
c = pd.read_csv(RAW + "customers.csv", keep_default_na=False)
c["city"] = c["city"].str.strip().str.title()
g = c["gender"].str.strip().str.lower()
c["gender"] = np.select([g.isin(["m", "male"]), g.isin(["f", "female"])], ["male", "female"], "unknown")
c["age"] = pd.to_numeric(c["age"], errors="coerce")
n_bad_age = int(((c["age"] < 10) | (c["age"] > 100)).sum())
c.loc[(c["age"] < 10) | (c["age"] > 100), "age"] = np.nan
n_missing_age = int(c["age"].isna().sum())
c["age"] = c["age"].fillna(c.groupby("city")["age"].transform("median")).round().astype(int)
c["signup_date"] = pd.to_datetime(c["signup_date"])
print("cities:", c["city"].value_counts().to_dict())
print("genders:", c["gender"].value_counts().to_dict())
print(f"ages set missing (out of range): {n_bad_age}; total filled: {n_missing_age}; age range {c['age'].min()}-{c['age'].max()}")

# ---------------- orders ----------------
o = pd.read_csv(RAW + "orders.csv", keep_default_na=False, dtype=str)
p = pd.read_csv(RAW + "products.csv")
steps = {}
o["order_ts"] = pd.to_datetime(o["order_ts"])                        # rule 1
n0 = len(o); o = o.drop_duplicates(); steps["exact duplicates removed"] = n0 - len(o)   # rule 2
n0 = len(o)
o = o.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last")    # rule 3
steps["superseded (earlier pending) records removed"] = n0 - len(o)
for col in ["quantity", "discount_pct"]:
    o[col] = pd.to_numeric(o[col])
o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")                   # rule 4
o = o.merge(p[["product_id", "category", "price_inr"]], on="product_id", how="left")
miss = o["order_amount"].isna()
steps["order_amount filled from price"] = int(miss.sum())
o.loc[miss, "order_amount"] = (o.loc[miss, "price_inr"] * o.loc[miss, "quantity"] * (1 - o.loc[miss, "discount_pct"] / 100)).round(2)
o["delivery_minutes"] = pd.to_numeric(o["delivery_minutes"], errors="coerce")          # rule 5
neg = o["delivery_minutes"] < 0
steps["negative delivery_minutes set missing"] = int(neg.sum())
o.loc[neg, "delivery_minutes"] = np.nan
comp = o["status"] == "completed"
cat_med = o["delivery_minutes"].where(comp).groupby(o["category"]).transform("median")
fill = comp & o["delivery_minutes"].isna()
steps["completed delivery_minutes imputed (category median)"] = int(fill.sum())
o.loc[fill, "delivery_minutes"] = cat_med[fill]
o = o.drop(columns=["category", "price_inr"]).sort_values("order_id").reset_index(drop=True)
for k, v in steps.items():
    print(f"{k}: {v}")
print("rows:", len(o), o["status"].value_counts().to_dict())
print("missing amount:", int(o["order_amount"].isna().sum()),
      "| missing delivery completed:", int(o.loc[o.status == "completed", "delivery_minutes"].isna().sum()),
      "| cancelled:", int(o.loc[o.status == "cancelled", "delivery_minutes"].isna().sum()))
print(f"completed revenue: {o.loc[o.status == 'completed', 'order_amount'].sum():,.2f}")

# ---------------- store ----------------
o_out = o.copy(); o_out["order_ts"] = o_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
c_out = c.copy(); c_out["signup_date"] = c_out["signup_date"].dt.strftime("%Y-%m-%d")
c_out.to_csv(CLEAN + "customers.csv", index=False)
p.to_csv(CLEAN + "products.csv", index=False)
o_out.to_csv(CLEAN + "orders.csv", index=False)
if os.path.exists("data/instamart.db"):
    os.remove("data/instamart.db")
with sqlite3.connect("data/instamart.db") as con:
    c_out.to_sql("customers", con, index=False)
    p.to_sql("products", con, index=False)
    o_out.to_sql("orders", con, index=False)
print("wrote data/clean/*.csv and data/instamart.db")
