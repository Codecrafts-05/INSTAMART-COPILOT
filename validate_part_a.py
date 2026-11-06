"""Part A task 5: assert reference checkpoints."""
import sqlite3
import pandas as pd

c = pd.read_csv("data/clean/customers.csv")
o = pd.read_csv("data/clean/orders.csv")
con = sqlite3.connect("data/instamart.db")
tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
assert {"customers", "products", "orders"} <= tables, tables
comp = o[o.status == "completed"]
canc = o[o.status == "cancelled"]

assert len(o) == 12309 and o.order_id.is_unique
assert len(comp) == 11110 and len(canc) == 1199
assert not (o.status == "pending").any()
assert o.order_amount.isna().sum() == 0
assert comp.delivery_minutes.isna().sum() == 0 and canc.delivery_minutes.isna().sum() == 1199
assert round(comp.order_amount.sum(), 2) == 2998668.80
assert c.city.value_counts().to_dict() == {"Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}
assert c.gender.value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}
assert c.age.isna().sum() == 0 and c.age.between(18, 60).all()
assert con.execute("select count(*) from orders").fetchone()[0] == 12309
print("ALL CHECKPOINTS PASSED")
