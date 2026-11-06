"""Part A task 7: pandas cross-check of Q2 and exploratory statistics."""
import pandas as pd

o = pd.read_csv("data/clean/orders.csv")
c = pd.read_csv("data/clean/customers.csv")
p = pd.read_csv("data/clean/products.csv")
comp = o[o.status == "completed"].merge(c[["customer_id", "city"]], on="customer_id").merge(p[["product_id", "category"]], on="product_id")

pv = pd.pivot_table(comp, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True)
print(pv.round(2).to_string())
print("pivot grand total:", round(pv.loc["All", "All"], 2))

agg = comp.groupby("city").agg(total=("order_amount", "sum"), mean=("order_amount", "mean"), n=("order_amount", "count")).round(2)
print(agg.to_string())

x = comp["order_amount"]
q1, q3 = x.quantile(.25), x.quantile(.75); iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"mean {x.mean():.2f} median {x.median():.2f} skew {x.skew():.2f} Q1 {q1:.2f} Q3 {q3:.2f}")
print(f"fences {lo:.2f} / {hi:.2f}; outliers below {int((x < lo).sum())} above {int((x > hi).sum())}")
corr = comp[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print(corr.round(3).to_string())
pairs = corr.where(~pd.DataFrame(__import__("numpy").eye(4, dtype=bool), index=corr.index, columns=corr.columns)).abs().stack()
print("largest |corr| pair:", pairs.idxmax(), round(corr.loc[pairs.idxmax()], 3))
