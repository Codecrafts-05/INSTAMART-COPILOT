"""Part A task 6: run the seven queries in sql/queries.sql and print the stated results."""
import re, sqlite3
import pandas as pd

text = open("sql/queries.sql", encoding="utf-8").read()
queries = [q.strip() for q in re.split(r"^-- Q\d:.*$", text, flags=re.M) if q.strip()]
assert len(queries) == 7
con = sqlite3.connect("data/instamart.db")
R = {}
for i, q in enumerate(queries, 1):
    R[i] = pd.read_sql_query(q, con)

print("Q1 customers:", len(R[1]))
q2 = R[2]
print("Q2 rows:", len(q2), "| total:", round(q2.revenue.sum(), 2), "| max cell:", q2.iloc[0].tolist())
print("Q3 products:", len(R[3]))
print("Q4 customers without completed order:", len(R[4]))
print("Q5 customers above average:", len(R[5]), "| average:", R[5].avg_total.iloc[0])
q6 = R[6]
print("Q6 top per city:")
print(q6[q6.row_num == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
print("Q6 rows where RANK != ROW_NUMBER:", int((q6.rnk != q6.row_num).sum()))
print("Q7 daily rows:", len(R[7]), "| last cumulative:", R[7].cumulative_revenue.iloc[-1])
