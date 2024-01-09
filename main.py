"""


Returns:
    _type_: _description_
"""

import os
import numpy as np
import pandas as pd
import argparse

from gamspy import Container
from gamspy import Model
from gamspy import Set
from gamspy import Parameter
from gamspy import Variable
from gamspy import Equation
from gamspy import Sum


def read_data():

    parser = argparse.ArgumentParser(description="Supply Chain Management")
    parser.add_argument('--nw', type=int, help='Number of warehouses (default: 1)', default=1)
    parser.add_argument('--mb', type=int, help='Budget in USD (default: $1000000)', default=1000000)
    parser.add_argument('--mc', type=int, help='Capacity in containers (1000)', default=1000)
    parser.add_argument('--out', type=str, help='Output file name', default="output.csv")


    args = parser.parse_args()

    data_path = "data"

    # Set i
    products_info = pd.read_excel(os.path.join(data_path, "Products.xlsx"), sheet_name="Sayfa1", index_col=None)
    products = [i.replace(" ", "") for i in products_info.Products.tolist()]
    set_i = products

    # Set j
    distance = pd.read_excel(os.path.join(data_path, "distance-matrix.xlsx"), sheet_name="Sayfa1", index_col=0).fillna(0)
    cities = [i.replace("İ", "i").lower() for i in distance.columns]
    distance.columns = cities
    distance.index = cities
    set_j = cities

    # Set k
    ports = pd.read_excel(os.path.join(data_path, "Limanlar.xlsx"), sheet_name="Sayfa1", index_col=None)
    ports = [i.replace("İ", "i").lower() for i in ports.PortName.to_list()]
    set_k = ports


    # Parametes

    # w
    forecasts = pd.read_csv(os.path.join(data_path, "export_forecasts.csv"), index_col=0)
    param_w = np.array(forecasts["2024"]/forecasts["2024"].max())

    # d
    param_d = distance[ports].T.unstack()

    # q
    quality = pd.read_excel(os.path.join(data_path, "Quality_ProdCap.xlsx"), sheet_name="Quality", index_col=0).fillna(0).T
    quality.columns = [i.replace("İ", "i").lower() for i in quality.columns]
    quality.index = [i.replace(" ", "").lower() for i in quality.index]
    param_q = quality[cities].T.unstack()

    # v
    param_v = np.array([int(i.split()[0]) for i in products_info["Quantity (1 Container)"]])

    # cap
    product_capacity = pd.read_excel(os.path.join(data_path, "Quality_ProdCap.xlsx"), sheet_name="ProdCap", index_col=0).fillna(0).T
    product_capacity.columns = [i.replace("İ", "i").lower() for i in product_capacity.columns]
    product_capacity.index = [i.replace(" ", "").lower() for i in product_capacity.index]
    param_cap = product_capacity[cities].T.unstack()

    # wn
    param_wn = args.nw       # Userinput

    # max_cap
    param_mc = args.mc      # Userinput

    # warehouse_cost
    warehouse_house = pd.read_excel(os.path.join(data_path, "warehouse_cost.xlsx"), sheet_name="Sheet1", index_col=None)
    param_wc = np.array(warehouse_house["cost"])*30

    # revenue
    param_revenue = np.array(products_info["Revenue (1 Container - $)"])

    # budget
    param_mb = args.mb     # Userinput

    # transportation_cost
    param_trans_cost = distance[ports].T.unstack()*2

    model_data = {
        "i": set_i,
        "j": set_j,
        "k": set_k,
        "w": param_w,
        "d": param_d,
        "q": param_q,
        "v": param_v,
        "cap": param_cap,
        "wn": param_wn,
        "mc": param_mc,
        "wc": param_wc,
        "revenue": param_revenue,
        "budget": param_mb,
        "transport_cost": param_trans_cost,
        "out": args.out
    }

    return model_data


def main():

    model_data = read_data()

    # Create a container
    m = Container()


    # Create the model sets
    i = Set(m, name='i', description='Products considered', records=model_data["i"])
    j = Set(m, name='j', description='Potential warehouse locations', records=model_data["j"])
    k = Set(m, name='k', domain=j, description='Exporting Hubs (Ports)', records=model_data["k"])


    # Create the model parameters
    w = Parameter(m, name='w', domain=i, description='Weights for product i', records=model_data["w"])
    d = Parameter(m, name='d', domain=[j, k], description='Distance from city j to exporting hub k', records=model_data["d"])
    q = Parameter(m, name='q', domain=[i, j], description='Quality of product i in city j', records=model_data["q"])
    v = Parameter(m, name='v', domain=i, description='Volume of product i', records=model_data["v"])
    p_cap = Parameter(m, name='p_cap', domain=[i, j], description='Production Capacity of product i in city j', records=model_data["cap"])
    wn = Parameter(m, name='wn', description='Maximum number of warehouses', records=model_data["wn"])
    m_cap = Parameter(m, name='m_cap', description='Maximum capacity of warehouses', records=model_data["mc"])
    wc = Parameter(m, name='wc', domain=j, description='Warehouse cost per container per city', records=model_data["wc"])
    revenue = Parameter(m, name='revenue', domain=i, description='Revenue per container per product', records=model_data["revenue"])
    m_budget = Parameter(m, name='m_budget', description='Maximum budget for warehouses', records=model_data["budget"])
    transport_cost = Parameter(m, name="transport_cost", domain=[j, k], description='Transportation cost from city j to exporting hub k', records=model_data["transport_cost"])


    # Create the model variables
    x = Variable(m, name='x', domain=j, type="binary", description='Whether to build a warehouse in city j')
    y = Variable(m, name='y', domain=[i, j], type="integer", description='Amount of product i stocked at city j')

    o1 = Variable(m, name='o1', type="free", description='Variable to take first objective function value')
    o2 = Variable(m, name='o2', type="free", description='Variable to take second objective function value')
    o3 = Variable(m, name='o3', type="free", description='Variable to take third objective function value')

    o_all = Variable(m, name='o_all', type="free", description='Variable to take the total objective function value')


    # Define the model equations
    obj1 = Equation(m, name='obj1', description='Maximize the total products value')
    obj2 = Equation(m, name='obj2', description='Minimize the total distance')
    obj3 = Equation(m, name='obj3', description='Maximize the total products quality')

    weighted_obj = Equation(m, name='weighted_obj', description='Weighted objective function')

    num_warehouses = Equation(m, name='num_warehouses', description='Number of warehouses built')

    store_if_built = Equation(m, name='store_if_built', domain=[i, j], description='Amount of product i stored at city j if a warehouse is built')

    budget = Equation(m, name='budget', description='Budget constraint')

    capacity = Equation(m, name='capacity', domain=j, description='Capacity constraint')

    ind_cap = Equation(m, name='ind_cap', domain=[i, j], description='Individual capacity constraint for product i in city j')


    # Maximize the total products value
    obj1[...] = o1 == Sum([i, j], w[i] * y[i, j])

    # Maximize the total products quality
    obj2[...] = o2 == Sum([i, j], (q[i, j] + p_cap[i, j]) * x[j])

    # Maximize the total profit
    obj3[...] = o3 == Sum([i, j], revenue[i] * y[i, j]) - Sum(j, wc[j] * x[j]) - Sum([i, j, k], transport_cost[j, k] * y[i, j])

    # Weighted objective function
    weighted_obj[...] = o_all == (1*o1) - (10000*o2) + (100000*o3)


    # CONSTRAINTS

    # Number of warehouses built
    num_warehouses[...] = Sum(j, x[j]) <= wn

    # Amount of product i stored at city j if a warehouse is built
    store_if_built[i, j] = y[i, j] <= x[j] * 1000000000000

    # Budget constraint
    budget[...] = Sum(j, x[j] * wc[j]) <= m_budget

    # Capacity constraint
    capacity[j] = Sum(i, y[i, j]) <= m_cap

    # Individual capacity constraint
    ind_cap[i, j] = y[i, j] <= p_cap[i, j]


    # Solve the model
    scm = Model(m, name="scm", equations=m.getEquations(), problem="MIP", sense="MAX", objective=o_all)
    scm.solve()


    warehouses = x.records.j[x.records.level == 1].tolist()
    results = y.pivot()[warehouses].to_dict()
    output = open(model_data["out"], "w")
    sizes = []
    for warehouse in results:
        print(f"Warehouse,{warehouse}", file=output)
        temp = 0
        for product, amount in results[warehouse].items():
            if amount > 0:
                print(f"{product},{amount}", file=output)
                temp += amount
        sizes.append(temp)
        print("", file=output)
    output.close()


    print("Problem solved..")
    print("Warehouses:")
    for idx, w in enumerate(warehouses):
        print("\t", w, "  ->  ", int(sizes[idx]))

if __name__ == '__main__':
    main()
