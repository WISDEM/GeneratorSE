import os
import numpy as np
import pandas as pd

def copy_data(prob_in, prob_out):

    # Get all OpenMDAO inputs and outputs into a dictionary
    def create_dict(prob):
        dict_omdao = prob.model.list_inputs(val=True, hierarchical=False, prom_name=True, units=False, desc=False, out_stream=None)
        temp = prob.model.list_outputs(val=True, hierarchical=False, prom_name=True, units=False, desc=False, out_stream=None)
        dict_omdao.extend(temp)
        my_dict = {}
        for k in range(len(dict_omdao)):
            my_dict[ dict_omdao[k][1]["prom_name"] ] = dict_omdao[k][1]["val"]

        return my_dict

    in_dict = create_dict(prob_in)
    out_dict = create_dict(prob_out)

    for k in in_dict:
        if k in out_dict:
            prob_out[k] = in_dict[k]

    return prob_out

def save_data(fname, prob):
    # Remove file extension
    froot = os.path.splitext(fname)[0]

    # Get all OpenMDAO inputs and outputs into a dictionary
    var_dict = prob.model.list_inputs(prom_name=True, units=True, desc=True, out_stream=None)
    out_dict = prob.model.list_outputs(prom_name=True, units=True, desc=True, out_stream=None)
    var_dict.extend(out_dict)

    data = {}
    data["variables"] = []
    data["units"] = []
    data["values"] = []
    data["description"] = []
    for k in range(len(var_dict)):
        unit_str = var_dict[k][1]["units"]
        if unit_str is None:
            unit_str = ""

        iname = var_dict[k][1]["prom_name"]
        if iname in data["variables"]:
            continue

        data["variables"].append(iname)
        data["units"].append(unit_str)
        data["values"].append(var_dict[k][1]["val"])
        data["description"].append(var_dict[k][1]["desc"])
    df = pd.DataFrame(data)
    #df.to_excel(froot + ".xlsx", index=False)
    df.to_csv(froot + ".csv", index=False)

def load_data(fname, prob):
    # Remove file extension
    fname = os.path.splitext(fname)[0] + ".csv"

    if os.path.exists(fname):
        df = pd.read_csv(fname)

        for k in range(len(df.index)):
            key = df["variables"].iloc[k]
            if key.find("field_coil") >= 0: continue
            units = str(df["units"].iloc[k])
            val_str = df["values"].iloc[k]
            val_str_clean = val_str.replace("[","").replace("]","").strip().replace(" ", ", ")
            try:
                #print("TRY",key, val_str, val_str_clean)
                val = np.fromstring(val_str_clean, sep=",")
                if units.lower() in ["nan","unavailable"]:
                    prob.set_val(key, val)
                else:
                    prob.set_val(key, val, units=units)
            except:
                print("FAIL", key, val_str, val_str_clean)
                #breakpoint()
                continue

    return prob
