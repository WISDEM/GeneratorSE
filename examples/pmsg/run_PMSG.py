import openmdao.api as om
from generatorse.ms_pmsg.ms_pmsg import PMSG_Inner_rotor_Opt
from generatorse.ms_pmsg.structural import PMSG_Inner_Rotor_Structural
from generatorse.driver.nlopt_driver import NLoptDriver
import os
import pandas as pd
import numpy as np
import platform

if platform.system().lower() == 'darwin':
    os.environ["CX_BOTTLE"] = "FEMM"
    os.environ["WINEPATH"] = "/Users/gbarter/bin/wine"
    os.environ["FEMMPATH"] = "/Users/gbarter/Library/Application Support/CrossOver/Bottles/FEMM/drive_c/femm42/bin/femm.exe"

ratings_known = [15, 17, 20, 25]
rotor_diameter = {}
rotor_diameter[15] = 242.24
rotor_diameter[17] = 257.88
rotor_diameter[20] = 279.71
rotor_diameter[25] = 312.73
rated_speed = {}
rated_speed[15] = 7.49
rated_speed[17] = 7.04
rated_speed[20] = 6.49
rated_speed[25] = 5.80
target_eff = 0.97
fsql = "log.sql"

mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file

def cleanup_femm_files(clean_dir):
    files = os.listdir(clean_dir)
    for file in files:
        if file.endswith(".ans") or file.endswith(".fem") or file.endswith(".csv"):
            os.remove(os.path.join(clean_dir, file))

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

def optimize_magnetics_design(prob_in=None, output_dir=None, cleanup_flag=True, opt_flag=False, restart_flag=True, femm_flag=False, obj_str="cost", ratingMW=17):
    if output_dir is None:
        output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    ratingMW = int(ratingMW)
    target_torque = 1e6 * ratingMW/(np.pi*rated_speed[ratingMW]/30.0)/target_eff

    # Clean run directory before the run
    if cleanup_flag:
        cleanup_femm_files(mydir)

    prob = om.Problem()
    prob.model = PMSG_Inner_rotor_Opt(magnetics_by_fea=femm_flag)

    prob.driver = NLoptDriver()
    prob.driver.options['optimizer'] = 'LN_COBYLA'
    prob.driver.options["maxiter"] = 500
    #prob.driver = om.ScipyOptimizeDriver()
    #prob.driver.options['optimizer'] = 'COBYLA' #'SLSQP' #
    #prob.driver.options["maxiter"] = 500
    #prob.driver = om.DifferentialEvolutionDriver()
    #prob.driver.options["max_gen"] = 15
    #prob.driver.options["pop_size"] = 60
    #prob.driver.options["penalty_exponent"] = 3

    recorder = om.SqliteRecorder(os.path.join(output_dir, fsql))
    prob.driver.add_recorder(recorder)
    prob.add_recorder(recorder)
    prob.driver.recording_options["excludes"] = ["*_df"]
    prob.driver.recording_options["record_constraints"] = True
    prob.driver.recording_options["record_desvars"] = True
    prob.driver.recording_options["record_objectives"] = True

    prob.model.add_design_var("g", lower=0.020, upper=0.15, ref=0.1)
    prob.model.add_design_var("h_s", lower=0.1, upper=0.4, ref=0.1)
    prob.model.add_design_var("h_yr", lower=0.15, upper=0.3, ref=0.1)
    prob.model.add_design_var("p", lower=50, upper=300, ref=100)
    prob.model.add_design_var("l_s", lower=1, upper=1.75)
    prob.model.add_design_var("N_c", lower=1, upper=7)
    prob.model.add_design_var("I_s", lower=500, upper=2000, ref=1e3)
    prob.model.add_constraint("E_p_ratio", lower=0.8, upper=1.20)
    prob.model.add_constraint("torque_ratio", lower=1.0, upper=1.2)
    prob.model.add_constraint("gen_eff", lower=0.95)

    if obj_str.lower() == 'cost':
        prob.model.add_objective("cost_total", ref=1e6)
    elif obj_str.lower() == 'mass':
        prob.model.add_objective("mass_total", ref=1e6)
    elif obj_str.lower() in ['eff','efficiency']:
        prob.model.add_objective("gen_eff", scaler=-1.0)
    else:
        print('Objective?', obj_str)

    prob.model.approx_totals(method="fd")

    prob.setup()
    # --- Design Variables ---

    print(ratingMW, obj_str, rated_speed[ratingMW], target_torque)
    if prob_in is None:
        # Specific costs
        prob["C_Cu"] = 10.3  #  https://markets.businessinsider.com/commodities/copper-price
        prob["C_Fe"] = 0.556
        prob["C_Fes"] = 0.50139
        prob["C_PM"] = 30.0
        prob["resisitivty_Cu"] = 1.724e-8  # 1.8e-8 * 1.4  # Copper resisitivty
        prob["rho_Copper"] = 8900.0  # Kg/m3 copper density
        prob["rho_Fe"] = 7700.0  # Steel density
        prob["rho_Fes"] = 7850
        prob["rho_PM"] = 8442.37093661195  # magnet density
                
        # ## Initial design variables for a PMSG designed for a 15MW turbine
        prob["B_r"] = 1.279
        prob["E"] = 2.0e11
        prob["E_p_target"] = 3300.0
        prob["Eta_target"] = 0.95
        prob["H_c"] = 1.0 #units="A/m", desc="coercivity"
        prob["I_s"] = 1945.9858772
        prob["J_s"] = 3.0
        prob["N_c"] = 2.2532984
        prob["P_Fe0e"] = 1.0
        prob["P_Fe0h"] = 4.0
        prob["R_no"] = 0.35
        prob["R_sh"] = 0.3
        prob["b_r"] = 0.200
        prob["b_s_tau_s"] = 0.45
        prob["b_st"] = 0.4

        prob["cost_adder"] = 0.0  # 700k, could maybe increase a bit at 25MW
        prob["mass_adder"] = 0.0   # 77t, could maybe increase a bit at 25MW

        prob["d_r"] = 0.500
        prob["d_s"] = 0.6
        prob["g"] = 0.01
        prob["g1"] = 9.806
        prob["h_m"] = 0.1
        prob["h_s"] = 0.15
        prob["h_s1"] = 0.010
        prob["h_s2"] = 0.010
        prob["h_sr"] = 0.02
        prob["h_ss"] = 0.02
        prob["h_yr"] = 0.125
        prob["h_ys"] = 0.05
        prob["k_fes"] = 0.95
        prob["k_sfil"] = 0.65
        prob["l_s"] = 0.6
        prob["m"] = 3  # phases
        prob["mu_0"] = np.pi * 4 * 1e-7
        prob["mu_r"] = 1.06
        prob["n_r"] = 6
        prob["n_s"] = 5
        prob["p"] = 150.0
        prob["phi"] = 90
        prob["q1"] = 1  # slots per pole per phase
        prob["r_g"] = 1.0
        prob["ratio"]= 0.95
        
        prob["t_s"] = 0.02
        prob["t_wr"] = 0.02
        prob["t_ws"] = 0.04
        prob["theta_bd"] = 0.00
        prob["theta_sh"] = 0.00
        prob["u_allow_pcent"] = 10
        prob["y_allow_pcent"] = 20
        prob["y_bd"] = 0.00
        prob["y_sh"] = 0.00
        prob["z_allow_deg"] = 0.5
        
        if restart_flag:
            prob = load_data(os.path.join(output_dir, "LTS_output"), prob)

        # Have to set these last in case we initiatlized from a different rating
        prob["P_rated"] = ratingMW * 1e6
        prob["T_rated"] = target_torque
        prob["N_nom"] = rated_speed[ratingMW]

    else:
        prob = copy_data(prob_in, prob)

    prob.model.approx_totals(method="fd")

    if opt_flag:
        prob.run_driver()
    else:
        prob.run_model()

    # Clean run directory after the run
    if cleanup_flag:
        cleanup_femm_files(mydir)

    #prob.model.list_outputs(val=True, hierarchical=True)
    print("Final solution:")
    print("E_p_ratio", prob["E_p_ratio"])
    print("gen_eff", prob["gen_eff"])
    print("N_c", prob["N_c"])
    print("l_s", prob["l_s"])
    print("Torque_ratio", prob["torque_ratio"])

    return prob

def optimize_structural_design(prob_in=None, output_dir=None, opt_flag=False):
    if output_dir is None:
        output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    prob_struct = om.Problem()
    prob_struct.model = PMSG_Inner_Rotor_Structural()

    prob_struct.driver = om.ScipyOptimizeDriver()
    prob_struct.driver.options["optimizer"] = "SLSQP"
    prob_struct.driver.options["maxiter"] = 100

    #recorder = om.SqliteRecorder("log.sql")
    #prob_struct.driver.add_recorder(recorder)
    #prob_struct.add_recorder(recorder)
    #prob_struct.driver.recording_options["excludes"] = ["*_df"]
    #prob_struct.driver.recording_options["record_constraints"] = True
    #prob_struct.driver.recording_options["record_desvars"] = True
    #prob_struct.driver.recording_options["record_objectives"] = True

    prob_struct.model.add_design_var("h_sr", lower=0.045, upper=0.25)
    prob_struct.model.add_design_var("h_ss", lower=0.045, upper=0.25)
    prob_struct.model.add_design_var("n_r", lower=5, upper=15)
    prob_struct.model.add_design_var("b_r", lower=0.1, upper=1.5)
    prob_struct.model.add_design_var("d_r", lower=0.1, upper=1.5)
    prob_struct.model.add_design_var("t_wr", lower=0.001, upper=0.2)
    prob_struct.model.add_design_var("n_s", lower=5, upper=15)
    prob_struct.model.add_design_var("b_st", lower=0.1, upper=1.5)
    prob_struct.model.add_design_var("d_s", lower=0.1, upper=1.5)
    prob_struct.model.add_design_var("t_ws", lower=0.001, upper=0.2)
    prob_struct.model.add_objective("mass_structural", ref=1e6)

    prob_struct.model.add_constraint("U_rotor_radial_constraint", upper=1.0)
    prob_struct.model.add_constraint("U_rotor_axial_constraint", upper=1.0)
    prob_struct.model.add_constraint("U_stator_radial_constraint", upper=1.0)
    prob_struct.model.add_constraint("U_stator_axial_constraint", upper=1.0)

    prob_struct.model.approx_totals(method="fd")

    prob_struct.setup()
    # --- Design Variables ---

    if prob_in is None:
        # Initial design variables for a PMSG designed for a 15MW turbine
        #prob_struct["Sigma_shear"] = 74.99692029e3
        prob_struct["Sigma_normal"] = 378.45123826e3
        prob_struct["T_e"] = 9e06
        prob_struct["l_eff_stator"] = 1.44142189  # rev 1 9.94718e6
        prob_struct["l_eff_rotor"] = 1.2827137
        prob_struct["D_a"] = 7.74736313
        prob_struct["delta_em"] = 0.0199961
        prob_struct["h_s"] = 0.1803019703
        prob_struct["D_sc"] = 7.78735533
        prob_struct["rho_steel"] = 7850
        prob_struct["rho_Fe"] = 7700
        prob_struct["Tilt_angle"] = 90.0
        prob_struct["R_shaft_outer"] = 1.25
        prob_struct["R_nose_outer"] = 0.95
        prob_struct["u_allow_pcent"] = 50
        prob_struct["y_allow_pcent"] = 20
        prob_struct["h_yr"] = 0.1254730934
        prob_struct["h_yr_s"] = 0.025
        prob_struct["h_ys"] = 0.050
        prob_struct["t_rdisc"] = 0.05
        prob_struct["t_sdisc"] = 0.1
        prob_struct["y_bd"] = 0.00
        prob_struct["theta_bd"] = 0.00
        prob_struct["y_sh"] = 0.00
        prob_struct["theta_sh"] = 0.00

        #prob_struct["mass_copper"] = 60e3
        prob_struct["mass_SC"] = 4000
    else:
        prob_struct = copy_data(prob_in, prob_struct)

    prob_struct.model.approx_totals(method="fd")

    if opt_flag:
        prob_struct.run_driver()
    else:
        prob_struct.run_model()

    return prob_struct

def write_all_data(prob, output_dir=None):
    if output_dir is None:
        output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    save_data(os.path.join(output_dir, "LTS_output"), prob)

    ratingMW = float(prob.get_val("P_rated", units="MW"))
    raw_data = [
        ["Rating",                        "P_rated",                ratingMW, "MW", ""],
        ["Armature diameter",             "D_a",                    float(prob.get_val("D_a", units="m")), "m", "(5-9)"],
        ["Field coil diameter",           "D_sc",                   float(prob.get_val("D_sc", units="m")), "m", ""],
        ["Airgap lenth",                  "delta_em",               float(prob.get_val("delta_em", units="mm")), "mm", "(60-150)"],
        ["Stack length",                  "l_s",                    float(prob.get_val("l_s", units="m")), "m", "(1-1.75)"],
        ["l_eff_rotor",                   "l_eff_rotor",            float(prob.get_val("l_eff_rotor", units="m")), "m", ""],
        ["l_eff_stator",                  "l_eff_stator",           float(prob.get_val("l_eff_stator", units="m")), "m", ""],
        ["l/d ratio",                     "K_rad",                  float(prob.get_val("K_rad")), "", ""],
        ["Alpha",                         "alpha",                  float(prob.get_val("alpha", units="deg")), "deg", "(0.1-1)"],
        ["delta Alpha",                   "dalpha",                 float(prob.get_val("dalpha", units="deg")), "deg", "(1-5)"],
        ["Beta",                          "beta",                   float(prob.get_val("beta", units="deg")), "deg", ""],
        ["Geometry constraint",           "con_angle",              float(prob.get_val("con_angle", units="deg")), "deg", ">0"],
        ["Slot_aspect_ratio",             "Slot_aspect_ratio",      float(prob.get_val("Slot_aspect_ratio")), "", ""],
        ["Pole pitch",                    "tau_p",                  float(prob.get_val("tau_p", units="mm")), "mm", ""],
        ["Slot pitch",                    "tau_s",                  float(prob.get_val("tau_s", units="mm")), "mm", ""],
        ["Stator slot height",            "h_s",                    float(prob.get_val("h_s", units="mm")), "mm", "(100-400)"],
        ["Stator slotwidth",              "b_s",                    float(prob.get_val("b_s", units="mm")), "mm", ""],
        ["Stator tooth width",            "b_t",                    float(prob.get_val("b_t", units="mm")), "mm", ""],
        ["Rotor yoke height",             "h_yr",                   float(prob.get_val("h_yr", units="mm")), "mm", "(10-450)"],
        ["Field coil height",             "h_sc",                   float(prob.get_val("h_sc", units="mm")), "mm", "(30-150)"],
        ["Field coil width",              "W_sc",                   float(prob.get_val("W_sc", units="mm")), "mm", ""],
        ["Outer width",                   "Outer_width",            float(prob.get_val("Outer_width", units="mm")), "mm", ""],
        ["Coil separation distance",      "a_m",                    float(prob.get_val("a_m", units="mm")), "mm", ""],
        ["Pole pairs",                    "p1",            float(np.round(prob.get_val("p1"))), "", "(20-30)"],
        ["Generator Terminal voltage",    "E_p",                    float(prob.get_val("E_p", units="V")), "Volts", ""],
        ["Terminal voltage target",       "E_p_target",             float(prob.get_val("E_p_target", units="V")), "Volts", ""],
        ["Terminal voltage constraint",   "E_p_ratio",              float(prob.get_val("E_p_ratio")), "", "0.8 < x < 1.2"],
        ["Stator current",                "I_s",                    float(prob.get_val("I_s", units="A")), "A", "(500-3000)"],
        ["Armature slots",                "Slots",                  float(prob.get_val("Slots")), "slots", ""],
        ["Armature turns/phase/pole",     "N_c",           float(np.round(prob.get_val("N_c"))),  "turns", "(1-15)"],
        ["Armature current density",      "J_s",                    float(prob.get_val("J_s", units="A/mm/mm")), "A/mm^2", ""],
        ["Resistance per phase",          "R_s",                    float(prob.get_val("R_s", units="ohm")), "ohm", ""],
        ["Shear stress",                  "Sigma_shear",            float(prob.get_val("Sigma_shear", units="kPa")), "kPa", ""],
        ["Normal stress",                 "Sigma_normal",           float(prob.get_val("Sigma_normal", units="kPa")), "kPa", ""],
        ["Torque",                        "Torque_actual",          float(prob.get_val("Torque_actual", units="MN*m")), "MNm", ""],
        ["Torque rated target",           "T_rated",                float(prob.get_val("T_rated", units="MN*m")), "MNm", ""],
        ["Torque constraint",             "torque_ratio",           float(prob.get_val("torque_ratio")), "", "1.0 < x < 1.2"],
        ["Field coil turns",              "N_sc",          float(np.round(prob.get_val("N_sc"))),  "turns", "(1500-3500)"],
        ["Field coil current in",         "I_sc_in",                float(prob.get_val("I_sc_in", units="A")), "A", "(150-700)"],
        ["Field coil current out",        "I_sc_out",               float(prob.get_val("I_sc_out", units="A")), "A", ""],
        ["Critical current load margin",  "load_margin",            float(prob.get_val("load_margin")), "", "(0.85-0.95)"],
        #["Critical current constraint",   "Critical_current_ratio", float(prob.get_val("Critical_current_ratio")), "", "<1.2"],
        ["Layer count",                   "N_l",                    float(prob.get_val("N_l")), "layers", ""],
        ["Turns per layer",               "N_sc_layer",             float(prob.get_val("N_sc_layer")), "turns", ""],
        ["length per racetrack",          "l_sc",                   float(prob.get_val("l_sc", units="km")), "km", ""],
        ["Mass per coil",                 "mass_SC_racetrack",      float(prob.get_val("mass_SC_racetrack", units="kg")), "kg", ""],
        ["Total mass of SC coils",        "mass_SC",                float(prob.get_val("mass_SC", units="t")), "Tons", ""],
        ["B_rymax",                       "B_rymax",                float(prob.get_val("B_rymax", units="T")), "Tesla", "<2.1"],
        ["B_g",                           "B_g",                    float(prob.get_val("B_g", units="T")), "Tesla", ""],
        ["B_coil_max",                    "B_coil_max",             float(prob.get_val("B_coil_max", units="T")), "Tesla", "6<"],
        ["B_g - Coil max constraint",     "constr_B_g_coil",        float(prob.get_val("constr_B_g_coil")), "", "<1.0"],
        ["Coil max ratio",                "Coil_max_ratio",         float(prob.get_val("Coil_max_ratio")), "", "<1.2"],
        ["Copper mass",                   "mass_copper",            float(prob.get_val("mass_copper", units="t")), "Tons", ""],
        ["Iron mass",                     "mass_iron",              float(prob.get_val("mass_iron", units="t")), "Tons", ""],
        ["Total active mass",             "mass_active",            float(prob.get_val("mass_active", units="t")), "Tons", ""],
        ["Efficiency",                    "gen_eff",                float(prob.get_val("gen_eff"))*100, "%", "97<="],
        ["Rotor disc thickness",          "t_rdisc",                float(prob.get_val("t_rdisc", units="mm")), "mm", "(25-500)"],
        ["Rotor yoke thickness",          "h_yr_s",                 float(prob.get_val("h_yr_s", units="mm")), "mm", "(25-500)"],
        ["Stator disc thickness",         "t_sdisc",                float(prob.get_val("t_sdisc", units="mm")), "mm", "(25-500)"],
        ["Stator yoke thickness",         "h_ys",                   float(prob.get_val("h_ys", units="mm")), "mm", "(25-500)"],
        ["Rotor radial deflection",       "u_ar",                   float(prob.get_val("u_ar", units="mm")), "mm", ""],
        ["Rotor radial limit",            "u_allowable_r",          float(prob.get_val("u_allowable_r", units="mm")), "mm", ""],
        ["Rotor radial constraint",       "U_rotor_radial_constraint", float(prob.get_val("U_rotor_radial_constraint")), "", "<1"],
        ["Rotor axial deflection",        "y_ar",                   float(prob.get_val("y_ar", units="mm")), "mm", ""],
        ["Rotor axial limit",             "y_allowable_r",          float(prob.get_val("y_allowable_r", units="mm")), "mm", ""],
        ["Rotor axial constraint",        "U_rotor_axial_constraint", float(prob.get_val("U_rotor_axial_constraint")), "", "<1"],
        ["Rotor torsional twist",         "twist_r",                float(prob.get_val("twist_r", units="deg")), "deg", ""],
        ["Stator radial deflection",      "u_as",                   float(prob.get_val("u_as", units="mm")), "mm", ""],
        ["Stator radial limit",           "u_allowable_s",          float(prob.get_val("u_allowable_s", units="mm")), "mm", ""],
        ["Stator radial constraint",      "U_stator_radial_constraint", float(prob.get_val("U_stator_radial_constraint")), "", "<1"],
        ["Stator axial deflection",       "y_as",                   float(prob.get_val("y_as", units="mm")), "mm", ""],
        ["Stator axial limit",            "y_allowable_s",          float(prob.get_val("y_allowable_s", units="mm")), "mm", ""],
        ["Stator axial constraint",       "U_stator_axial_constraint", float(prob.get_val("U_stator_axial_constraint")), "", "<1"],
        ["Stator torsional twist",        "twist_s",                float(prob.get_val("twist_s", units="deg")), "deg", ""],
        ["Rotor structural mass",         "Structural_mass_rotor",  float(prob.get_val("Structural_mass_rotor", units="t")), "tons", ""],
        ["Stator structural mass",        "Structural_mass_stator", float(prob.get_val("Structural_mass_stator", units="t")), "tons", ""],
        ["Total structural mass",         "mass_structural",        float(prob.get_val("mass_structural", units="t")), "tons", ""],
        ["Total generator mass",          "mass_total",             float(prob.get_val("mass_total", units="t")), "tons", ""],
        ["Total generator cost",          "cost_total",             float(prob.get_val("cost_total", units="USD"))/1000., "k$", ""]
    ]

    df = pd.DataFrame(raw_data, columns=["Parameters", "Symbol", "Values", "Units", "Limit"])
    df.to_excel(os.path.join(output_dir, f"Optimized_LTSG_{ratingMW}_MW.xlsx"))

def run_all(output_str, opt_flag, obj_str, ratingMW):
    output_dir = os.path.join(mydir, output_str)

    # Optimize just magnetrics with GA and then structural with SLSQP
    prob = optimize_magnetics_design(output_dir=output_dir, opt_flag=opt_flag, obj_str=obj_str, ratingMW=int(ratingMW), restart_flag=True)
    prob_struct = optimize_structural_design(prob_in=prob, output_dir=output_dir, opt_flag=opt_flag)

    # Bring all data together
    prob["h_sr"] = prob_struct["h_sr"]
    prob["h_ss"] = prob_struct["h_ss"]
    prob["n_r"]  = prob_struct["n_r"]
    prob["n_s"]  = prob_struct["n_s"]
    prob["b_r"]  = prob_struct["b_r"]
    prob["d_r"]  = prob_struct["d_r"]
    prob["t_wr"] = prob_struct["t_wr"]
    prob["t_ws"] = prob_struct["t_ws"]
    prob["b_st"] = prob_struct["b_st"]
    prob["d_s"]  = prob_struct["d_s"]
    prob.run_model()

    # Write to xlsx and csv files
    write_all_data(prob, output_dir=output_dir)
    prob.model.list_outputs(val=True, hierarchical=True)
    cleanup_femm_files(mydir)

if __name__ == "__main__":
    opt_flag = False
    for k in ratings_known:
        for obj in ["cost", "mass"]:
            run_all(f"outputs{k}-{obj}", opt_flag, obj, k)
