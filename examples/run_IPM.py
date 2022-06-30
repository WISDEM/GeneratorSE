import openmdao.api as om
from generatorse.ipm.IPM import PMSG_Outer_rotor_Opt
from generatorse.ipm.structural import PMSG_Outer_Rotor_Structural
import os
import pandas as pd
import numpy as np
import platform

if platform.system().lower() == "darwin":
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
                breakpoint()
                continue

    return prob

def optimize_magnetics_design(prob_in=None, output_dir=None, cleanup_flag=True, opt_flag=False, restart_flag=True, obj_str="cost", ratingMW=17):
    if output_dir is None:
        output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    ratingMW = int(ratingMW)
    target_torque = 1e6 * ratingMW/(np.pi*rated_speed[ratingMW]/30.0)/target_eff

    # Clean run directory before the run
    if cleanup_flag:
        cleanup_femm_files(mydir)

    prob = om.Problem()
    prob.model = PMSG_Outer_rotor_Opt()

    #prob.driver = om.ScipyOptimizeDriver()
    #prob.driver.options["optimizer"] = "COBYLA" #"SLSQP" #
    #prob.driver.options["maxiter"] = 500 #50
    prob.driver = om.DifferentialEvolutionDriver()
    prob.driver.options["max_gen"] = 20
    prob.driver.options["pop_size"] = 60
    prob.driver.options["penalty_exponent"] = 3

    recorder = om.SqliteRecorder(os.path.join(output_dir, fsql))
    prob.driver.add_recorder(recorder)
    prob.add_recorder(recorder)
    prob.driver.recording_options["excludes"] = ["*_df"]
    prob.driver.recording_options["record_constraints"] = True
    prob.driver.recording_options["record_desvars"] = True
    prob.driver.recording_options["record_objectives"] = True

    prob.model.add_design_var("D_a", lower=6, upper=9., ref=8.0 )
    prob.model.add_design_var("g", lower=0.007, upper=0.015, ref=0.01)
    prob.model.add_design_var("l_s", lower=1.5, upper=2.5, ref=2.0)
    prob.model.add_design_var("h_t", lower=0.02, upper=0.5, ref=0.1)
    prob.model.add_design_var("h_ys", lower=0.02, upper=0.2, ref=0.1)
    prob.model.add_design_var("h_yr", lower=0.02, upper=0.2, ref=0.1)
    prob.model.add_design_var("h_m", lower=0.005, upper=0.05, ref=0.01)
    # prob.model.add_design_var("b_t", lower=0.02, upper=0.5, ref=0.1)
    prob.model.add_design_var("pp", lower=70, upper=260, ref=100.0)
    # prob.model.add_design_var("alpha_v",lower=60, upper=160, ref=100)
    prob.model.add_design_var("N_c", lower=2, upper=10, ref=10)
    prob.model.add_design_var("I_s",lower=2900, upper=8500, ref=5000)
    prob.model.add_design_var("d_mag",lower=0.05, upper=0.25, ref=0.1)
    # prob.model.add_design_var("d_sep",lower=0.00, upper=0.020, ref=0.01)
    # prob.model.add_design_var("m_sep", lower=0.005, upper=0.01, ref=0.01)
    prob.model.add_design_var("magnet_l_pc",lower=0.7, upper=1.0)
    # prob.model.add_design_var("J_s",lower=3, upper=10)

    prob.model.add_constraint("B_rymax", upper=2.53)
    # prob.model.add_constraint("K_rad",    lower=0.15, upper=0.3)
    # prob.model.add_constraint("E_p", lower=0.9 * 3300, ref=3000)
    prob.model.add_constraint("E_p_ratio", upper=1.1)
    prob.model.add_constraint("torque_ratio", lower=0.97)
    prob.model.add_constraint("r_outer_active", upper=9.25 / 2.)
    # prob.model.add_constraint("T_e", upper=1.05*target_torque, ref=20e6)

    if not obj_str.lower() in ["eff","efficiency"]:
        prob.model.add_constraint("gen_eff", lower=0.955)

    if obj_str.lower() == "cost":
        prob.model.add_objective("cost_total", ref=1e6)
    elif obj_str.lower() == "mass":
        prob.model.add_objective("mass_total", ref=1e6)
    elif obj_str.lower() in ["eff","efficiency"]:
        prob.model.add_objective("gen_eff", scaler=-1.0)

    prob.model.approx_totals(method="fd")

    prob.setup()
    print(ratingMW, obj_str, rated_speed[ratingMW], target_torque)

    if prob_in is None:
        # Initial design variables for a PMSG designed for a 15MW turbine
        #prob["P_rated"]        =   17000000.0
        #prob["T_rated"]        =   23.03066
        #prob["N_nom"]          =   7.7
        prob["E_p_target"]     = 3300.0

        # These are the current design variables
        prob["D_a"]            = 7.5
        prob["l_s"]            = 2. #2.5
        prob["h_t"]            = 0.28631632 #0.2
        # prob["b_t"]            = 0.26488789
        prob["pp"]             = 80
        prob["g"]              = 0.0075153
        prob["N_c"]            = 6. #3.22798541 #6
        prob["I_s"]            = 10566.22104108 # 3500
        prob["h_m"]            = 0.020 #0.015
        prob["d_mag"]          = 0.03870702 #0.08

        prob["h_ys"]           = 0.025 # 0.05
        prob["h_yr"]           = 0.025
        prob["magnet_l_pc"]    = 1.0
        prob["J_s"]            = 6
        prob["phi"]            = 90
        prob["b"]              = 2.
        prob["c"]              = 5.0

        #Specific costs
        prob["C_Cu"]           = 10.3
        prob["C_Fe"]           = 1.02
        prob["C_Fes"]          = 1.0
        prob["C_PM"]           = 110.0

        #Material properties
        prob["rho_Fe"]         = 7700.0                 #Steel density
        prob["rho_Fes"]        = 7850.0                 #Steel density
        prob["rho_Copper"]     = 8900.0                  # Kg/m3 copper density
        prob["rho_PM"]         = 7600.0                  # magnet density
        prob["resistivity_Cu"] = 1.8*10**(-8)*1.4			# Copper resisitivty

        #Support structure parameters
        prob["R_no"]           = 0.925
        prob["R_sh"]           = 1.25
        prob["t_r"]            = 0.8369078 #  0.06
        prob["h_sr"]           = 0.04
        prob["t_s"]            = 0.07264606 #  0.06
        prob["h_ss"]           = 0.04
        prob["y_sh"]           = 0.0
        prob["theta_sh"]       = 0.0
        prob["y_bd"]           = 0.0
        prob["theta_bd"]       = 0.0
        prob["u_allow_pcent"]  = 8.5       # % radial deflection
        prob["y_allow_pcent"]  = 1.0       # % axial deflection
        prob["z_allow_deg"]    = 0.05       # torsional twist

        if restart_flag:
            prob = load_data(os.path.join(output_dir, "IPM_output"), prob)

        # Have to set these last in case we initiatlized from a different rating
        prob["P_rated"] = ratingMW * 1e6
        prob["T_rated"] = target_torque
        prob["N_nom"] = rated_speed[ratingMW]
        #if obj_str.lower() == "cost":
        #    prob["D_a"] = 9.0
        #elif obj_str.lower() == "mass":
        #    prob["D_a"] = 5.0
        #else:
        #    prob["D_a"] = 7.0

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
    print("l_s", prob["l_s"])
    print("Torque ratio", prob["torque_ratio"])

    return prob

def optimize_structural_design(prob_in=None, output_dir=None, opt_flag=False):
    if output_dir is None:
        output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    prob_struct = om.Problem()
    prob_struct.model = PMSG_Outer_Rotor_Structural()

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

    # prob_struct.model.add_design_var("h_yr", lower=0.025, upper=0.15, ref=0.1)
    # prob_struct.model.add_design_var("h_ys", lower=0.025, upper=0.15, ref=0.1)
    prob_struct.model.add_design_var("t_r", lower=0.05, upper=0.3, ref=0.1)
    prob_struct.model.add_design_var("t_s", lower=0.05, upper=0.3, ref=0.1)
    prob_struct.model.add_design_var("h_ss", lower=0.04, upper=0.2, ref=0.1)
    prob_struct.model.add_design_var("h_sr", lower=0.04, upper=0.2, ref=0.1)

    prob_struct.model.add_objective("structural_mass", ref=1e6)

    prob_struct.model.add_constraint("con_uar", upper=1.0)
    prob_struct.model.add_constraint("con_yar", upper=1.0)
    prob_struct.model.add_constraint("con_uas", upper=1.0)
    prob_struct.model.add_constraint("con_yas", upper=1.0)

    prob_struct.model.approx_totals(method="fd")

    prob_struct.setup()
    # --- Design Variables ---

    if prob_in is None:
        # Initial design variables for a PMSG designed for a 15MW turbine
        prob_struct["Sigma_shear"] = 74.99692029e3
        prob_struct["Sigma_normal"] = 378.45123826e3
        prob_struct["T_e"] = 9e06
        prob_struct["l_s"] = 1.44142189  # rev 1 9.94718e6
        prob_struct["D_a"] = 7.74736313
        prob_struct["r_g"] = 4
        prob_struct["h_t"] = 0.1803019703
        prob_struct["rho_Fes"] = 7850
        prob_struct["rho_Fe"] = 7700
        prob_struct["phi"] = 90.0
        prob_struct["R_sh"] = 1.25
        prob_struct["R_nor"] = 0.95
        prob_struct["u_allow_pcent"] = 50
        prob_struct["y_allow_pcent"] = 20
        prob_struct["h_sr"] = 0.1254730934
        prob_struct["h_ss"] = 0.050
        prob_struct["t_r"] = 0.05
        prob_struct["t_s"] = 0.1
        prob_struct["y_bd"] = 0.00
        prob_struct["theta_bd"] = 0.00
        prob_struct["y_sh"] = 0.00
        prob_struct["theta_sh"] = 0.00

        prob_struct["Copper"] = 60e3
        prob_struct["M_Fest"] = 4000

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

    save_data(os.path.join(output_dir, "IPM_output"), prob)

    ratingMW = float(prob.get_val("P_rated", units="MW"))

    raw_data = [
        ["Rating",                                "P_rated",           ratingMW, "MW", ""],
        ["Air gap diameter",                      "r_g",               2*float(prob.get_val("r_g",units="m")), "m", ""],
        ["Stator diameter",                       "D_a",               float(prob.get_val("D_a", units="m")), "m", "(6-10.5)"],
        ["Overall Outer diameter",                "D_outer",           float(prob.get_val("D_outer",units="m")), "m", ""],
        ["Air gap length",                        "g",                 float(prob.get_val("g",units="mm")), "mm", "(7-15)"],
        ["Stator length",                         "l_s",               float(prob.get_val("l_s",units="m")), "m", "(1.5-2.5)"],
        ["l/d ratio",                             "K_rad",             float(prob.get_val("K_rad")), "", ""],
        # ["Slot_aspect_ratio",                     "Slot_aspect_ratio", float(prob.get_val("Slot_aspect_ratio")), "", ""],
        ["Pole pitch",                            "tau_p",             float(prob.get_val("tau_p",units="mm")), "mm", ""],
        ["Slot pitch",                            "tau_s",             float(prob.get_val("tau_s",units="mm")), "mm", ""],
        ["Stator slot height",                    "h_s",               float(prob.get_val("h_s",units="mm")), "mm", ""],
        ["Stator slotwidth",                      "b_s",               float(prob.get_val("b_s",units="mm")), "mm", ""],
        ["Stator tooth width",                    "b_t",               float(prob.get_val("b_t",units="mm")), "mm", "(150-350)"],
        ["Stator tooth height",                   "h_t",               float(prob.get_val("h_t",units="mm")), "mm", "(150-350)"],
        ["Stator yoke height",                    "h_ys",              float(prob.get_val("h_ys",units="mm")), "mm", "(25-150)"],
        ["Rotor yoke height",                     "h_yr",              float(prob.get_val("h_yr",units="mm")), "mm", "(25-150)"],
        ["Magnet height",                         "h_m",               float(prob.get_val("h_m",units="mm")), "mm", "(10-50)"],
        ["Magnet width",                          "l_m",               float(prob.get_val("l_m",units="mm")), "mm", ""],
        ["Magnet distance from inner radius",     "d_mag",             float(prob.get_val("d_mag",units="mm")), "mm", "(50-250)"],
        # ["Bridge separation width",               "d_sep",             float(prob.get_val("d_sep",units="mm")), "mm", "(0-20)"],
        # ["Bridge separation width",               "m_sep",             float(prob.get_val("m_sep",units="mm")), "mm", "(5-10)"],
        ["Pole:bridge ratio",                     "magnet_l_pc",       float(prob.get_val("magnet_l_pc")), "", "(0.7-1)"],
        ["V-angle",                               "alpha_v",           float(prob.get_val("alpha_v",units="deg")), "deg", "(60-160)"],
        #["Barrier distance",                      "w_fe",              float(prob.get_val("w_fe",units="mm")), "mm", ""],
        ["Peak air gap flux density fundamental", "B_g",               float(prob.get_val("B_g",units="T")), "T", ""],
        ["Peak statorflux density",               "B_smax",            float(prob.get_val("B_smax",units="T")), "T", ""],
        ["Peak rotor yoke flux density",          "B_rymax",           float(prob.get_val("B_rymax",units="T")), "T", "<2.53"],
        ["Pole pairs",                            "pp",                float(prob.get_val("pp")), "-", "(70-260)"],
        ["Generator output frequency",            "f",                 float(prob.get_val("f",units="Hz")), "Hz", ""],
        ["Generator output phase voltage",        "E_p",               float(prob.get_val("E_p",units="V")), "V", ""],
        ["Terminal voltage target",               "E_p_target",        float(prob.get_val("E_p_target", units="V")), "Volts", ""],
        ["Terminal voltage constraint",           "E_p_ratio",         float(prob.get_val("E_p_ratio")), "", "0.9 < x < 1.1"],
        ["Generator Output phase current",        "I_s",               float(prob.get_val("I_s",units="A")), "A", "(2900-8500)"],
        ["Stator resistance",                     "R_s",               float(prob.get_val("R_s")), "ohm/phase", ""],
        ["Stator slots",                          "S",                 float(prob.get_val("S")), "slots", ""],
        ["Stator turns",                          "N_c",               float(prob.get_val("N_c")), "turns", "(2-32)"],
        ["Conductor cross-section",               "A_Cuscalc",         float(prob.get_val("A_Cuscalc",units="mm**2")), "mm^2", ""],
        ["Stator Current density ",               "J_s",               float(prob.get_val("J_s",units="A/mm/mm")), "A/mm^2", "(3-6)"],
        ["Electromagnetic Torque",                "T_e",               float(prob.get_val("T_e",units="MN*m")), "MNm", ""],
        ["Torque rated target",                   "T_rated",           float(prob.get_val("T_rated", units="MN*m")), "MNm", ""],
        ["Torque constraint",                     "torque_ratio",      float(prob.get_val("torque_ratio")), "", "0.97 < x < 1.05"],
        ["Shear stress",                          "Sigma_shear",       float(prob.get_val("Sigma_shear",units="kN/m**2")), "kPa", ""],
        ["Normal stress",                         "Sigma_normal",      float(prob.get_val("Sigma_normal",units="kN/m**2")), "kPa", ""],
        ["Generator Efficiency ",                 "gen_eff",           100*float(prob.get_val("gen_eff")), "%", ">=95.5"],
        ["Iron mass",                             "Iron",              float(prob.get_val("Iron",units="t")), "tons", ""],
        ["Magnet mass",                           "mass_PM",           float(prob.get_val("mass_PM",units="t")), "tons", ""],
        ["Copper mass",                           "Copper",            float(prob.get_val("Copper",units="t")), "tons", ""],
        ["Structural Mass",                       "structural_mass",   float(prob.get_val("structural_mass",units="t")), "tons", ""],
        ["Rotor disc thickness",                  "t_r",               float(prob.get_val("t_r",units="mm")), "mm", "(0.05-0.3)"],
        ["Rotor rim thickness",                   "h_sr",              float(prob.get_val("h_sr",units="mm")), "mm", "(0.04-0.2)"],
        ["Stator disc thickness",                 "t_s",               float(prob.get_val("t_s",units="mm")), "mm", "(0.05-0.3)"],
        ["Stator rim thickness",                  "h_ss",              float(prob.get_val("h_ss",units="mm")), "mm", "(0.04-0.2)"],
        ["Rotor radial deflection constraint",    "con_uar",           float(prob.get_val("con_uar")), "", "<1"],
        ["Rotor axial deflection constraint",     "con_yar",           float(prob.get_val("con_yar")), "", "<1"],
        ["Stator radial deflection constraint",   "con_uas",           float(prob.get_val("con_uas")), "", "<1"],
        ["Stator axial deflection constraint",    "con_yas",           float(prob.get_val("con_yas")), "", "<1"],
        ["Total Mass",                            "mass_total",        float(prob.get_val("mass_total",units="t")), "tons", ""],
        ["Total Material Cost",                   "cost_total",        1e-3*float(prob.get_val("cost_total")), "k$", ""],
        ]

    df = pd.DataFrame(raw_data, columns=["Parameters", "Symbol", "Values", "Units", "Limit"])
    df.to_excel(os.path.join(output_dir, f"Optimized_IPM_PMSG_{ratingMW}_MW.xlsx"))

def run_all(output_str, opt_flag, obj_str, ratingMW):
    output_dir = os.path.join(mydir, output_str)

    # Optimize just magnetrics with GA and then structural with SLSQP
    prob = optimize_magnetics_design(output_dir=output_dir, opt_flag=opt_flag, obj_str=obj_str, ratingMW=int(ratingMW), restart_flag=False)
    prob_struct = optimize_structural_design(prob_in=prob, output_dir=output_dir, opt_flag=opt_flag)

    # Bring all data together
    prob = copy_data(prob_struct, prob)
    prob.run_model()

    # Write to xlsx and csv files
    write_all_data(prob, output_dir=output_dir)
    prob.model.list_outputs(val=True, hierarchical=True)
    #cleanup_femm_files(mydir)

if __name__ == "__main__":
    opt_flag = False
    run_all("outputs17-mass", opt_flag, "mass", 17)
    #for k in ratings_known:
    #    for obj in ["cost", "mass"]:
    #        run_all(f"outputs{k}-{obj}", opt_flag, obj, k)
