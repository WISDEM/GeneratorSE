import openmdao.api as om
from generatorse.lts.lts import LTS_Outer_Rotor_Opt
from generatorse.lts.structural import LTS_Outer_Rotor_Structural
from generatorse.driver.nlopt_driver import NLoptDriver
from generatorse.common.femm_util import cleanup_femm_files
from generatorse.common.run_util import copy_data, load_data, save_data
import os
import pandas as pd
import numpy as np
import platform

if platform.system().lower() == 'darwin':
    os.environ["CX_BOTTLE"] = "FEMM"
    os.environ["WINEPATH"] = "/Users/gbarter/bin/wine"
    os.environ["FEMMPATH"] = "/Users/gbarter/Library/Application Support/CrossOver/Bottles/FEMM/drive_c/femm42/bin/femm.exe"

ratings_known = [15, 17, 20, 22, 25]
rotor_diameter = {}
rotor_diameter[15] = 242.24
rotor_diameter[17] = 257.88
rotor_diameter[20] = 279.71
rotor_diameter[22] = 293.36
rotor_diameter[25] = 312.73
rated_speed = {}
rated_speed[15] = 7.49
rated_speed[17] = 7.04
rated_speed[20] = 6.49
rated_speed[22] = 6.18
rated_speed[25] = 5.80
target_eff = 0.97
fsql = "log.sql"
output_root = "LTS_output"

mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file

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
    prob.model = LTS_Outer_Rotor_Opt()

    prob.driver = NLoptDriver()
    prob.driver.options['optimizer'] = 'LN_COBYLA'
    prob.driver.options["maxiter"] = 250
    prob.driver.options["tol"] = 1e-8
    #prob.driver = om.ScipyOptimizeDriver()
    #prob.driver.options['optimizer'] = 'SLSQP'
    #prob.driver.options["maxiter"] = 50
    #prob.driver.options["tol"] = 1e-8
    #prob.driver = om.DifferentialEvolutionDriver()
    #prob.driver.options["max_gen"] = 20
    #prob.driver.options["pop_size"] = 30
    #prob.driver.options["penalty_exponent"] = 3

    D_a_up = 9. if ratingMW < 20 else 9.5
    #prob.model.add_design_var("delta_em", lower=0.060, upper=0.15, ref=0.08)
    #if obj_str.lower() == "cost":
    #prob.model.add_design_var("D_a", lower=5, upper=D_a_up)
    prob.model.add_design_var("h_sc", lower=0.03, upper=0.25, ref=0.1)
    prob.model.add_design_var("h_s", lower=0.05, upper=0.4, ref=0.1)
    prob.model.add_design_var("h_yr", lower=0.15, upper=0.3, ref=0.1)
    prob.model.add_design_var("p", lower=15, upper=40, ref=100)
    #prob.model.add_design_var("l_s", lower=0.75, upper=1.75)
    #prob.model.add_design_var("alpha", lower=0.1, upper=1)
    prob.model.add_design_var("dalpha", lower=0.2, upper=0.6)
    prob.model.add_design_var("N_sc", lower=1500, upper=3500, ref=1e3)
    #prob.model.add_design_var("N_c", lower=1, upper=7)
    #prob.model.add_design_var("load_margin", lower=0.85, upper=0.95)

    # prob.model.add_constraint("Slot_aspect_ratio", lower=4.0, upper=10.0)  # 11
    #prob.model.add_constraint("con_angle", lower=0.001, ref=0.1)
    # Differential evolution driver cannot do double-sided constraints, so have to hack it
    #prob.model.add_constraint("E_p", lower=0.8 * 3300, ref=3000)
    prob.model.add_constraint("E_p_ratio", lower=0.8, upper=1.20)
    prob.model.add_constraint("constr_B_g_coil", upper=1.0)
    #prob.model.add_constraint("B_coil_max", lower=6.0)
    #prob.model.add_constraint("Coil_max_ratio", upper=1.2)
    #prob.model.add_constraint("Critical_current_ratio",upper=1.2)
    #prob.model.add_constraint("B_rymax", upper=2.3)
    prob.model.add_constraint("torque_ratio", lower=1.0+0.15)
    #prob.model.add_constraint("Torque_actual", upper=1.2*target_torque, ref=20e6)
    prob.model.add_constraint("gen_eff", lower=0.97)

    if obj_str.lower() == 'cost':
        prob.model.add_objective("cost_total", ref=1e6)
    elif obj_str.lower() == 'mass':
        prob.model.add_objective("mass_total", ref=1e6)
    else:
        print('Objective?', obj_str)

    prob.model.approx_totals(method="fd", step=1e-3, form='forward')

    prob.setup()
    print('************************')
    print('Design objectives:')
    print('Generator rating in MW: ', ratingMW)
    print('Rated speed in rpm: ',rated_speed[ratingMW])
    print('Target torque in MNm: ', target_torque*1e-6)
    print('Objective function: ', obj_str)
    print('************************')

    if prob_in is None:
        prob["m"] = 6  # phases
        prob["q"] = 2  # slots per pole per phase
        prob["b_s_tau_s"] = 0.45
        prob["conductor_area"] = 1.8 * 1.2e-6
        prob["K_h"] = 2  # specific hysteresis losses W/kg @ 1.5 T
        prob["K_e"] = 0.5  # specific hysteresis losses W/kg @ 1.5 T

        # ## Initial design variables for a PMSG designed for a 15MW turbine
        prob["mass_adder"] = 77e3   # 77t, could maybe increase a bit at 25MW
        prob["cost_adder"] = 700e3  # 700k, could maybe increase a bit at 25MW
        prob["E_p_target"] = 3300.0
        prob["p"] = 30.0
        prob["dalpha"] = 0.33
        prob["N_sc"] = 2811.37208924
        prob["N_c"] = 2.2532984
        prob["I_s"] = 1945.9858772
        prob["J_s"] = 3.0

        # Material properties
        prob["rho_steel"] = 7850
        prob["rho_Fe"] = 7700.0  # Steel density
        prob["rho_Copper"] = 8900.0  # Kg/m3 copper density
        prob["rho_NbTi"] = 8442.37093661195  # magnet density
        prob["resisitivty_Cu"] = 1.724e-8  # 1.8e-8 * 1.4  # Copper resisitivty
        prob["U_b"] = 1  # brush voltage drop
        prob["Y"] = 10  # Short pitch

        prob["Tilt_angle"] = 90.0
        prob["R_shaft_outer"] = 1.25
        prob["R_nose_outer"] = 0.95
        prob["u_allow_pcent"] = 30
        prob["y_allow_pcent"] = 20
        prob["h_yr_s"] = 0.025 #0.18866371
        prob["h_ys"] = 0.025
        prob["t_rdisc"] = 0.02507061
        prob["t_sdisc"] = 0.025

        if restart_flag:
            prob = load_data(os.path.join(output_dir, output_root), prob)

        # Have to set these last in case we initiatlized from a different rating
        prob["P_rated"] = ratingMW * 1e6
        prob["T_rated"] = target_torque
        prob["N_nom"] = rated_speed[ratingMW]
        prob["delta_em"] = 0.06
        prob["load_margin"] = 0.95
        prob["y_bd"] = 1e-4
        prob["theta_bd"] = 1e-3
        prob["y_sh"] = 1e-4
        prob["theta_sh"] = 1e-3
        prob["gamma"] = 1.5
        prob["l_s"] = 1.0
        prob["D_a"] = prob["D_generator"] = D_a_up
        prob["N_c"] = 1.0

        #Specific costs
        prob["C_Cu"]           = 7.3
        prob["C_Fe"]           = 1.56
        prob["C_Fes"]          = 4.44
        prob["C_PM"]           = 66.72
        prob["C_NbTi"]         = 45.43

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
    print("N_sc", prob["N_sc"])
    print("B_coil_max", prob["B_coil_max"])
    print("l_s", prob["l_s"])
    print("Torque_ratio", prob["torque_ratio"])

    return prob

def optimize_structural_design(prob_in=None, output_dir=None, opt_flag=False):
    if output_dir is None:
        output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    prob_struct = om.Problem()
    prob_struct.model = LTS_Outer_Rotor_Structural()

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

    prob_struct.model.add_design_var("h_yr_s", lower=0.0250, upper=0.5, ref=0.1)
    prob_struct.model.add_design_var("h_ys", lower=0.025, upper=0.6, ref=0.1)
    prob_struct.model.add_design_var("t_rdisc", lower=0.025, upper=0.5, ref=0.1)
    prob_struct.model.add_design_var("t_sdisc", lower=0.025, upper=0.5, ref=0.1)
    prob_struct.model.add_objective("mass_structural", ref=1e6)

    prob_struct.model.add_constraint("con_uas", upper=1.0)
    prob_struct.model.add_constraint("con_yas", upper=1.0)
    prob_struct.model.add_constraint("con_uar", upper=1.0)
    prob_struct.model.add_constraint("con_yar", upper=1.0)

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
        prob_struct["y_bd"] = 1e-4
        prob_struct["theta_bd"] = 1e-3
        prob_struct["y_sh"] = 1e-4
        prob_struct["theta_sh"] = 1e-3
        prob_struct["gamma"] = 1.5

        #prob_struct["mass_copper"] = 60e3
        prob_struct["mass_NbTi"] = 4000
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

    save_data(os.path.join(output_dir, output_root), prob)

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
        ["delta Alpha",                   "dalpha",                 float(prob.get_val("dalpha")), "", "(0.05-0.95)"],
        ["Beta",                          "beta",                   float(prob.get_val("beta", units="deg")), "deg", ""],
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
        ["Field coil current out",        "I_sc_out",               float(prob.get_val("I_sc_out", units="A")), "A", ""],
        ["Critical current load margin",  "load_margin",            float(prob.get_val("load_margin")), "", "(0.85-0.95)"],
        #["Critical current constraint",   "Critical_current_ratio", float(prob.get_val("Critical_current_ratio")), "", "<1.2"],
        ["Layer count",                   "N_l",                    float(prob.get_val("N_l")), "layers", ""],
        ["Turns per layer",               "N_sc_layer",             float(prob.get_val("N_sc_layer")), "turns", ""],
        ["length per racetrack",          "l_sc",                   float(prob.get_val("l_sc", units="km")), "km", ""],
        ["Mass per coil",                 "mass_NbTi_racetrack",      float(prob.get_val("mass_NbTi_racetrack", units="kg")), "kg", ""],
        ["Total mass of SC coils",        "mass_NbTi",                float(prob.get_val("mass_NbTi", units="t")), "Tons", ""],
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
        ["Rotor axial deflection",        "y_ar",                   float(prob.get_val("y_ar", units="mm")), "mm", ""],
        ["Rotor axial limit",             "y_allowable_r",          float(prob.get_val("y_allowable_r", units="mm")), "mm", ""],
        ["Rotor torsional twist",         "twist_r",                float(prob.get_val("twist_r", units="deg")), "deg", ""],
        ["Stator radial deflection",      "u_as",                   float(prob.get_val("u_as", units="mm")), "mm", ""],
        ["Stator radial limit",           "u_allowable_s",          float(prob.get_val("u_allowable_s", units="mm")), "mm", ""],
        ["Stator axial deflection",       "y_as",                   float(prob.get_val("y_as", units="mm")), "mm", ""],
        ["Stator axial limit",            "y_allowable_s",          float(prob.get_val("y_allowable_s", units="mm")), "mm", ""],
        ["Partial safety factor",         "gamma",                  float(prob.get_val("gamma")), "", ""],
        ["Radial deflection constraint-rotor", "con_uar",           float(prob.get_val("con_uar")), "-", "< 1.0"],
        ["Radial deflection constraint-rotor", "con_uas",           float(prob.get_val("con_uas")), "-", "< 1.0"],
        ["Axial deflection constraint-rotor" , "con_yar",           float(prob.get_val("con_yar")), "-", "< 1.0"],
        ["Axial deflection constraint-rotor" , "con_yas",           float(prob.get_val("con_yas")), "-", "< 1.0"],
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
    prob_struct = optimize_structural_design(prob_in=prob, output_dir=output_dir, opt_flag=True)

    # Bring all data together
    prob["h_yr_s"] = prob_struct["h_yr_s"]
    prob["h_ys"] = prob_struct["h_ys"]
    prob["t_rdisc"] = prob_struct["t_rdisc"]
    prob["t_sdisc"] = prob_struct["t_sdisc"]
    prob.run_model()

    # Write to xlsx and csv files
    write_all_data(prob, output_dir=output_dir)
    prob.model.list_outputs(val=True, hierarchical=True)
    cleanup_femm_files(mydir, output_dir)

if __name__ == "__main__":
    opt_flag = False #True
    #run_all("outputs15-mass", opt_flag, "mass", 15)
    #run_all("outputs17-mass", opt_flag, "mass", 17)
    #run_all("outputs20-mass", opt_flag, "mass", 20)
    #run_all("outputs25-mass", opt_flag, "mass", 25)
    #run_all("outputs15-cost", opt_flag, "cost", 15)
    #run_all("outputs17-cost", opt_flag, "cost", 17)
    #run_all("outputs20-cost", opt_flag, "cost", 20)
    #run_all("outputs22-cost", opt_flag, "cost", 22)
    #run_all("outputs25-cost", opt_flag, "cost", 25)
    #for _ in range(2):
    for k in ratings_known:
        for obj in ["cost"]:#, "mass"]:
            run_all(f"outputs{k}-{obj}", opt_flag, obj, k)
