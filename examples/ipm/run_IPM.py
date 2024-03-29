import openmdao.api as om
from generatorse.ipm.ipm import PMSG_Outer_rotor_Opt
from generatorse.ipm.structural import PMSG_Outer_Rotor_Structural
from generatorse.driver.nlopt_driver import NLoptDriver
from generatorse.common.femm_util import cleanup_femm_files
from generatorse.common.run_util import copy_data, load_data, save_data
import os
import pandas as pd
import numpy as np
import platform

if platform.system().lower() == "darwin":
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
target_eff = 0.95
fsql = "log.sql"
output_root = "IPM_output"
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
    prob.model = PMSG_Outer_rotor_Opt(debug_prints=False)

    prob.driver = NLoptDriver()
    prob.driver.options['optimizer'] = 'LN_COBYLA'
    prob.driver.options["maxiter"] = 200
    prob.driver.options["tol"] = 1e-8
    #prob.driver = om.DifferentialEvolutionDriver()
    #prob.driver.options["max_gen"] = 20
    #prob.driver.options["pop_size"] = 40
    #prob.driver.options["penalty_exponent"] = 3

    #recorder = om.SqliteRecorder(os.path.join(output_dir, fsql))
    #prob.driver.add_recorder(recorder)
    #prob.add_recorder(recorder)
    #prob.driver.recording_options["excludes"] = ["*_df"]
    #prob.driver.recording_options["record_constraints"] = True
    #prob.driver.recording_options["record_desvars"] = True
    #prob.driver.recording_options["record_objectives"] = True

    D_a_up = 9. if ratingMW < 20 else 9.5
    h_m_up = 0.05 if ratingMW < 22 else 0.06
    #prob.model.add_design_var("D_a", lower=6, upper=D_a_up)
    #prob.model.add_design_var("g", lower=0.007, upper=0.015, ref=0.01) # always lower bound
    #prob.model.add_design_var("l_s", lower=0.75, upper=2.5)
    prob.model.add_design_var("h_t", lower=0.04, upper=0.350)
    prob.model.add_design_var("h_ys", lower=0.02, upper=0.3)
    prob.model.add_design_var("h_yr", lower=0.02, upper=0.3)
    prob.model.add_design_var("h_m", lower=0.005, upper=h_m_up, ref=0.01)
    prob.model.add_design_var("pp", lower=50, upper=100, ref=100.0)
    prob.model.add_design_var("N_c", lower=2, upper=10, ref=10)
    #prob.model.add_design_var("d_mag",lower=0.05, upper=0.25, ref=0.1) # always lower bound
    prob.model.add_design_var("b_t", lower=0.02, upper=0.5, ref=0.1)
    #prob.model.add_design_var("J_s",lower=3, upper=6)

    prob.model.add_constraint("B_rymax", upper=2.6)
    #prob.model.add_constraint("B_smax", upper=2.6)
    # prob.model.add_constraint("K_rad",    lower=0.15, upper=0.3)
    #prob.model.add_constraint("E_p", upper=1.2 * 3300, ref=3000)
    prob.model.add_constraint("E_p_ratio", lower=0.85, upper=1.15)
    prob.model.add_constraint("torque_ratio", lower=1.0+0.1)
    #prob.model.add_constraint("T_e", upper=1.2*target_torque, ref=20e6)
    #prob.model.add_constraint("r_outer_active", upper=11. / 2.)

    prob.model.add_constraint("gen_eff", lower=0.955+0.02)

    if obj_str.lower() == "cost":
        prob.model.add_objective("cost_total", ref=1e6)
    elif obj_str.lower() == "mass":
        prob.model.add_objective("mass_total", ref=1e5)

    prob.model.approx_totals(method="fd", step = 1e-4, form='central')

    prob.setup()
    print('************************')
    print('Design objectives:')
    print('Generator rating in MW: ', ratingMW)
    print('Rated speed in rpm: ',rated_speed[ratingMW])
    print('Target torque in MNm: ', target_torque*1e-6)
    print('Objective function: ', obj_str)
    print('************************')

    if prob_in is None:
        # Initial design variables for a PMSG designed for a 15MW turbine
        prob["E_p_target"]     = 3300.0

        # These are the current design variables
        prob["D_a"]            = D_a_up
        prob["l_s"]            = 2.3
        prob["h_t"]            = 0.168
        prob["pp"]             = 85.0
        prob["g"]              = 0.007
        prob["N_c"]            = 3.37
        prob["I_s"]            = 5864.0
        prob["h_m"]            = 0.0295
        prob["d_mag"]          = 0.05
        prob["h_ys"]           = 0.051
        prob["h_yr"]           = 0.02
        prob["magnet_l_pc"]    = 1.0
        prob["J_s"]            = 6
        prob["phi"]            = 90
        prob["b_t"]            = 0.1
        prob["b"]              = 2.
        prob["c"]              = 5.0

        #Material properties
        prob["rho_Fe"]         = 7700.0                 #Steel density
        prob["rho_Fes"]        = 7850.0                 #Steel density
        prob["rho_Copper"]     = 8900.0                  # Kg/m3 copper density
        prob["rho_PM"]         = 7600.0                  # magnet density
        prob["resistivity_Cu"] = 1.8e-8*1.4			# Copper resisitivty

        #Support structure parameters
        prob["R_no"]           = 0.925
        prob["R_sh"]           = 1.25
        prob["t_r"]            = 0.057 #  0.06
        prob["h_sr"]           = 0.1
        prob["t_s"]            = 0.1
        prob["h_ss"]           = 0.1
        prob["y_sh"]           = 1e-4
        prob["theta_sh"]       = 1e-3
        prob["y_bd"]           = 1e-4
        prob["theta_bd"]       = 1e-3
        prob["u_allow_pcent"]  = 8.5       # % radial deflection
        prob["y_allow_pcent"]  = 1.0       # % axial deflection
        prob["z_allow_deg"]    = 0.05       # torsional twist
        prob["gamma"]          = 1.5

        if restart_flag:
            prob = load_data(os.path.join(output_dir, output_root), prob)

        # Have to set these last in case we initiatlized from a different rating
        prob["P_rated"] = ratingMW * 1e6
        prob["T_rated"] = target_torque
        prob["N_nom"] = rated_speed[ratingMW]
        prob["d_mag"] = 0.05
        prob["g"] = 0.007
        prob["D_a"] = prob["D_generator"] = D_a_up
        prob["l_s"] = 2.5
        prob["J_s"] = 6.0
        prob["h_yr"] = 0.1

        #Specific costs
        prob["C_Cu"]           = 7.3
        prob["C_Fe"]           = 1.56
        prob["C_Fes"]          = 4.44
        prob["C_PM"]           = 66.72
        prob["C_NbTi"]         = 45.43

    else:
        prob = copy_data(prob_in, prob)

    # prob.model.approx_totals(method="fd", step=1e-3)

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

def optimize_structural_design(prob_in=None, output_dir=None, opt_flag=False, ratingMW=17):
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

    prob_struct.model.add_objective("mass_structural", ref=1e6)

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
        prob_struct["h_sr"] = h_sr[ratingMW]
        prob_struct["h_ss"] = h_ss[ratingMW]
        prob_struct["t_r"] = t_r[ratingMW]
        prob_struct["t_s"] = t_s[ratingMW]
        prob_struct["gamma"] = 1.5

        prob_struct["mass_copper"] = 60e3
        prob_struct["M_Fest"] = 4000

    else:
        prob_struct = copy_data(prob_in, prob_struct)

    prob_struct["y_bd"] = 1e-4
    prob_struct["theta_bd"] = 1e-3
    prob_struct["y_sh"] = 1e-4
    prob_struct["theta_sh"] = 1e-3

    # prob_struct.model.approx_totals(method="fd")

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
        ["Rating",                                "P_rated",           ratingMW, "MW", ""],
        ["Air gap radius",                        "r_g",               float(prob.get_val("r_g",units="m")), "m", ""],
        ["Stator diameter",                       "D_a",               float(prob.get_val("D_a", units="m")), "m", "(6-10)"],
        ["Overall Outer diameter",                "D_outer",           float(prob.get_val("D_outer",units="m")), "m", ""],
        ["Air gap length",                        "g",                 float(prob.get_val("g",units="mm")), "mm", "(7-15)"],
        ["Stator length",                         "l_s",               float(prob.get_val("l_s",units="m")), "m", "(0.75-2.5)"],
        ["l/d ratio",                             "K_rad",             float(prob.get_val("K_rad")), "", ""],
        # ["Slot_aspect_ratio",                     "Slot_aspect_ratio", float(prob.get_val("Slot_aspect_ratio")), "", ""],
        ["Pole pitch",                            "tau_p",             float(prob.get_val("tau_p",units="mm")), "mm", ""],
        ["Slot pitch",                            "tau_s",             float(prob.get_val("tau_s",units="mm")), "mm", ""],
        ["Stator slot height",                    "h_s",               float(prob.get_val("h_s",units="mm")), "mm", ""],
        ["Stator slotwidth",                      "b_s",               float(prob.get_val("b_s",units="mm")), "mm", ""],
        ["Stator tooth width",                    "b_t",               float(prob.get_val("b_t",units="mm")), "mm", "(20-500)"],
        ["Stator tooth height",                   "h_t",               float(prob.get_val("h_t",units="mm")), "mm", "(40-350)"],
        ["Stator yoke height",                    "h_ys",              float(prob.get_val("h_ys",units="mm")), "mm", "(20-300)"],
        ["Rotor yoke height",                     "h_yr",              float(prob.get_val("h_yr",units="mm")), "mm", "(20-300)"],
        ["Magnet height",                         "h_m",               float(prob.get_val("h_m",units="mm")), "mm", "(5-50)"],
        ["Magnet width",                          "l_m",               float(prob.get_val("l_m",units="mm")), "mm", ""],
        ["Magnet distance from inner radius",     "d_mag",             float(prob.get_val("d_mag",units="mm")), "mm", "(50-250)"],
        # ["Bridge separation width",               "d_sep",             float(prob.get_val("d_sep",units="mm")), "mm", "(0-20)"],
        # ["Bridge separation width",               "m_sep",             float(prob.get_val("m_sep",units="mm")), "mm", "(5-10)"],
        ["Pole:bridge ratio",                     "magnet_l_pc",       float(prob.get_val("magnet_l_pc")), "", ""],
        ["V-angle",                               "alpha_v",           float(prob.get_val("alpha_v",units="deg")), "deg", ""],
        #["Barrier distance",                      "w_fe",              float(prob.get_val("w_fe",units="mm")), "mm", ""],
        ["Peak air gap flux density fundamental", "B_g",               float(prob.get_val("B_g",units="T")), "T", ""],
        ["Peak statorflux density",               "B_smax",            float(prob.get_val("B_smax",units="T")), "T", "< 2.53"],
        ["Peak rotor yoke flux density",          "B_rymax",           float(prob.get_val("B_rymax",units="T")), "T", "< 2.53"],
        ["Pole pairs",                            "pp",                float(prob.get_val("pp")), "-", "(60-260)"],
        ["Generator output frequency",            "f",                 float(prob.get_val("f",units="Hz")), "Hz", ""],
        ["Generator output phase voltage",        "E_p",               float(prob.get_val("E_p",units="V")), "V", ""],
        ["Terminal voltage target",               "E_p_target",        float(prob.get_val("E_p_target", units="V")), "Volts", ""],
        ["Terminal voltage constraint",           "E_p_ratio",         float(prob.get_val("E_p_ratio")), "", "0.8 < x < 1.2"],
        ["Generator Output phase current",        "I_s",               float(prob.get_val("I_s",units="A")), "A", "(2500-8500)"],
        ["Stator resistance",                     "R_s",               float(prob.get_val("R_s")), "ohm/phase", ""],
        ["Stator slots",                          "S",                 float(prob.get_val("S")), "slots", ""],
        ["Stator turns",                          "N_c",               float(prob.get_val("N_c")), "turns", "(2-10)"],
        ["Conductor cross-section",               "A_Cuscalc",         float(prob.get_val("A_Cuscalc",units="mm**2")), "mm^2", ""],
        ["Stator Current density ",               "J_s",               float(prob.get_val("J_s",units="A/mm/mm")), "A/mm^2", "(3-10)"],
        ["Electromagnetic Torque",                "T_e",               float(prob.get_val("T_e",units="MN*m")), "MNm", ""],
        ["Torque rated target",                   "T_rated",           float(prob.get_val("T_rated", units="MN*m")), "MNm", ""],
        ["Torque constraint",                     "torque_ratio",      float(prob.get_val("torque_ratio")), "", "1.0 < x < 1.2"],
        ["Shear stress",                          "Sigma_shear",       float(prob.get_val("Sigma_shear",units="kN/m**2")), "kPa", ""],
        ["Normal stress",                         "Sigma_normal",      float(prob.get_val("Sigma_normal",units="kN/m**2")), "kPa", ""],
        ["Generator Efficiency ",                 "gen_eff",           100*float(prob.get_val("gen_eff")), "%", ">=95.5"],
        ["Iron mass",                             "mass_iron",              float(prob.get_val("mass_iron",units="t")), "tons", ""],
        ["Magnet mass",                           "mass_PM",           float(prob.get_val("mass_PM",units="t")), "tons", ""],
        ["Copper mass",                           "mass_copper",            float(prob.get_val("mass_copper",units="t")), "tons", ""],
        ["Structural Mass",                       "mass_structural",   float(prob.get_val("mass_structural",units="t")), "tons", ""],
        ["Rotor disc thickness",                  "t_r",               float(prob.get_val("t_r",units="mm")), "mm", "(0.05-0.3)"],
        ["Rotor rim thickness",                   "h_sr",              float(prob.get_val("h_sr",units="mm")), "mm", "(0.04-0.2)"],
        ["Stator disc thickness",                 "t_s",               float(prob.get_val("t_s",units="mm")), "mm", "(0.05-0.3)"],
        ["Stator rim thickness",                  "h_ss",              float(prob.get_val("h_ss",units="mm")), "mm", "(0.04-0.2)"],
        ["Partial safety factor",                 "gamma",             float(prob.get_val("gamma")), "", ""],
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

    # Optimize just magnetics with GA and then structural with SLSQP
    prob = optimize_magnetics_design(output_dir=output_dir, opt_flag=opt_flag, obj_str=obj_str,
                                     ratingMW=int(ratingMW), restart_flag=True, cleanup_flag=False)
    prob_struct = optimize_structural_design(prob_in=prob, output_dir=output_dir,
                                             opt_flag=True, ratingMW=int(ratingMW))

    # Bring all data together
    for k in ["h_sr","h_ss","t_r","t_s"]:
        prob[k]  = prob_struct[k]
    prob.run_model()

    # Write to xlsx and csv files
    #prob.model.list_inputs(val=True, hierarchical=True, units=True, desc=True)
    prob.model.list_outputs(val=True, hierarchical=True)
    write_all_data(prob, output_dir=output_dir)
    cleanup_femm_files(mydir, output_dir)

if __name__ == "__main__":
    opt_flag = True
    for k in range(2):
        #run_all("outputs15-mass", opt_flag, "mass", 15)
        #run_all("outputs17-mass", opt_flag, "mass", 17)
        #run_all("outputs20-mass", opt_flag, "mass", 20)
        #run_all("outputs22-mass", opt_flag, "mass", 22)
        #run_all("outputs25-mass", opt_flag, "mass", 25)
        #run_all("outputs15-cost", opt_flag, "cost", 15)
        #run_all("outputs17-cost", opt_flag, "cost", 17)
        #run_all("outputs20-cost", opt_flag, "cost", 20)
        run_all("outputs22-cost", opt_flag, "cost", 22)
        run_all("outputs25-cost", opt_flag, "cost", 25)
    #for k in ratings_known:
    #    for obj in ["cost"]: #, "mass"]:
    #            run_all(f"outputs{k}-{obj}", opt_flag, obj, k)
