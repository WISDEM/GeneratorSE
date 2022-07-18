import openmdao.api as om
from generatorse.ms_pmsg.ms_pmsg import PMSG_Inner_rotor_Opt
from generatorse.ms_pmsg.structural import PMSG_Inner_Rotor_Structural
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

gear_ratio = 120
ratings_known = [15, 17, 20, 25]
rotor_diameter = {}
rotor_diameter[15] = 242.24
rotor_diameter[17] = 257.88
rotor_diameter[20] = 279.71
rotor_diameter[25] = 312.73
rated_speed = {}
rated_speed[15] = gear_ratio * 7.49
rated_speed[17] = gear_ratio * 7.04
rated_speed[20] = gear_ratio * 6.49
rated_speed[25] = gear_ratio * 5.80
target_eff = 0.97
fsql = "log.sql"
output_root = "MS-PMSG_output"
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file

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
    prob.driver.options["maxiter"] = 200
    prob.driver.options["tol"] = 1e-6
    #prob.driver = om.DifferentialEvolutionDriver()
    #prob.driver.options["max_gen"] = 15
    #prob.driver.options["pop_size"] = 30
    #prob.driver.options["penalty_exponent"] = 3

    #recorder = om.SqliteRecorder(os.path.join(output_dir, fsql))
    #prob.driver.add_recorder(recorder)
    #prob.add_recorder(recorder)
    #prob.driver.recording_options["excludes"] = ["*_df"]
    #prob.driver.recording_options["record_constraints"] = True
    #prob.driver.recording_options["record_desvars"] = True
    #prob.driver.recording_options["record_objectives"] = True

    prob.model.add_design_var("r_g", lower=0.5, upper=2.0)
    prob.model.add_design_var("l_s", lower=0.5, upper=2.5)
    prob.model.add_design_var("h_s", lower=0.025, upper=0.1, ref=0.01)
    prob.model.add_design_var("g", lower=0.006, upper=0.009, ref=0.01)
    prob.model.add_design_var("h_yr", lower=0.01, upper=0.1, ref=0.01)
    prob.model.add_design_var("h_ys", lower=0.01, upper=0.1, ref=0.01)
    prob.model.add_design_var("p", lower=4, upper=10)
    prob.model.add_design_var("N_c", lower=2, upper=12)
    prob.model.add_design_var("h_m", lower=0.005, upper=0.075, ref=0.01)
    prob.model.add_design_var("I_s", lower=500, upper=6000, ref=1e3)
    prob.model.add_design_var("ratio", lower=0.7, upper=0.85)

    #prob.model.add_constraint("E_p", upper=1.2 * 3300, ref=3000)
    prob.model.add_constraint("E_p_ratio", lower=0.8, upper=1.20)
    prob.model.add_constraint("torque_ratio", lower=1.0, upper=1.2)
    #prob.model.add_constraint("T_e", upper=1.2*target_torque, ref=1e5)
    prob.model.add_constraint("gen_eff", lower=0.96)

    if obj_str.lower() == 'cost':
        prob.model.add_objective("cost_total", ref=1e5)
    elif obj_str.lower() == 'mass':
        prob.model.add_objective("mass_total", ref=1e5)
    elif obj_str.lower() in ['eff','efficiency']:
        prob.model.add_objective("gen_eff", scaler=-1.0)
    else:
        print('Objective?', obj_str)

    prob.model.approx_totals(method="fd")

    prob.setup()
    print('************************')
    print('Design objectives:')
    print('Generator rating in MW: ', ratingMW)
    print('Rated speed in rpm: ',rated_speed[ratingMW])
    print('Target torque in MNm: ', target_torque*1e-6)
    print('Objective function: ', obj_str)
    print('************************')

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
        prob["H_c"] = 1.0 #units="A/m", desc="coercivity"
        prob["I_s"] = 1945.9858772
        prob["J_s"] = 3.0
        prob["N_c"] = 3.0
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
        prob["g"] = 0.008
        prob["g1"] = 9.806
        prob["h_m"] = 0.01
        prob["h_s"] = 0.05
        prob["h_s1"] = 0.010
        prob["h_s2"] = 0.010
        prob["h_sr"] = 0.02
        prob["h_ss"] = 0.02
        prob["h_yr"] = 0.05
        prob["h_ys"] = 0.05
        prob["k_fes"] = 0.95
        prob["k_sfil"] = 0.65
        prob["l_s"] = 0.6
        prob["m"] = 3  # phases
        prob["mu_0"] = np.pi * 4 * 1e-7
        prob["mu_r"] = 1.06
        prob["n_r"] = 6
        prob["n_s"] = 5
        prob["p"] = 6.0
        prob["phi"] = 90
        prob["q1"] = 1  # slots per pole per phase
        prob["r_g"] = 2.0
        prob["ratio"]= 0.7

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
        prob["gamma"] = 1.5

        if restart_flag:
            prob = load_data(os.path.join(output_dir, output_root), prob)

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
    prob_struct.model.add_objective("mass_structural", ref=1e3)

    prob_struct.model.add_constraint("con_bar", upper=1.0)
    prob_struct.model.add_constraint("con_uar", upper=1.0)
    prob_struct.model.add_constraint("con_yar", upper=1.0)
    prob_struct.model.add_constraint("con_zar", upper=1.0)

    prob_struct.model.add_constraint("con_bas", upper=1.0)
    prob_struct.model.add_constraint("con_uas", upper=1.0)
    prob_struct.model.add_constraint("con_yas", upper=1.0)
    prob_struct.model.add_constraint("con_zas", upper=1.0)

    prob_struct.model.approx_totals(method="fd")

    prob_struct.setup()

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
        ["specific current loading"                           , "A_1"               , float(prob.get_val("A_1", units="A/m")), "A/m"],
        ["Peak air gap flux density"                          , "B_g"               , float(prob.get_val("B_g", units="T")), "T"],
        ["remnant flux density"                               , "B_r"               , float(prob.get_val("B_r", units="T")), "T"],
        ["Peak Rotor yoke flux density"                       , "B_rymax"           , float(prob.get_val("B_rymax", units="T")), "T"],
        ["Peak Stator Yoke flux density"                      , "B_symax"           , float(prob.get_val("B_symax", units="T")), "T"],
        ["Peak Teeth flux density"                            , "B_tmax"            , float(prob.get_val("B_tmax", units="T")), "T"],
        ["no of slots per pole per phase"                     , "q1"                , float(prob.get_val("q1")), "-"],
        ["Air-gap radius"                                     , "r_g"               , float(prob.get_val("r_g", units="m")), "m", "(0.75-2.0)"],
        ["ratio of magnet width to pole pitch"                , "ratio"             , float(prob.get_val("ratio")), "-", "(0.7-0.85)"],
        ["Copper resistivity"                                 , "resisitivty_Cu"    , float(prob.get_val("resisitivty_Cu", units="ohm*m")), "ohm*m"],
        ["airgap length"                                      , "g"                 , float(prob.get_val("g", units="m")), "m", "(0.06-0.09)"],
        ["magnet height"                                      , "h_m"               , float(prob.get_val("h_m", units="m")), "m", "(0.005-0.075)"],
        ["Yoke height"                                        , "h_s"               , float(prob.get_val("h_s", units="m")), "m", "(0.05-0.1)"],
        ["Slot Opening height"                                , "h_s1"              , float(prob.get_val("h_s1")), "-"],
        ["Wedge Opening height"                               , "h_s2"              , float(prob.get_val("h_s2")), "-"],
        ["Rotor yoke height"                                  , "h_yr"              , float(prob.get_val("h_yr", units="m")), "m", "(0.01-0.1)"],
        ["Stator yoke height"                                 , "h_ys"              , float(prob.get_val("h_ys", units="m")), "m", "(0.01-0.1)"],
        ["Stator outer diameter"                              , "D_out"             , float(prob.get_val("D_out", units="m")), "m"],
        ["Stator current"                                     , "I_s"               , float(prob.get_val("I_s", units="A")), "A", "(500-6000)"],
        ["Current density"                                    , "J_s"               , float(prob.get_val("J_s", units="A/mm**2")), "A/mm**2"],
        ["core length"                                        , "l_s"               , float(prob.get_val("l_s", units="m")), "m", "(0.5-2.5)"],
        ["no of phases"                                       , "m"                 , float(prob.get_val("m")), "-"],
        ["No of pole pairs"                                   , "p"                 , float(prob.get_val("p")), "-", "(4-10)"],
        ["Aspect ratio"                                       , "K_rad"             , float(prob.get_val("K_rad")), "-"],
        ["Stator core length"                                 , "L_t"               , float(prob.get_val("L_t", units="m")), "m"],
        ["turns per coil"                                     , "N_c"               , float(prob.get_val("N_c")), "-", "(2-12)"],
        ["rated speed"                                        , "N_nom"             , float(prob.get_val("N_nom", units="rpm")), "rpm"],
        ["turns per phase"                                    , "N_s"               , float(prob.get_val("N_s")), "-"],
        ["magnet width"                                       , "b_m"               , float(prob.get_val("b_m", units="m")), "m"],
        ["slot width"                                         , "b_s"               , float(prob.get_val("b_s", units="m")), "m"],
        ["ratio of Slot width to slot pitch"                  , "b_s_tau_s"         , float(prob.get_val("b_s_tau_s")), "-"],
        ["relative permeability of magnet"                    , "mu_r"              , float(prob.get_val("mu_r")), "-"],
        ["Pole pitch"                                         , "tau_p"             , float(prob.get_val("tau_p", units="m")), "m"],
        ["Stator slot pitch"                                  , "tau_s"             , float(prob.get_val("tau_s", units="m")), "m"],
        ["Stator resistance"                                  , "R_s"               , float(prob.get_val("R_s", units="ohm")), "ohm"],
        ["Stator slots"                                       , "S"                 , float(prob.get_val("S")), "-"],
        #["Slot aspect ratio"                                  , "Slot_aspect_ratio" , float(prob.get_val("Slot_aspect_ratio")), "-"],
        ["Stator phase voltage"                               , "E_p"               , float(prob.get_val("E_p", units="V")), "V"],
        ["Voltage constraint"                                 , "E_p_ratio"         , float(prob.get_val("E_p_ratio")), "", "0.8 < x < 1.2"],
        ["Target voltage"                                     , "E_p_target"        , float(prob.get_val("E_p_target", units="V")), "V"],
        ["Normal stress"                                      , "Sigma_normal"      , float(prob.get_val("Sigma_normal", units="kN/m**2")), "kN/m**2"],
        ["Shear stress"                                       , "Sigma_shear"       , float(prob.get_val("Sigma_shear", units="kN/m**2")), "kN/m**2"],
        ["Electromagnetic torque"                             , "T_e"               , float(prob.get_val("T_e", units="kN*m")), "kN*m"],
        ["Rated torque"                                       , "T_rated"           , float(prob.get_val("T_rated", units="kN*m")), "kN*m"],
        ["torque constraint"                                  , "torque_ratio"      , float(prob.get_val("torque_ratio")), "-", "1.0 < x < 1.2"],
        ["Copper losses"                                      , "P_Cu"              , float(prob.get_val("P_Cu", units="kW")), "kW"],
        ["specific eddy losses W/kg @ 1.5 T"                  , "P_Fe0e"            , float(prob.get_val("P_Fe0e")), "-"],
        ["specific hysteresis losses W/kg @ 1.5 T"            , "P_Fe0h"            , float(prob.get_val("P_Fe0h")), "-"],
        ["Magnet losses"                                      , "P_Ftm"             , float(prob.get_val("P_Ftm", units="kW")), "kW"],
        ["Total loss"                                         , "Losses"            , float(prob.get_val("Losses", units="kW")), "kW"],
        ["Generator efficiency"                               , "gen_eff"           , float(prob.get_val("gen_eff")), "", "0.95 < x"],
        ["Generator output frequency"                         , "f"                 , float(prob.get_val("f", units="Hz")), "Hz"],
        ["Iron stacking factor"                               , "k_fes"             , float(prob.get_val("k_fes")), "-"],
        ["slot fill factor"                                   , "k_sfil"            , float(prob.get_val("k_sfil")), "-"],
        ["winding factor"                                     , "k_wd"              , float(prob.get_val("k_wd")), "-"],
        ["Bedplate nose outer radius"                         , "R_no"              , float(prob.get_val("R_no", units="m")), "m"],
        ["Main shaft outer radius"                            , "R_sh"              , float(prob.get_val("R_sh", units="m")), "m"],
        ["Main shaft tilt angle"                              , "phi"               , float(prob.get_val("phi", units="deg")), "deg"],
        ["rotor arm circumferential dimension"                , "b_r"               , float(prob.get_val("b_r", units="m")), "m", "(0.1-1.5)"],
        ["stator arm circumferential dimension"               , "b_st"              , float(prob.get_val("b_st", units="m")), "m", "(0.1-1.5)"],
        ["tooth width"                                        , "b_t"               , float(prob.get_val("b_t", units="m")), "m"],
        ["rotor arm depth"                                    , "d_r"               , float(prob.get_val("d_r", units="m")), "m", "(0.1-1.5)"],
        ["stator arm depth"                                   , "d_s"               , float(prob.get_val("d_s", units="m")), "m", "(0.1-1.5)"],
        ["Rotor rim thickness"                                , "h_sr"              , float(prob.get_val("h_sr", units="m")), "m", "(0.045-0.25)"],
        ["Stator rim thickness"                               , "h_ss"              , float(prob.get_val("h_ss", units="m")), "m", "(0.045-0.25)"],
        ["Rotor arms"                                         , "n_r"               , float(prob.get_val("n_r")), "-", "(5-15)"],
        ["Stator arms"                                        , "n_s"               , float(prob.get_val("n_s")), "-", "(5-15)"],
        ["rotor arm thickness"                                , "t_wr"              , float(prob.get_val("t_wr", units="m")), "m", "(0.001-0.2)"],
        ["stator arm thickness"                               , "t_ws"              , float(prob.get_val("t_ws", units="m")), "m", "(0.001-0.2)"],
        ["Stator disc thickness"                              , "t_s"               , float(prob.get_val("t_s", units="m")), "m"],
        ["Partial safety factor"                              , "gamma"             , float(prob.get_val("gamma")), "", ""],
        ["Rotor radial deflection"                            , "u_ar"              , float(prob.get_val("u_ar", units="m")), "m"],
        ["Stator radial deflection"                           , "u_as"              , float(prob.get_val("u_as", units="m")), "m"],
        ["Allowable radial deflection percent"                , "u_allow_pcent"     , float(prob.get_val("u_allow_pcent")), "-"],
        ["Rotor axial deflection"                             , "y_ar"              , float(prob.get_val("y_ar", units="m")), "m"],
        ["Stator axial deflection"                            , "y_as"              , float(prob.get_val("y_as", units="m")), "m"],
        ["Allowable axial deflection percent"                 , "y_allow_pcent"     , float(prob.get_val("y_allow_pcent")), "-"],
        ["Rotor circumferential deflection"                   , "z_ar"              , float(prob.get_val("z_ar", units="m")), "m"],
        ["Stator circumferential deflection"                  , "z_as"              , float(prob.get_val("z_as", units="m")), "m"],
        ["Allowable torsional twist"                          , "z_allow_deg"       , float(prob.get_val("z_allow_deg", units="deg")), "deg"],
        ["Circumferential arm space constraint"               , "con_bar"           , float(prob.get_val("con_bar")), "-", "< 1.0"],
        ["Circumferential arm space constraint"               , "con_bas"           , float(prob.get_val("con_bas")), "-", "< 1.0"],
        ["Radial deflection constraint-rotor"                 , "con_uar"           , float(prob.get_val("con_uar")), "-", "< 1.0"],
        ["Radial deflection constraint-rotor"                 , "con_uas"           , float(prob.get_val("con_uas")), "-", "< 1.0"],
        ["Axial deflection constraint-rotor"                  , "con_yar"           , float(prob.get_val("con_yar")), "-", "< 1.0"],
        ["Axial deflection constraint-rotor"                  , "con_yas"           , float(prob.get_val("con_yas")), "-", "< 1.0"],
        ["Torsional deflection constraint-rotor"              , "con_zar"           , float(prob.get_val("con_zar")), "-", "< 1.0"],
        ["Torsional deflection constraint-rotor"              , "con_zas"           , float(prob.get_val("con_zas")), "-", "< 1.0"],
        ["Copper density"                                     , "rho_Copper"        , float(prob.get_val("rho_Copper", units="kg/m**3")), "kg/m**3"],
        ["Electrical steel density"                           , "rho_Fe"            , float(prob.get_val("rho_Fe", units="kg/m**3")), "kg/m**3"],
        ["Structural Steel densit"                            , "rho_Fes"           , float(prob.get_val("rho_Fes", units="kg/m**3")), "kg/m**3"],
        ["magnet mass density"                                , "rho_PM"            , float(prob.get_val("rho_PM", units="kg/m**3")), "kg/m**3"],
        ["Rotor yoke mass"                                    , "M_Fery"            , float(prob.get_val("M_Fery", units="t")), "t"],
        ["Stator teeth mass"                                  , "M_Fest"            , float(prob.get_val("M_Fest", units="t")), "t"],
        ["Stator yoke mass"                                   , "M_Fesy"            , float(prob.get_val("M_Fesy", units="t")), "t"],
        ["Iron mass"                                          , "mass_iron"              , float(prob.get_val("mass_iron", units="t")), "t"],
        ["Iron mass"                                          , "mass_Fe"           , float(prob.get_val("mass_Fe", units="t")), "t"],
        ["Magnet mass"                                        , "mass_PM"           , float(prob.get_val("mass_PM", units="t")), "t"],
        ["Copper Mass"                                        , "mass_copper"            , float(prob.get_val("mass_copper", units="t")), "t"],
        ["Active mass"                                        , "mass_active"       , float(prob.get_val("mass_active", units="t")), "t"],
        ["Mass to add to total for unaccounted elements"      , "mass_adder"        , float(prob.get_val("mass_adder", units="t")), "t"],
        ["Rotor structural mass"                              , "mass_structural_rotor"  , float(prob.get_val("mass_structural_rotor", units="t")), "t"],
        ["Stator structural mass"                             , "mass_structural_stator" , float(prob.get_val("mass_structural_stator", units="t")), "t"],
        ["Total structural mass"                              , "mass_structural"   , float(prob.get_val("mass_structural", units="t")), "t"],
        ["Specific cost of copper"                            , "C_Cu"              , float(prob.get_val("C_Cu", units="USD/kg")), "USD/kg"],
        ["Specific cost of magnetic steel/iron"               , "C_Fe"              , float(prob.get_val("C_Fe", units="USD/kg")), "USD/kg"],
        ["Specific cost of structural steel"                  , "C_Fes"             , float(prob.get_val("C_Fes", units="USD/kg")), "USD/kg"],
        ["Specific cost of Magnet"                            , "C_PM"              , float(prob.get_val("C_PM", units="USD/kg")), "USD/kg"],
        ["Cost to add to total for unaccounted elements"      , "cost_adder"        , 1e-3*float(prob.get_val("cost_adder", units="USD")), "k$"],
        ["Total mass"                                         , "mass_total"        , float(prob.get_val("mass_total", units="t")), "t"],
        ["Total cost"                                         , "cost_total"        , 1e-3*float(prob.get_val("cost_total", units="USD")), "k$"],
    ]

    df = pd.DataFrame(raw_data, columns=["Parameters", "Symbol", "Values", "Units", "Limit"])
    df.to_excel(os.path.join(output_dir, f"Optimized_MS-PMSG_{ratingMW}_MW.xlsx"))

def run_all(output_str, opt_flag, obj_str, ratingMW):
    output_dir = os.path.join(mydir, output_str)

    # Optimize just magnetrics with GA and then structural with SLSQP
    prob=None
    #prob = optimize_magnetics_design(output_dir=output_dir, opt_flag=True, obj_str=obj_str, ratingMW=int(ratingMW), restart_flag=True, femm_flag=False)
    prob = optimize_magnetics_design(output_dir=output_dir, opt_flag=opt_flag, obj_str=obj_str, ratingMW=int(ratingMW), prob_in=prob, femm_flag=True, cleanup_flag=False)
    prob_struct = optimize_structural_design(prob_in=prob, output_dir=output_dir, opt_flag=True)

    # Bring all data together
    for k in ["h_sr","h_ss","n_r","n_s","b_r","d_r","t_wr","t_ws","b_st","d_s"]:
        prob[k]  = prob_struct[k]
    prob.run_model()

    # Write to xlsx and csv files
    #prob.model.list_inputs(val=True, hierarchical=True, units=True, desc=True)
    #prob.model.list_outputs(val=True, hierarchical=True)
    write_all_data(prob, output_dir=output_dir)
    cleanup_femm_files(mydir, output_dir)

if __name__ == "__main__":
    opt_flag = True
    #run_all("outputs15-mass", opt_flag, "mass", 15)
    #run_all("outputs17-mass", opt_flag, "mass", 17)
    run_all("outputs20-mass", opt_flag, "mass", 20)
    #run_all("outputs25-mass", opt_flag, "mass", 25)
    #run_all("outputs15-cost", opt_flag, "cost", 15)
    #run_all("outputs17-cost", opt_flag, "cost", 17)
    #run_all("outputs20-cost", opt_flag, "cost", 20)
    #run_all("outputs25-cost", opt_flag, "cost", 25)
    #for k in ratings_known:
    #    for obj in ["cost", "mass"]:
    #        run_all(f"outputs{k}-{obj}", opt_flag, obj, k)
