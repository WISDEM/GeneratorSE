import openmdao.api as om
from generatorse.ipm.IPM import PMSG_Outer_rotor_Opt
from generatorse.ipm.structural import PMSG_Outer_Rotor_Structural
import os
import pandas as pd
import numpy as np

def cleanup_femm_files(clean_dir):
    files = os.listdir(clean_dir)
    for file in files:
        if file.endswith(".ans") or file.endswith(".fem") or file.endswith(".csv"):
            os.remove(os.path.join(clean_dir, file))
            
            
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
    df.to_excel(froot + ".xlsx", index=False)
    df.to_csv(froot + ".csv", index=False)            

if __name__ == "__main__":

    

    
    
    
    mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
    output_dir = os.path.join(mydir, 'outputs', 'test11')
    os.makedirs(output_dir, exist_ok=True)



    cleanup_flag = True
    # Clean run directory before the run
    if cleanup_flag:
        cleanup_femm_files(mydir)

    prob = om.Problem()
    prob.model = PMSG_Outer_rotor_Opt()

    prob.driver = om.ScipyOptimizeDriver()  # pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'COBYLA' #'SLSQP' #
    prob.driver.options["maxiter"] = 500 #50

    recorder = om.SqliteRecorder(os.path.join(output_dir,"log.sql"))
    prob.driver.add_recorder(recorder)
    prob.add_recorder(recorder)
    prob.driver.recording_options["excludes"] = ["*_df"]
    prob.driver.recording_options["record_constraints"] = True
    prob.driver.recording_options["record_desvars"] = True
    prob.driver.recording_options["record_objectives"] = True

    prob.model.add_constraint('B_rymax',upper=2.53)									#2
#    prob.model.add_constraint('K_rad',    lower=0.15,upper=0.3)						#10
    prob.model.add_constraint('gen_eff',lower=0.955)						            #14
    prob.model.add_constraint("E_p_ratio", lower=0.9, upper=1.10)
    prob.model.add_constraint("torque_ratio", lower=0.97, upper=1.05) 
    #structural constraints
#    prob.model.add_constraint('con_uar',lower = 1e-2)  #1e-2           #17
#    prob.model.add_constraint('con_yar', lower = 1e-2)                             #18
#    prob.model.add_constraint('con_uas', lower = 1e-2)                             #20
#    prob.model.add_constraint('con_yas',lower = 1e-2)                             #21
   
    
    prob.model.add_objective('cost_total',scaler=1e3)
    
    prob.model.add_design_var('D_a', lower=6, upper=10.5, ref=8.0 ) 
    prob.model.add_design_var('g', lower=0.007, upper=0.015 ) 
    prob.model.add_design_var('l_s', lower=1.5, upper=2.5, ref=2.0)  
    prob.model.add_design_var('h_t', lower=0.15, upper=0.35 )  
    prob.model.add_design_var('p', lower=70, upper=260, ref=75.0)
    #prob.model.add_design_var('l_fe_ratio', lower=0.1, upper=0.9 ) 
    prob.model.add_design_var('alpha_v',lower=60,upper=160)
    prob.model.add_design_var('N_c',lower=2,upper=32)
    prob.model.add_design_var('I_s',lower=2900,upper=8500, ref=5000)
    prob.model.add_design_var('d_mag',lower=0.05,upper=0.25, ref=0.225)
    #prob.model.add_design_var('d_sep',lower=0.00,upper=0.020, ref=0.0125)
    prob.model.add_design_var('h_m', lower=0.010, upper=0.050)
    prob.model.add_design_var('m_sep', lower=0.005, upper=0.01)

    prob.model.add_design_var('ratio',lower=0.7,upper=1.0)
    prob.model.add_design_var('J_s',lower=3,upper=6,ref=4.5)
    
    prob.model.add_design_var('h_yr', lower=0.025, upper=0.15 )
    prob.model.add_design_var('h_ys', lower=0.025, upper=0.15 )
    prob.model.add_design_var('t_r', lower=0.05, upper=0.3 ) 
    prob.model.add_design_var('t_s', lower=0.05, upper=0.3 )  
    prob.model.add_design_var('h_ss', lower=0.04, upper=0.2)
    prob.model.add_design_var('h_sr', lower=0.04, upper=0.2)
    
    
    prob.setup()
    
	#Initial design variables for a PMSG designed for a 15MW turbine
    
    prob['P_rated']     =   17000000.0
    
    prob['T_rated']     =   23.03066         #rev 1 9.94718e6
    
    prob['E_p_target']     =  3300
    
    prob['N_nom']       =   7.7  #8.68                # rpm 9.6
    
    prob['D_a']         =   9.0 # rev 1 6.8
   
    prob['l_s']         =   2.5	# rev 2.1
    
    prob['h_t']         =  0.2 # rev 1 0.3
    
    prob['p']           =   70#100.0    # rev 1 160
    prob['g']           =   0.007#0.00627 #100.0    # rev 1 160
    
  
    prob['h_ys']        =   0.05# rev 1 0.045
    
    prob['h_yr']        =   0.025 # rev 1 0.045
    
    prob['alpha_v']       =110#50.013#100.0    # rev 1 160
    
    prob['N_c']           =   6#100.0    # rev 1 160
    
    prob['I_s']           =   3500 #100.0    # rev 1 160
    
    prob["h_m"]         =   0.015
    
    prob["d_mag"]         =   0.08
    
    prob["d_sep"]         =   0.010*0
    
    prob["m_sep"]     =  0.005
    #prob['alpha_m']           =  0.7#100.0    # rev 1 160
    
    prob['l_fe_ratio']           =  0.85#100.0    # rev 1 160
    
    prob['ratio']        =   1.0# rev 1 0.045

  
    
    prob['J_s']           =   6 #100.0    # rev 1 160
    
    prob['phi']           =   90 #100.0    # rev 1 160 
    prob['b']   =   2.
    
    prob['c']   =5.0

#	
    #Specific costs
    prob['C_Cu']        =   10.3
    prob['C_Fe']    	=   1.02
    prob['C_Fes']       =   1.0
    prob['C_PM']        =   110.0
    
    
    #Material properties
    
    prob['rho_Fe']      =   7700.0                 #Steel density
    prob['rho_Fes']     =   7850.0                 #Steel density
    prob['rho_Copper']  =   8900.0                  # Kg/m3 copper density
    prob['rho_PM']      =   7600.0                  # magnet density
    prob['resistivity_Cu']      =   1.8*10**(-8)*1.4			# Copper resisitivty
    
    #Support structure parameters
    prob['R_no']        =0.925		# Nose outer radius
    prob['R_sh']        = 1.25			# Shaft outer radius =(2+0.25*2+0.3*2)*0.5
    
    prob['t_r']         =   0.06 			# Rotor disc thickness
    prob['h_sr']        =   0.04           # Rotor cylinder thickness

    prob['t_s']         =   0.06 			# Stator disc thickness
    prob['h_ss']        =   0.04           # Stator cylinder thickness
    
    prob['y_sh']         =   0.0005*0			# Shaft deflection
    prob['theta_sh']     =   0.00026*0      # Slope at shaft end

    prob['y_bd']         =   0.0005*0# deflection at bedplate
    prob['theta_bd']     =  0.00026*0   # Slope at bedplate end
    
    prob['u_allow_pcent']=  8.5       # % radial deflection
    prob['y_allow_pcent']=  1.0       # % axial deflection
    
    prob['z_allow_deg']  =  0.05       # torsional twist

      

        
    prob.model.approx_totals(method='fd')    
    
    prob.run_model()
    #prob.run_driver()
    
    #prob.model.list_outputs(values = True, hierarchical=True)
    
        # Clean run directory after the run
    if cleanup_flag:
        cleanup_femm_files(mydir)

    #prob.model.list_outputs(values = True, hierarchical=True)
    
    raw_data = {
        "Parameters": [
            'Rating',
            'Air gap diameter',
            'Overall Outer diameter',
            "Air gap length",
            'Stator length',
            'l/d ratio',
            'Slot_aspect_ratio',
            'Pole pitch',
            'Slot pitch',
            'Stator slot height',
            'Stator slotwidth',
            'Stator tooth width',
            'Stator tooth height',
            'Stator yoke height', 
            'Rotor yoke height',
            'Magnet height',
            'Magnet width',
            "V-angle",
            "Barrier distance",
            "Barrier ratio",
            'Peak air gap flux density fundamental',
            'Peak statorflux density',
            'Peak rotor yoke flux density',
            'Pole pairs', 
            'Generator output frequency',
            'Generator output phase voltage',
            'Generator Output phase current',
            'Stator resistance',
            'Stator slots',
            'Stator turns',
            'Conductor cross-section',
            'Stator Current density ',
            'Electromagnetic Torque',
            'Shear stress',
            'Normal stress',
            'Generator Efficiency ',
            'Iron mass',
            'Magnet mass',
            'Copper mass',
            'Structural Mass',
            '**************************',
            'Rotor disc thickness',
            'Rotor rim thickness',
            'Stator disc thickness',
            'Stator rim thickness',
            'Rotor radial deflection',
            'Rotor axial deflection',
            'Stator radial deflection',
            'Stator axial deflection',
            '**************************',
            'Total Mass',
            'Total Material Cost',
             ],
        "Values": [
            prob['P_rated']/1000000,
            2*prob['r_g'],
            prob['D_outer'],
            prob.get_val("g",units='mm'),
            prob['l_s'],
            prob['K_rad'],
            prob['Slot_aspect_ratio'],
            prob.get_val("tau_p",units='mm'),
            prob.get_val("tau_s",units='mm'),
            prob.get_val("h_s",units='mm'),
            prob.get_val("b_s",units='mm'),
            prob.get_val("b_t",units='mm'),
            prob.get_val("h_t",units='mm'),
            prob.get_val("h_ys",units='mm'),
            prob.get_val("h_yr",units='mm'),
            prob.get_val("h_m",units='mm'),
            prob.get_val("l_m",units='mm'),
            prob.get_val("alpha_v",units='deg'),
            prob.get_val("w_fe",units='mm'),
            prob.get_val("l_fe_ratio"),
            prob['B_g'],
            prob['B_smax'],
            prob['B_rymax'],
            prob['p1'],
            prob['f'],
            prob['E_p'],
            prob['I_s'],
            prob['R_s'],
            prob['S'],
            prob['N_c'],
            prob['A_Cuscalc'],
            prob['J_s'],
            prob.get_val("T_e",units='MN*m'),
            prob.get_val("Sigma_shear",units='kN/m**2'),
            prob.get_val("Sigma_normal",units='kN/m**2'),
            prob['gen_eff']*100,
            prob.get_val("Iron",units='t'),
            prob.get_val("mass_PM",units='t'),
            prob.get_val("Copper",units='t'),
            prob.get_val("Structural_mass",units='t'),
            '************************',
            prob.get_val("t_r",units='mm'),
            prob.get_val("h_sr",units='mm'),
            prob.get_val("t_s",units='mm'),
            prob.get_val("h_ss",units='mm'),
            prob.get_val("u_ar",units='mm'),
            prob.get_val("y_ar",units='mm'),
            prob.get_val("u_as",units='mm'),
            prob.get_val("y_as",units='mm'),
            '**************************',
            prob.get_val("mass_total",units='t'),
            prob['cost_total']/1000,
            ],
        'Limit': [
            '',
            '',
            '',
            '',
            "",
            '(0.15-0.3)',
            '(4-10)',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '<2',
            '<2',
            '<2',
            '',
            '',
            '',
            prob['E_p_target'],
            '',
            '',
            '',
            '',
            '',
            '3-6',
            '',
            '',
            '',
            '>=95.4',
            '',
            '',
            '',
            '',
            '************************',
            '',
            '',
            '',
            '',
            prob.get_val("u_allowable_r",units='mm'),
            prob.get_val("y_allowable_r",units='mm'),
            prob.get_val("u_allowable_s",units='mm'),
            prob.get_val("y_allowable_s",units='mm'),
            '**************************',
            '',
            '',
            ],
        'Units':[
            'MW',
            'm',
            'm',
            'mm',
            'm',
            '',
            '',
            'mm',
            'mm',
            'mm',
            'mm',
            'mm',
            'mm',
            'mm',
            'mm',
            'mm',
            'mm',
            'deg',
            'mm',
            '',
            'T',
            'T',
            'T',
            '-',
            'Hz',
            'V',
            'A',
            'ohm/phase',
            'slots',
            'turns',
            'mm^2',
            'A/mm^2',
            'MNm',
            'kPa',
            'kPa',
            '%',
            'tons',
            'tons',
            'tons',
            'tons',
            '************',
            'mm',
            'mm',
            'mm',
            'mm',
            'mm',
            'mm',
            'mm',
            'mm',
            '************',
            'tons',
            'k$',
            ]}
    
    df=pd.DataFrame(raw_data, columns=['Parameters','Values','Limit','Units'])
    
    print (df)
    
    df.to_excel('Optimized_IPM_PMSG_'+str(prob['P_rated'][0]/1e6)+'_MW.xlsx')

    print("Final solution:")
    print("E_p_ratio", prob["E_p_ratio"])
    #print("con_angle", prob["con_angle"])
    print("gen_eff", prob["gen_eff"])
    print("Torque_actual", prob["T_e"])
