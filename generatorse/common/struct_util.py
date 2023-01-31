
import numpy as np
import openmdao.api as om



def shell_constant(R, t, l, x, v, E):

    Lambda = (3 * (1 - v**2) / (R**2 * t**2)) ** 0.25
    D = E * t**3 / (12 * (1 - v**2))
    C_14 = (np.sinh(Lambda * l)) ** 2 + (np.sin(Lambda * l)) ** 2
    C_11 = (np.sinh(Lambda * l)) ** 2 - (np.sin(Lambda * l)) ** 2
    F_2 = np.cosh(Lambda * x) * np.sin(Lambda * x) + np.sinh(Lambda * x) * np.cos(Lambda * x)
    C_13 = np.cosh(Lambda * l) * np.sinh(Lambda * l) - np.cos(Lambda * l) * np.sin(Lambda * l)
    F_1 = np.cosh(Lambda * x) * np.cos(Lambda * x)
    F_4 = np.cosh(Lambda * x) * np.sin(Lambda * x) - np.sinh(Lambda * x) * np.cos(Lambda * x)

    return D, Lambda, C_14, C_11, F_2, C_13, F_1, F_4


def plate_constant(a, b, v, r_o, t, E):

    D = E * t**3 / (12 * (1 - v**2))
    C_2 = 0.25 * (1 - (b / a) ** 2 * (1 + 2 * np.log(a / b)))
    C_3 = 0.25 * (b / a) * (((b / a) ** 2 + 1) * np.log(a / b) + (b / a) ** 2 - 1)
    C_5 = 0.5 * (1 - (b / a) ** 2)
    C_6 = 0.25 * (b / a) * ((b / a) ** 2 - 1 + 2 * np.log(a / b))
    C_8 = 0.5 * (1 + v + (1 - v) * (b / a) ** 2)
    C_9 = (b / a) * (0.5 * (1 + v) * np.log(a / b) + 0.25 * (1 - v) * (1 - (b / a) ** 2))
    L_11 = (1 / 64) * (1 + 4 * (b / a) ** 2 - 5 * (b / a) ** 4 - 4 * (b / a) ** 2 * (2 + (b / a) ** 2) * np.log(a / b))
    L_17 = 0.25 * (1 - 0.25 * (1 - v) * ((1 - (r_o / a) ** 4) - (r_o / a) ** 2 * (1 + (1 + v) * np.log(a / r_o))))

    L_14 = 1 / 16 * (1 - (b / a) ** 2 - 4 * (b / a) ** 2 * np.log(a / b))

    return D, C_2, C_3, C_5, C_6, C_8, C_9, L_11, L_17, L_14


class structural_constraints(om.ExplicitComponent):
    def setup(self):

        self.add_input("gamma", 1.5, desc="partial safety factor")

        self.add_input("b_r", 0.0, units="m", desc="arm width b_r")
        self.add_input("u_ar", 0.0, units="m", desc="Rotor radial deflection")
        self.add_input("y_ar", 0.0, units="m", desc="Rotor axial deflection")
        self.add_input("z_ar", 0.0, units="m", desc="Rotor circumferential deflection")
        self.add_input("u_allowable_r", 1.0, units="m", desc="Allowable radial rotor")
        self.add_input("y_allowable_r", 1.0, units="m", desc="Allowable axial rotor")
        self.add_input("z_allowable_r", 1.0, units="m", desc="Allowable circum rotor")
        self.add_input("b_allowable_r", 1.0, units="m", desc="Allowable arm dimensions")

        self.add_input("b_st", 0.0, units="m", desc="arm width b_r")
        self.add_input("u_as", 0.0, units="m", desc="Rotor radial deflection")
        self.add_input("y_as", 0.0, units="m", desc="Rotor axial deflection")
        self.add_input("z_as", 0.0, units="m", desc="Rotor circumferential deflection")
        self.add_input("u_allowable_s", 1.0, units="m", desc="Allowable radial rotor")
        self.add_input("y_allowable_s", 1.0, units="m", desc="Allowable axial rotor")
        self.add_input("z_allowable_s", 1.0, units="m", desc="Allowable circum rotor")
        self.add_input("b_allowable_s", 1.0, units="m", desc="Allowable arm dimensions")

        self.add_output('con_bar', val=0.0, desc='Circumferential arm space constraint (<1)')
        self.add_output('con_uar', val=0.0, desc='Radial deflection constraint-rotor (<1)')
        self.add_output('con_yar', val=0.0, desc='Axial deflection constraint-rotor (<1)')
        self.add_output('con_zar', val=0.0, desc='Torsional deflection constraint-rotor (<1)')

        self.add_output('con_bas', val=0.0, desc='Circumferential arm space constraint (<1)')
        self.add_output('con_uas', val=0.0, desc='Radial deflection constraint-rotor (<1)')
        self.add_output('con_yas', val=0.0, desc='Axial deflection constraint-rotor (<1)')
        self.add_output('con_zas', val=0.0, desc='Torsional deflection constraint-rotor (<1)')

        self.declare_partials("*", "*", method="fd")


    def compute(self, inputs, outputs):
        gm = float(inputs["gamma"])
        
        # Constraint outputs
        outputs["con_bar"] = gm * np.abs(inputs["b_r"])  / inputs["b_allowable_r"]
        outputs["con_uar"] = gm * np.abs(inputs["u_ar"]) / inputs["u_allowable_r"]
        outputs["con_yar"] = gm * np.abs(inputs["y_ar"]) / inputs["y_allowable_r"]
        outputs["con_zar"] = gm * np.abs(inputs["z_ar"]) / inputs["z_allowable_r"]

        # Constraint outputs
        outputs["con_bas"] = gm * np.abs(inputs["b_st"]) / inputs["b_allowable_s"]
        outputs["con_uas"] = gm * np.abs(inputs["u_as"]) / inputs["u_allowable_s"]
        outputs["con_yas"] = gm * np.abs(inputs["y_as"]) / inputs["y_allowable_s"]
        outputs["con_zas"] = gm * np.abs(inputs["z_as"]) / inputs["z_allowable_s"]
