
import numpy as np



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

