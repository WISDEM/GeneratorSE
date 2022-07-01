import os

import openmdao.api as om
import matplotlib.pyplot as plt



class Convergence_Trends_Opt(om.ExplicitComponent):
    """
    Deprecating this for now and using OptView from PyOptSparse instead.
    """

    def initialize(self):

        self.options.declare("opt_options")

    def compute(self, inputs, outputs):

        folder_output = self.options["opt_options"]["folder_output"]
        optimization_log = os.path.join(folder_output, self.options["opt_options"]["file_name"])
        if os.path.exists(optimization_log):
            cr = om.CaseReader(optimization_log)
            cases = cr.get_cases()
            rec_data = {}
            design_vars = {}
            responses = {}
            iterations = []
            for i, it_data in enumerate(cases):
                iterations.append(i)

                # Collect DVs and responses separately for DOE
                for design_var in [it_data.get_design_vars()]:
                    for dv in design_var:
                        if i == 0:
                            design_vars[dv] = []
                        design_vars[dv].append(design_var[dv])

                for response in [it_data.get_responses()]:
                    for resp in response:
                        if i == 0:
                            responses[resp] = []
                        responses[resp].append(response[resp])

                # parameters = it_data.get_responses()
                for parameters in [it_data.get_responses(), it_data.get_design_vars()]:
                    for j, param in enumerate(parameters.keys()):
                        if i == 0:
                            rec_data[param] = []
                        rec_data[param].append(parameters[param])

            for param in rec_data.keys():
                try:
                    fig, ax = plt.subplots(1, 1, figsize=(5.3, 4))
                    ax.plot(iterations, rec_data[param])
                    ax.set(xlabel="Number of Iterations", ylabel=param)
                    fig_name = "Convergence_trend_" + param + ".png"
                    fig.savefig(os.path.join(folder_output, fig_name))
                    plt.close(fig)
                except ValueError:
                    pass


class PlotRecorder(om.Group):
    def initialize(self):
        self.options.declare("opt_options")

    def setup(self):
        self.add_subsystem("conv_plots", Convergence_Trends_Opt(opt_options=self.options["opt_options"]))


if __name__ == "__main__":

    opt_options = {}
    opt_options["folder_output"] = "C:\GeneratorSE\examples\outputs17-mass"
    opt_options["file_name"] = "log.sql"

    wt_opt = om.Problem(model=PlotRecorder(opt_options=opt_options))
    wt_opt.setup(derivatives=False)
    wt_opt.run_model()
