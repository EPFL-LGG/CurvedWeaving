import sys
from gh_optimizer_helper import generate_grasshopper_data_gh
from gh_optimization_diagram import optimize_model
from io_redirection import suppress_stdout as so
model_name = sys.argv[1]

if optimize_model(model_name):
    print("\nOptimized: " + model_name)
    print("\nGenerating Rhino/Grasshopper Geometry...")
    generate_grasshopper_data_gh(model_name)
        