import importlib.resources
import os
from pathlib import Path
import argparse
import sys

#from momp.io.input import set_dir
from momp.utils.practical import set_dir
from momp.lib.control import init_dataclass
from momp.lib.convention import Setting

from .parser import create_parser
#from .parser import find_param_file

from types import SimpleNamespace
from momp.lib.assertion import ROMPValidator, ROMPConfigError

from momp.lib.parser import ensure_config_exists


package = "momp"
#base_dir = importlib.resources.files(package)
#base_dir = Path(base_dir).expanduser().as_posix().replace(str(Path.home()), "~")

with importlib.resources.as_file(importlib.resources.files(package)) as p:
    base_dir = Path(p)

print(f"package base dir {base_dir}")

#config_file = set_dir("params/config.in")
#
#if os.path.exists(config_file):
#    print("config_file:", config_file)
#else:
#    print("config_file not found:", config_file)
#    config_file = os.path.join(base_dir, "params/config.in")
#    print("config_file:", config_file)
#    if os.path.exists(config_file):
#        print("config_file found:", config_file)
#    else:
#        print("config_file not found:", config_file)

# -------
def get_config_path_pre_parse():
    # create a pre-parser just to find the '-p' flag
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("-p", "--param", default="params/config.in")
    # parse_known_args ignores all other CLI arguments (like --model_list, etc.)
    args, _ = pre_parser.parse_known_args()
    return args.param

# Use the pre-parsed path instead of the hardcoded one
#requested_path = get_config_path_pre_parse()
config_path = get_config_path_pre_parse()
requested_path = ensure_config_exists(config_path)

##config_file = set_dir(requested_path)
##
##if not os.path.exists(config_file):
##    # Fallback to package directory if local path doesn't exist
##    config_file = os.path.join(base_dir, requested_path)

# 2. Get the "Smart Path" (either a local Path or a Resource Traversable)
config_item = set_dir(requested_path)

# 3. Check if it exists (Both Path and Traversable support .exists())
if not config_item.exists():
    raise FileNotFoundError(f"Could not find: {requested_path}")

# 4. Open the file safely
if isinstance(config_item, Path):
    # It's a normal local file
    with open(config_item, "r") as f:
        params_in = f.read()
else:
    # It's a package resource - use as_file context manager here
    with resources.as_file(config_item) as actual_path:
        with open(actual_path, "r") as f:
            params_in = f.read()
# -------

#with open(config_file, "r") as f:
#    params_in = f.read()

params_in = "\n".join(
    line for line in params_in.splitlines() if not line.strip().startswith("#")
)

exec(params_in)

excluded_vars = {"f", "config_file_path", "params_in"}


#if not Path(globals()["dir_in"]).is_absolute():
#    dir_in = set_dir(globals()["dir_in"])
#

work_dir = (
    Path(globals()["work_dir"])
    .expanduser()
    .resolve()
)

pkg_dir = (
    Path(globals()["pkg_dir"])
    .expanduser()
    .resolve()
)

if not Path(globals()["ref_model_dir"]).is_absolute():
    ref_model_dir = set_dir(globals()["ref_model_dir"])

if not Path(globals()["dir_out"]).is_absolute():
    dir_out = set_dir(globals()["dir_out"])

if not Path(globals()["dir_fig"]).is_absolute():
    dir_fig = set_dir(globals()["dir_fig"])

if not Path(globals()["obs_dir"]).is_absolute():
    obs_dir = set_dir(globals()["obs_dir"])

if globals().get("thresh_file") is not None:
    if not Path(globals()["thresh_file"]).is_absolute():
        thresh_file = set_dir(globals()["thresh_file"])

if globals().get("shpfile_dir") is not None:
    if not Path(globals()["shpfile_dir"]).is_absolute():
        shpfile_dir = set_dir(globals()["shpfile_dir"])

if globals().get("nc_mask") is not None:
    if not Path(globals()["nc_mask"]).is_absolute():
        nc_mask = set_dir(globals()["nc_mask"])


print("work_dir = ", work_dir)
print("pkg_dir = ", pkg_dir)
print("ref_model_dir = ", ref_model_dir)
print("out_dir = ", dir_out)
print("out_fig = ", dir_fig)
print("obs_dir = ", obs_dir)
print("thresh_file = ", thresh_file)
print("shpfile_dir = ", shpfile_dir)
print("nc_mask = ", nc_mask)
#sys.exit()


os.makedirs(dir_fig, exist_ok=True)
os.makedirs(dir_out, exist_ok=True)

# ----
# placeholder --- add dir assertion here 
# ----

# print("dir_in = ",dir_in)

# dic = {var: getattr(config, var) for var in dir(config) if not var.startswith("__")}
# dic = {var: getattr(params_in, var) for var in dir(params_in) if not var.startswith("__")}
# dic = {var: globals()[var] for var in globals() if not var.startswith("__")}

dic = {
    var: globals()[var]
    for var in globals()
    if not var.startswith("__")  # Exclude special Python variables
    and not callable(globals()[var])  # Exclude functions or callable objects
    # and not isinstance(globals()[var], type(os))
    # and isinstance(globals()[var], (str, int, float, bool, Path))
    and var not in globals().get("__builtins__", {})
    and var not in excluded_vars and var != "excluded_vars"
}

#if dic["ref_model"] == "climatology" and dic["model_list"][0] != "climatology":
#    print("\n NOTE 'climatology' is not specified in 'model_list' ")
#    dic["model_list"] = ("climatology",) + dic["model_list"]

# print("\ndic= ", dic)
# print(json.dumps(dic, indent=4))


#parser = create_parser(dic)
#args = parser.parse_args()

#----------  this block below doesn't work with jupyter notebook ---------
#args = create_parser(dic)
#
##this block is to test -p param/param_user.py, need #from .parser import find_param_file
##param_path = find_param_file(args.param)
##print("\n param_path = ", param_path)
##spec = importlib.util.spec_from_file_location("param_module", str(param_path))
##param_module = importlib.util.module_from_spec(spec)
##spec.loader.exec_module(param_module)
##print("\n", param_module.json_structure)
##print("\n", param_module.model_list)
#
## 1. Start with the configuration from the file
#cfg = dic.copy()
#
## 2. Get all parsed arguments as a dictionary
#args_dict = vars(args)
#
## 3. Create a dictionary containing ONLY the key/value pairs that exist
##    in BOTH the parsed arguments AND the original config file (The Controlled Merge)
#overrides = {
#    key: value
#    for key, value in args_dict.items()
#    if key in cfg # Only update keys that were originally in the config
#}
#
## 4. Apply the overrides
#cfg.update(overrides)
#
#
#setting = init_dataclass(Setting, cfg)
#----------  this block above doesn't work with jupyter notebook ---------


_cfg = None
_setting = None


def build_cfg(cli_args=None):
    """
    Build configuration by overlaying CLI arguments on base config.
    """
    args = create_parser(dic, cli_args=cli_args)
#    print("\n\n\n\n dic:", dic["max_forecast_day"])
#    print("\n\n\n\ args:", args.max_forecast_day)

    cfg = dic.copy()
    args_dict = vars(args)

    workflow_keys = {
        "workflow",
        "cra_threshold",
        "cra_max_shift",
        "cra_init_index",
        "cra_init_date",
        "cra_save_fig",
    }
    overrides = {
        key: value
        for key, value in args_dict.items()
        if (key in cfg or key in workflow_keys) and value is not None
    }

    cfg.update(overrides)
    return cfg


def get_cfg(cli_args=None):
    global _cfg
    if _cfg is None:
        _cfg = build_cfg(cli_args)

    try:
        validator = ROMPValidator(_cfg)
        validator.validate()
        print("Configuration validated!")
    except ROMPConfigError as e:
        print(f"Error: Invalid Config!!! {e}")
        if not _cfg.get('debug'):
            #raise
            #raise ROMPConfigError("invalid config")
            sys.exit(1)

    #return _cfg
    return SimpleNamespace(**_cfg)


def get_setting(cli_args=None):
    global _setting
    if _setting is None:
        cfg = get_cfg(cli_args)
        #_setting = init_dataclass(Setting, cfg)
        _setting = init_dataclass(Setting, vars(cfg))
    return _setting

