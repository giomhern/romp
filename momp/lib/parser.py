import argparse
import ast
#from datetime import datetime
from pathlib import Path
from importlib import resources

def create_parser(config, cli_args=None):

    parser = argparse.ArgumentParser(description="ROMP Package Parameter Loader")

    #parser.add_argument("-p", "--param", required=True)

    #parser.add_argument(
    #    "-p",
    #    "--param",
    #    type=str,
    #    default="params/config.in",
    #    help="Parameter file path (default: params/config.in)"
    #)

    parser.add_argument(
        "--workflow",
        choices=["onset", "cra", "all"],
        default=config.get("workflow", "onset"),
        help=f"Workflow to run from momp-run (default: {config.get('workflow', 'onset')})"
    )

    parser.add_argument(
        "--model_list",
        nargs="+",
        default=config["model_list"],
        help=f"Model list (default: {config['model_list']})"
    )

    parser.add_argument(
        "--model_dir_list",
        nargs="+",
        default=config["model_dir_list"],
        help=f"Model directories aligned with --model_list (default: {config['model_dir_list']})"
    )

    parser.add_argument(
        "--model_var_list",
        nargs="+",
        default=config["model_var_list"],
        help=f"Model rainfall variables aligned with --model_list (default: {config['model_var_list']})"
    )

    parser.add_argument(
        "--unit_cvt_list",
        nargs="+",
        type=parse_optional_float,
        default=config["unit_cvt_list"],
        help=f"Model unit conversion factors aligned with --model_list (default: {config['unit_cvt_list']})"
    )

    parser.add_argument(
        "--file_pattern_list",
        nargs="+",
        default=config["file_pattern_list"],
        help=f"Model file patterns aligned with --model_list (default: {config['file_pattern_list']})"
    )

    parser.add_argument(
        "--verification_window_list",
        #nargs=2,
        #type=int,
        #action="append",
        #metavar=("START", "END"),
        type=parse_window_list,
        #default=((1, 15), (16, 30)),
        default=config['verification_window_list'],
        help="Verification window as start end day (default: {config['verification_window_list']}), \
                (e.g., '1,15 16,20')"
    )

    parser.add_argument(
        "--max_forecast_day",
        type=int,
        default=config['max_forecast_day'],
        help="Max forecast day (default: {config['max_forecast_day']})"
    )

    parser.add_argument(
        "--wet_init",
        type=float,
        default=config['wet_init'],
        help=f"Rain threshold (mm) for first potential wet day (default: {config['wet_init']})"
    )

    parser.add_argument(
        "--wet_threshold",
        type=float,
        default=config['wet_threshold'],
        help=f"Rainfall threshold (mm) if no thresh_file provided (default: {config['wet_threshold']})"
    )

    parser.add_argument(
        "--wet_spell",
        type=int,
        default=config['wet_spell'],
        help=f"Min days rainfall must stay above threshold (default: {config['wet_spell']})"
    )

    parser.add_argument(
        "--dry_threshold",
        type=float,
        default=config['dry_threshold'],
        help=f"Max rainfall (mm) to define a dry day (default: {config['dry_threshold']})"
    )

    parser.add_argument(
        "--dry_spell",
        type=int,
        default=config['dry_spell'],
        help=f"Max consecutive dry days allowed after onset (default: {config['dry_spell']})"
    )

    parser.add_argument(
        "--dry_extent",
        type=int,
        default=config['dry_extent'],
        help=f"Search window (days) after onset to check for dry spell (default: {config['dry_extent']})"
    )

    parser.add_argument(
        "--onset_percentage_threshold",
        type=float,
        default=config['onset_percentage_threshold'],
        help=f"Probability threshold (0.0-1.0) for ensemble onset (default: {config['onset_percentage_threshold']})"
    )

    # Allowed error margin (in days)
    parser.add_argument(
        "--tolerance_days_list",
        #type=parse_tuple,
        nargs='+',
        type=int,
        action=ForceTupleAction,
        default=config['tolerance_days_list'],
        help=f"Allowed error margins in days e.g., 3 5 (default: {config['tolerance_days_list']})"
    )

    # Time windows for grouping statistics
    parser.add_argument(
        "--day_bins",
        #type=parse_tuple,
        type=parse_window_list,
        default=config['day_bins'],
        help=f"Day bins for statistics e.g., '1,15 16,30' (default: {config['day_bins']})"
    )

    # Date range for evaluation
#    parser.add_argument(
#        "--start_date",
#        #type=parse_date,
#        type=parse_tuple,
#        default=config['start_date'],
#        help=f"Evaluation start date as (YYYY, M, D) (default: {config['start_date']})"
#    )

    parser.add_argument(
        "--start_date",
        nargs=3,
        type=int,
        action=ForceTupleAction,
        default=config['start_date'],
        help="Start date as: YYYY M D (e.g., 2024 5 1)"
    )
    
#    parser.add_argument(
#        "--start_date",
#        type=parse_num_to_tuple,
#        default=config['start_date'],
#        help=f"Start date as 'YYYY M D' (default in config: {config['start_date']})"
#    )

    parser.add_argument(
        "--end_date",
        #type=parse_tuple,
        nargs=3,
        type=int,
        action=ForceTupleAction,
        default=config['end_date'],
        help=f"Evaluation end date as YYYY M D (e.g., 2024 10 31) (default: {config['end_date']})"
    )

    parser.add_argument(
        "--show_plot",
        type=str2bool,
        default=config['show_plot'],
        help="Toggle individual model plot display, True of False (default: {config['show_plot']})"
    )

    parser.add_argument(
        "--show_panel",
        type=str2bool,
        default=config['show_panel'],
        help="Toggle panel plot display, True of False (default: {config['show_panel']})"
    )

    parser.add_argument(
        "--probabilistic",
        type=str2bool,
        #action='store_true' if not config['probabilistic'] else 'store_false',
        default=config['probabilistic'],
        help=f"Ensemble evaluation toggle, True or False (default: {config['probabilistic']})"
    )
    
    parser.add_argument(
        "--debug",
        type=str2bool,
        default=config['debug'],
        help="debug model for developers only, True of False (default: {config['debug']})"
    )

    parser.add_argument(
        "--parallel",
        type=str2bool,
        default=config['parallel'],
        help="parallelization, True of False (default: {config['parallel']})"
    )

    parser.add_argument(
        "--region",
        type=str,
        default=config['region'],
        help="region as defined in params.region_def (default: {config['region']})"
    )

    parser.add_argument(
        "--obs_dir",
        type=str,
        default=config['obs_dir'],
        help=f"Observation NetCDF directory (default: {config['obs_dir']})"
    )

    parser.add_argument(
        "--obs_file_pattern",
        nargs="+",
        default=config['obs_file_pattern'],
        help=f"Observation file pattern; use {{}} or {{year}} for year (default: {config['obs_file_pattern']})"
    )

    parser.add_argument(
        "--obs_var",
        type=str,
        default=config['obs_var'],
        help=f"Observation rainfall variable (default: {config['obs_var']})"
    )

    parser.add_argument(
        "--obs_unit_cvt",
        type=parse_optional_float,
        default=config['obs_unit_cvt'],
        help=f"Observation unit conversion factor (default: {config['obs_unit_cvt']})"
    )

    parser.add_argument(
        "--dir_out",
        type=str,
        default=config['dir_out'],
        help=f"Output data directory (default: {config['dir_out']})"
    )

    parser.add_argument(
        "--dir_fig",
        type=str,
        default=config['dir_fig'],
        help=f"Output figure directory (default: {config['dir_fig']})"
    )

    parser.add_argument(
        "--shpfile_dir",
        type=parse_optional_str,
        default=config['shpfile_dir'],
        help=f"Optional shapefile path or directory for CRA region masking (default: {config['shpfile_dir']})"
    )

    parser.add_argument(
        "--cra_threshold",
        type=float,
        default=config.get('cra_threshold', 20.0),
        help=f"CRA accumulated rainfall threshold in mm (default: {config.get('cra_threshold', 20.0)})"
    )

    parser.add_argument(
        "--cra_max_shift",
        type=int,
        default=config.get('cra_max_shift', 3),
        help=f"CRA maximum grid-cell shift searched in each direction (default: {config.get('cra_max_shift', 3)})"
    )

    parser.add_argument(
        "--cra_init_index",
        type=int,
        default=config.get('cra_init_index', 0),
        help=f"Index into common model initialization dates when --cra_init_date is omitted (default: {config.get('cra_init_index', 0)})"
    )

    parser.add_argument(
        "--cra_init_date",
        type=parse_optional_str,
        default=config.get('cra_init_date'),
        help=f"Explicit CRA initialization date, e.g. 2015-06-06 (default: {config.get('cra_init_date')})"
    )

    parser.add_argument(
        "--cra_save_fig",
        type=str2bool,
        default=config.get('cra_save_fig', True),
        help=f"Save CRA diagnostic figures, True or False (default: {config.get('cra_save_fig', True)})"
    )

#    parser.add_argument(
#        "--ensemble_list",
#        nargs="+",
#        type=int,
#        default=config["ensemble_list"],
#        help=f"Ensemble list (default: {config['ensemble_list']})"
#    )
#
#    parser.add_argument(
#        "--season_start",
#        type=str,
#        default=config["season_start"],
#        help=f"Season start (default: {config['season_start']})"
#    )
#
#    # Boolean argument with dynamic default
#    parser.add_argument(
#        "--MAE",
#        type=bool,
#        default=config["MAE"],
#        help=f"Use MAE (default: {config['MAE']})"
#    )


    #args = parser.parse_args()
    #args, unknown = parser.parse_known_args()

    # cli_args can be passed explicitly; if None, default to sys.argv
    if cli_args is None:
        args, unknown = parser.parse_known_args()
    else:
        args, unknown = parser.parse_known_args(cli_args)

    # ---- Convert list → tuple if needed ----
    # Only convert if user supplied; default already tuple
    tuple_keys = (
        "model_list",
        "model_dir_list",
        "model_var_list",
        "unit_cvt_list",
        "file_pattern_list",
        "obs_file_pattern",
    )
    for key in tuple_keys:
        value = getattr(args, key, None)
        if isinstance(value, list):
            setattr(args, key, tuple(value))

#    if isinstance(args.ensemble_list, list):
#        args.ensemble_list = tuple(args.ensemble_list)

#    return parser
    #print("\n\n\n config:", config["max_forecast_day"])
    #print("\n args:", args.max_forecast_day)
    #print("\n\n config:", config["verification_window_list"])
    #print("\n args:", args.verification_window_list)
    #print(args.start_date)
    #print(args.model_list)
    #print(args.day_bins)
    #print(args.tolerance_days_list)
    #print(args.show_panel)
    #print(args.dry_extent)


    return args



class ForceTupleAction(argparse.Action):
    """Custom action to convert nargs list into a tuple automatically."""
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, tuple(values))


def parse_window_list(string):
    """
    Converts a string input like '1,15 16,20' into ((1, 15), (16, 20))
    """
    try:
        # Split by space to get individual pairs, then split by comma
        pairs = [tuple(map(int, p.split(','))) for p in string.split()]
        return tuple(pairs)
    except Exception:
        raise argparse.ArgumentTypeError("Window list must be in format 'start,end start,end' (e.g., '1,15 16,20')")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_optional_str(value):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in ("none", "null"):
        return None
    return value


def parse_optional_float(value):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in ("none", "null"):
        return None
    return float(value)


# Takes a string like "(3, 5)" and turns it into a Python tuple (3, 5).
def parse_tuple(value):
    """
    Parses a string into its native Python type.
    Example: "(2019, 5, 1)" -> tuple
    Example: "((1, 5), (6, 10))" -> nested tuple
    """
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # Fallback for simple strings if eval fails
        return value


def parse_num_to_tuple(value):
    """
    Converts a space-separated string into a Python tuple of integers.
    CLI: "2024 5 1" -> Python: (2024, 5, 1)
    """
    try:
        # Split string by spaces, convert each part to int, then to tuple
        return tuple(map(int, value.split()))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid format: '{value}'. Expected space-separated integers like '2024 5 1'."
        )



def ensure_config_exists(param_path: str) -> Path:
    """
    Ensure a user-writable config exists at param_path.
    If missing, create it from the packaged template momp/params/config.in.
    """
#    dst = Path(param_path).expanduser()
#
#    if dst.exists():
#        return dst
#
#    dst.parent.mkdir(parents=True, exist_ok=True)
#
#    template = (
#        resources.files("momp")
#        .joinpath("params/config.in")
#        .read_text(encoding="utf-8")
#    )
#
#    dst.write_text(template, encoding="utf-8")
#    return dst

    if param_path == "params/config.in":
        source_tree_config = Path(__file__).resolve().parents[1] / "params" / "config.in"
        if source_tree_config.exists():
            return source_tree_config

        with resources.as_file(resources.files("momp").joinpath("params/config.in")) as p:
            return Path(p)

    return Path(param_path).expanduser().resolve()


## Helper to parse date tuples (YYYY, M, D), python script.py --start_date "(2019, 6, 1)"
## datetime(*d_tuple): The * "unpacks" the tuple. So datetime(*(2019, 5, 1)) becomes datetime(2019, 5, 1)
#def parse_date(value):
#    d_tuple = ast.literal_eval(value)
#    return datetime(*d_tuple).date()



#reserved for python -p option in loader.py need from pathlib import Path 
#def find_param_file(filename):
#    # First, check cwd
#    f = Path(filename)
#    if f.is_file():
#        return f.resolve()
#
#    # Then check package subdirectory 'params' relative to this script
#    package_dir = Path(__file__).parent
#    f2 = package_dir / "params" / filename
#    if f2.is_file():
#        return f2.resolve()
#
#    print(f"Parameter file {filename} not found")
#    sys.exit(1)
