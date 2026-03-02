import argparse
import datetime
import logging
import pathlib
import sys
import typing
import yaml
from ._file_system_interaction import is_valid_file

def get_parser(description: str) -> argparse.ArgumentParser:
    """
    Function that generates the argument parser for the processor. Here, all
    the arguments and help message are defined.
    
    Parameters
    ----------
    description : str
        Text description to include in the help message of the script.
    
    Returns
    -------
    argparse.ArgumentParser
        The created argument parser.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--conf",
        "-c",
        dest="config_file_path",
        required=True,
        metavar="FILE",
        type=lambda x: is_valid_file(x),  # type: ignore
        help="The YAML configuration file.",
    )
    return parser

def logged_main(description: str, main_fn: typing.Callable) -> None:
    """
    Function that wraps around your main function adding logging capabilities
    and basic configuration with a yaml file passed with `-c <config_file_path>`
    
    Parameters
    ----------
    description : str
        Text description to include in the help message of the script.
    main_fn : typing.Callable
        Main function to execute
    """
    start_time = datetime.datetime.now()

    # ---- Parsing
    parser = get_parser(description)
    args = parser.parse_args()

    # ---- Loading configuration file
    config_file = args.config_file_path
    with open(config_file) as yaml_file:
        config = yaml.full_load(yaml_file)

    # ---- Config

    # Log folder
    log_folder = config["log_dir"]
    log_folder = pathlib.Path(log_folder)
    log_folder.mkdir(exist_ok=True, parents=True)
    # Log level
    log_level = config["log_level"]

    # ---- Logging

    # Change root logger level from WARNING (default) to NOTSET in order for all
    # messages to be delegated.
    logging.getLogger().setLevel(logging.NOTSET)
    log_name = f"{main_fn.__name__}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%s')}.log"
    log_filename = log_folder / log_name

    # Add stdout handler, with level defined by the config file (i.e., print log
    # messages to screen).
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.getLevelName(log_level.upper()))
    frmttr_console = logging.Formatter("%(asctime)s [%(name)s-%(levelname)s]: %(message)s")
    console_handler.setFormatter(frmttr_console)
    logging.getLogger().addHandler(console_handler)

    # Add file rotating handler, with level DEBUG (i.e., all DEBUG, WARNING and
    # INFO log messages are printed in the log file, regardless of the
    # configuration).
    logfile_handler = logging.FileHandler(filename=log_filename)
    logfile_handler.setLevel(logging.DEBUG)
    frmttr_logfile = logging.Formatter("%(asctime)s [%(name)s-%(levelname)s]: %(message)s")
    logfile_handler.setFormatter(frmttr_logfile)
    logging.getLogger().addHandler(logfile_handler)

    logging.info("Configuration Complete.")

    # ---- Run tuning
    config["log_name"] = log_name
    main_fn(**config)

    # ---- Close Log

    logging.info(f"Close all. Execution time: {datetime.datetime.now()-start_time}")
    logging.getLogger().handlers.clear()
    logging.shutdown()
    