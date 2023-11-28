#!/usr/bin/env python3
from argparse import ArgumentParser
from utils.parsers import SettingParser
from trainer import Train
import logging
from time import time, sleep, strftime, gmtime
import os


def parse_arguments():
    parser = ArgumentParser(
        'Script for the training of models given a setting file')
    parser.add_argument(dest='root_path',
                        type=str,
                        help='Path to the run directory.')
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        help=r'''Set the device on which the run must be performed. 
                        Default is set to cuda''')
    parser.add_argument('--jobid',
                        type=str,
                        default='',
                        help=r'''Set the job id of the run. Default is set to None. Useful to run on cluster with slurm''')
    parser.add_argument('--save-memory',
                        action='store_true',
                        help=r'''Set the save memory mode''')
    parser.add_argument('--debug',
                        action='store_true',
                        help=r'''Set the debug mode''')

    return parser.parse_args()


def setup_logger(name, log_file, level=logging.INFO):
    """To set up as many loggers as you want"""

    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(
        '%(asctime)-8s %(message)s', datefmt='%Y%m%d:%H:%M:%S')

    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def get_run_id(job_id: str = ""):
    suffix = ''
    # if parsed_args.jobid != '':
    #     suffix = '.' + parsed_args.jobid
    if job_id != '':
        suffix = '.' + job_id
    run_id = strftime("%Y%m%d_%H%M%S", gmtime()) + suffix
    return run_id


def init_run(run_id: str, root_path, device: str):
    r"""
    Initialize the run by creating the run directory and the logger file
    and choosing the random hyperparameter of the setting file
    """
    # root_path = parsed_args.root_path
    run_path = root_path + '/' + run_id
    os.mkdir(run_path)
    logger = setup_logger(name=run_id, log_file=run_path + f'/training.log')
    logger.info('Loading of the setting file...')
    setting_parser = SettingParser(root_path + '/settings.yml')
    logger.info(f'Running on device {device}')
    setting_parser.update_setting(f'device: {device}')
    logger.info('Saving current configuration...')
    setting_parser.save_setting(run_path + f'/dumped_setting.yml')
    logger.info(str(setting_parser))
    logger.info('Done.')
    return setting_parser, run_path, logger


def start_train():
    # global parsed_args
    parsed_args = parse_arguments()
    main_logger = setup_logger(
        name='main', log_file=parsed_args.root_path + '/single_training.log')
    basic_run_id = get_run_id(parsed_args.jobid)
    run_id = basic_run_id + '_DEBUG' if parsed_args.debug else basic_run_id
    main_logger.info(f'> Start single training with run id: {run_id}')
    main_logger.info(f'>> Run path: {parsed_args.root_path}')
    main_logger.info(f'>> Device: {parsed_args.device}')
    main_logger.info(f'>> Debug mode: {parsed_args.debug}')
    main_logger.info(f'>> Save memory mode: {parsed_args.save_memory}')
    main_logger.info('')

    setting_parser, run_path, logger = init_run(
        run_id, parsed_args.root_path, parsed_args.device,
    )
    logger.info(f'Loading objects...')
    try:
        setting_parser.load_setting()
    except Exception as exception:
        logger.info('Error while loading objects:')
        logger.exception(exception)
        return

    if parsed_args.debug:
        logger.info(
            'Debug mode activated. Run will be performed only for one epoch.')
        logger.info('Verbose model activated.')
        from utils.inspections import VerboseModel
        setting_parser.model = VerboseModel(setting_parser.model)
        setting_parser.update_setting('epochs: 1')

    logger.info('Initializing trainer...')
    # trainer = Train(**vars(instantiated_classes), logger=logger)
    trainer = Train(**setting_parser.to_dict(), logger=logger)
    logger.info('Start training...')
    try:
        save_state_dict = not parsed_args.save_memory
        trainer.run(run_path, save_state_dict=save_state_dict)
    except Exception as exception:
        logger.info('Error while training:')
        logger.exception(exception)
        return


if __name__ == '__main__':
    # import sys
    # debug_mode = hasattr(sys, 'gettrace')
    # if debug_mode:
    #    sys.argv.extend(['data/runs/test_sandwich',
    #                    '--device', 'cpu', '--debug'])
    start_train()
