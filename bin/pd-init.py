#!/usr/bin/env python

import os

import palladio

import shutil

import argparse


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "destination_file", nargs='?', default="konfiguration.py")

    return parser.parse_args()


def main():

    args = parse_args()

    template_file = os.path.join(
        os.path.dirname(palladio.__file__),
        "config", "konfiguration.template.py")

    proceed = True

    if os.path.exists(args.destination_file):

        # Print a warning message and ask for confirmation
        print(
            f"Warning! File {args.destination_file} already exists! Overwrite? [y/N]")  # noqa
        answer = input()

        # Decide whether to proceed based on answer
        proceed = answer.upper() == 'Y'

    if proceed:
        shutil.copy(template_file, args.destination_file)
    else:
        print("Aborted")


if __name__ == "__main__":
    main()
