import datetime

import pandas as pd

from model.person import Person
from model.shift import Shift


def load_persons(file):
    """
    Load the persons file and
    instantiate them.
    :param file: path of the file
    :return: list of Person
    """
    df = pd.read_csv(file)

    # Convert responsable columns
    df['responsable'] = df['responsable'].apply(lambda x: True if x == 'oui' else False)

    return [Person(row['nom'], row['age'], row['responsable']) for _, row in df.iterrows()]


def load_shifts(file):
    """
    Load the shifts file and
    instantiate them.
    :param file: path of the file
    :return: list of Shift
    """
    df = pd.read_csv(file)
    df['debut'] = pd.to_datetime(df['debut'])
    df['fin'] = pd.to_datetime(df['fin'])

    # Convert major only column
    df['majeur'] = df['majeur'].apply(lambda x: True if x == 'oui' else False)

    # Fix a reference time
    ref_time = datetime.datetime.now()

    return [Shift(idx, (row['debut'] - ref_time).seconds, (row['fin'] - ref_time).seconds, row['nombre'], row['majeur']) for idx, row in df.iterrows()]
