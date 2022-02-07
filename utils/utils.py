import datetime
import pandas as pd
from model.person import Person
from model.shift import Shift


def shifts_overlap(s1, s2):
    """
    Compute whether the 2 shifts overlap and
    also which one begins earlier
    :param s1: first shift
    :param s2: second shift
    :return: True if the 2 shifts overlap,
    False otherwise, tuple (shift beginning earlier, shift beginning after)
    """
    do_overlap = s1.start <= s2.start < s1.end or s2.start <= s1.start < s2.end
    if s1.start <= s2.start:
        return do_overlap, s1, s2
    else:
        return do_overlap, s2, s1


def load_persons(file):
    """
    Load the persons file and
    instantiate them.
    :param file: path of the file
    :return: list of Person
    """
    df = pd.read_csv(file, usecols=['nom', 'age', 'responsable'])

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
    df = pd.read_csv(file, usecols=['id', 'debut', 'fin', 'nombre', 'majeur'])
    df['debut'] = pd.to_datetime(df['debut'])
    df['fin'] = pd.to_datetime(df['fin'])

    # Convert major only column
    df['majeur'] = df['majeur'].apply(lambda x: True if x == 'oui' else False)

    # Fix a reference time
    ref_time = datetime.datetime.now()

    return [Shift(row['id'], (row['debut'] - ref_time).seconds, (row['fin'] - ref_time).seconds, row['nombre'], row['majeur']) for _, row in df.iterrows()]


def load_availabilities(file):
    pass


def save_schedule(file, assignments):
    """
    Save the solution schedule as a CSV.
    :param file: path of the csv
    :param assignments: dict of the form
    shift object: list of persons objects
    :return: -
    """
    data = {}

    for shift, persons in assignments.items():
        data[shift.id] = ', '.join(persons)

    df = pd.DataFrame.from_dict(data, orient='index', columns=['personnes'])
    df.to_csv(file)
