import datetime

import numpy as np
import pandas as pd
from model.person import Person
from model.shift import Shift

def range_overlap(start1, end1, start2, end2):
    """
    Check whether the 2 ranges overlap.
    :param start1: start time of first range
    :param end1: end time of first range
    :param start2: start time of second range
    :param end2: end time of second range
    :return: True if the 2 ranges overlap,
        False otherwise
    """
    return start1 <= start2 < end1 or start2 <= start1 < end2

def shifts_overlap(s1, s2):
    """
    Compute whether the 2 shifts overlap and
    also which one begins earlier
    :param s1: first shift
    :param s2: second shift
    :return: True if the 2 shifts overlap,
    False otherwise, tuple (shift beginning earlier, shift beginning after)
    """
    do_overlap = range_overlap(s1.start, s1.end, s2.start, s2.end)
    if s1.start <= s2.start:
        return do_overlap, s1, s2
    else:
        return do_overlap, s2, s1


def compute_ref_time(file):
    """
    Compute the reference time for the model
    from the file containing the shifts. The reference time
    is defined as the start time of the earliest
    shift.
    :param file: path of the file
    :return: reference time as datetime
    """
    df = pd.read_csv(file, usecols=['debut'])
    df['debut'] = pd.to_datetime(df['debut'])

    return df['debut'].min()


def compute_time_from_ref_time(time, ref_time):
    """
    Compute the time from the reference time in
    minutes.
    :param time: time in datetime
    :param ref_time: reference time in datetime
    :return: time spend from reference time
        in minutes
    """
    return (time - ref_time).seconds // 60


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
    :return: list of Shift, reference time
    """
    df = pd.read_csv(file, usecols=['id', 'debut', 'fin', 'nombre', 'majeur'])
    df['debut'] = pd.to_datetime(df['debut'])
    df['fin'] = pd.to_datetime(df['fin'])

    # Convert major only column
    df['majeur'] = df['majeur'].apply(lambda x: True if x == 'oui' else False)

    # Fix a reference time as the start of the earliest shift
    ref_time = compute_ref_time(file)

    return [Shift(row['id'],
                  compute_time_from_ref_time(row['debut'], ref_time),
                  compute_time_from_ref_time(row['fin'], ref_time),
                  row['nombre'], row['majeur']) for _, row in df.iterrows()], ref_time


def load_availabilities(file):
    """
    Load the availabilities file and return it as
    a dict. Return None if the file is None
    :param file: path of the file
    :return: dict of the form:
        person name: list of tuples (ranges) (start not available, end no available)
        where the person is NOT available
    """
    if file is not None:
        df = pd.read_csv(file, usecols=['nom', 'debut', 'fin'])
        df['debut'] = pd.to_datetime(df['debut'])
        df['fin'] = pd.to_datetime(df['fin'])

        availabilities = {}
        for _, row in df.iterrows():
            if row['nom'] in availabilities.keys():
                availabilities[row['nom']].append((row['debut'], row['fin']))
            else:
                availabilities[row['nom']] = [(row['debut'], row['fin'])]

        return availabilities

    return None


def generate_availability_matrix(persons, shifts, availabilities, ref_time):
    """
    Build the (nb persons, nb shifts) numpy array representing the
    availabilities of the persons for all the shifts.
    :param persons: list of persons
    :param shifts: list of shifts
    :param availabilities: dict of list of tuples (start not available, end no available)
    :param ref_time: reference time in datetime
    :return: numpy array of shape (nb persons, nb shifts)
        containing True if the person is available at a given shift, False if not
    """
    availability = np.ones((len(persons), len(shifts)), dtype=bool)

    if availabilities is not None:
        for person_name, x in availabilities.items():
            for i, p in enumerate(persons):
                if person_name == p.name:
                    for j, s in enumerate(shifts):
                        for start, end in x:
                            # If the range where the person is not available
                            # and the range of the shift overlap, then
                            # mark as not available
                            start_ = compute_time_from_ref_time(start, ref_time)
                            end_ = compute_time_from_ref_time(end, ref_time)
                            if range_overlap(s.start, s.end, start_, end_):
                                availability[i, j] = False

    return availability


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
