class Shift:
    """
    A shift is defined by :
    - id
    - start time (wrt reference time)
    - end time (wrt reference time)
    - number of required persons
    - whether it requires major persons only
    """
    def __init__(self, id, start, end, nb_persons, is_major_only):
        self.id = id
        self.start = start
        self.end = end
        self.nb_persons = nb_persons
        self.is_major_only = is_major_only

    def __str__(self):
        return f'{self.id}, ({self.start}, {self.end}), {self.nb_persons} persons, {self.is_major_only}'