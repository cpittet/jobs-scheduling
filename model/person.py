class Person:
    """
    A person is defined by :
    - name
    - age
    - suited for being in charge or not
    """
    def __init__(self, name, age, can_charge):
        self.name = name
        self.age = age
        self.can_charge = can_charge

    def is_major(self):
        """
        Compute whether this person is major
        :return: True if the person is more than 18 years old,
        False otherwise
        """
        return self.age >= 18

    def __str__(self):
        return f'{self.name}, {self.age}, {self.can_charge}'
