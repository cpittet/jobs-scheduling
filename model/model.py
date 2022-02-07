import math

import ortools.sat.python.cp_model
from ortools.sat.python import cp_model
from itertools import combinations

from utils.utils import shifts_overlap


class Model:
    status_str_optimal = 'optimal'
    status_str_feasible = 'feasible'
    status_str_infeasible = 'infeasible'
    status_str_invalid = 'invalid'

    def __init__(self, min_nb_shifts, max_nb_shifts, alpha=1, beta=1, gamma=1):
        # Model use to solve the optimization
        self.model = cp_model.CpModel()

        # Assignments shift-person variables
        self.shifts_dict = {}

        # Shifts variables/constraints
        self.shifts_vars = {}

        # Minimum number of shifts by person
        self.min_nb_shifts = min_nb_shifts
        self.max_nb_shifts = max_nb_shifts

        # Coefficients for the objective function
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Objective variable
        self.objective_var = None

    def create_assignment_person_shift_variable(self, shift, person):
        """
        Create a binary assignment person-shift variable.
        :param shift: shift to be assigned
        :param person: person to be assigned
        :return: -
        """
        self.shifts_dict[(person, shift)] = self.model.NewBoolVar(
            name=f'shift {shift.id} ({shift.start}, {shift.end}) : {person}')

    def create_interval_shift_variable(self, shift, person):
        """
        Create an interval variable for each pair person-shift,
        active only when the assignment person-shift is active.
        This is used to avoid overlaps in assignments shifts for a person
        :param shift: shift to be assigned
        :param person: person to be assigned
        :return: -
        """
        # print(self.shifts_dict[(person, shift)])
        self.shifts_vars[(person, shift)] = self.model.NewOptionalIntervalVar(start=shift.start,
                                                                              size=shift.end - shift.start,
                                                                              end=shift.end,
                                                                              is_present=self.shifts_dict[
                                                                                  (person, shift)],
                                                                              name=f'shift {shift.id} ({shift.start}, {shift.end})')

    def create_variables(self, shift, person):
        """
        Create all the variables of the model for this shift and this person
        :param shift: shift to be assigned
        :param person: person to be assigned
        :return: -
        """
        # Assignment variable
        self.create_assignment_person_shift_variable(shift, person)

        # Interval variables/constraints to avoid overlaps
        self.create_interval_shift_variable(shift, person)

    def create_constraints(self, shift, person, availability):
        """
        Create all the constraints of the model for this shift and this person
        :param shift: shift to be assigned
        :param person: person to be assigned
        :param availability: availability of this person for this shift
        :return: -
        """
        # Flag to keep track whether this person was already forbidden
        # to be assigned to this shift
        is_forbidden = False

        # If the person is not available, add a constraint
        if not availability:
            # Add constraint as not available
            self.model.AddForbiddenAssignments([self.shifts_dict[(person, shift)]], [(True,)])
            is_forbidden = True

        # If the shift requires only major persons and the person is
        # minor, add a constraint
        if shift.is_major_only and (not person.is_major()) and (not is_forbidden):
            self.model.AddForbiddenAssignments([self.shifts_dict[(person, shift)]], [(True,)])

    def create_no_overlap_constraints(self):
        """
        Create the non overlap constraints for the
        assignments of persons to shifts.
        :return: -
        """
        intervals_to_constrain = {}

        for (person, shift), interval_var in self.shifts_vars.items():
            if person in intervals_to_constrain.keys():
                intervals_to_constrain[person].append(interval_var)
            else:
                intervals_to_constrain[person] = [interval_var]

        for intervals_per_person in intervals_to_constrain.values():
            # Must add no overlap constraints per person and not
            # all intervals at once regardless of person !
            self.model.AddNoOverlap(intervals_per_person)

    def create_general_constraints(self, shifts, persons):
        """
        Create general constraints on certain values
        :param shifts: shifts to be assigned
        :param persons: persons to be assigned
        :return: -
        """
        # Constraint for the number of person for each shift
        for s in shifts:
            # Ensure that shift s has the right number of persons assigned to it
            self.model.Add(sum(self.shifts_dict[(p, s)] for p in persons) == s.nb_persons)

        # Constraint for the minimum number of person by shift
        for p in persons:
            # Add the minimum and maximum number of shifts constraint for this person
            nb_shifts = sum(self.shifts_dict[(p, s)] for s in shifts)
            self.model.Add(nb_shifts >= self.min_nb_shifts)
            self.model.Add(nb_shifts <= self.max_nb_shifts)

        # Add the no overlap constraints on the shifts by person
        self.create_no_overlap_constraints()

    def compute_nb_shifts_per_person_objective(self, shifts, persons):
        """
        Compute the min and max number of shifts assigned to a single
        person.
        :param shifts: shifts to be assigned
        :param persons: persons to be assigned
        :return: difference between max and min number of shifts
        assigned to a single person
        """
        # Compute the number of shifts assigned to each person
        sums_vars = []
        for p in persons:
            tmp = self.model.NewIntVar(0, len(shifts), f'person_{p}_nb_shifts')
            self.model.Add(tmp == sum(self.shifts_dict[(p, s)] for s in shifts))

            sums_vars.append(tmp)

        # Set the variable to be the minimum number of shifts assigned to
        # a single person
        min_var_objective = self.model.NewIntVar(0, len(shifts), 'min_nb_shifts_objective')
        self.model.AddMinEquality(min_var_objective, sums_vars)

        # Set the variable to be the minimum number of shifts assigned to
        # a single person
        max_var_objective = self.model.NewIntVar(0, len(shifts), 'max_nb_shifts_objective')
        self.model.AddMaxEquality(max_var_objective, sums_vars)

        # Compute the difference between the 2 extremes
        return max_var_objective - min_var_objective

    def compute_has_manager_objective(self, shifts):
        """
        Compute the objective that encourage having
        at least one person that can be in charge in
        each shift.
        :return: objective for this soft constraint
        """
        """
        # Init all the shifts count to 1, otherwise
        # the problem is not feasible if a shift has
        # no person that can be in charge
        has_manager = {s.id: [self.model.NewConstant(1)] for s in shifts}

        # Compute the number of person that can
        # be in charge for each shift
        for (p, s), var in self.shifts_dict.items():
            if p.can_charge:
                # Use negation to use product afterwards,
                # has to define new variable for each negation
                # for multiplication below to work
                tmp = self.model.NewBoolVar(f'person_{p}_not_can_charge')
                self.model.Add(tmp == var.Not())
                has_manager[s.id].append(tmp)

        products = []
        for s_id, bools in has_manager.items():
            tmp = self.model.NewBoolVar(f'shift_id_{s_id}_has_manager_objective')

            # Product will be 1 if there is no person assigned to this shift
            # that can be in charge and 0 if there is at least one
            self.model.AddMultiplicationEquality(tmp, bools)

            products.append(tmp)

        # Want to minimize the number of shifts that has no person that
        # can be in charge
        objective_var = self.model.NewIntVar(0, len(shifts), 'has_manager_objective')
        self.model.Add(objective_var == sum(products))
        return objective_var
        """
        #""" Above solution better represens the objective but
        # currently yields an invalid model
        # Compute the number of person that can be in charge for
        # each shift
        has_manager = {s.id: self.model.NewConstant(0) for s in shifts}
        for (p, s), var in self.shifts_dict.items():
            if p.can_charge:
                has_manager[s.id] += var

        # Create new variables for each sum
        max_number_persons = max(s.nb_persons for s in shifts)
        sums = []
        for s_id, bools in has_manager.items():
            tmp = self.model.NewIntVar(0, max_number_persons, f'shift_{s_id}')
            self.model.Add(tmp == bools)

            sums.append(tmp)

        # Compute the minimum number of person in charge in a shift
        objective_var = self.model.NewIntVar(0, max_number_persons, 'has_manager_objective')
        self.model.AddMinEquality(objective_var, sums)

        return -objective_var
        #"""

    def compute_break_objective(self, shifts, persons):
        """
        Compute the objective that encourage having
        a break between 2 assignments for each person.
        :param shifts: shifts to be assigned
        :param persons : persons to be assigned
        :return: objective for this soft constraint
        """
        var_objective = 0
        for p in persons:
            # For each person, compute the total time
            # between her/his assigned tasks
            for s1, s2 in combinations(shifts, 2):
                # If the 2 shifts overlap, skip them
                do_overlap, s1_, s2_ = shifts_overlap(s1, s2)
                if not do_overlap:
                    # Add the break between the 2 shifts
                    # if this person was assigned to these 2 shifts
                    tmp_var = self.model.NewIntVar(0, 1, f'person_{p}_shifts_pair_{s1_}_{s2_}_break_objective')
                    self.model.AddMultiplicationEquality(tmp_var, [self.shifts_dict[(p, s1_)], self.shifts_dict[(p, s2_)]])
                    var_objective += tmp_var * (s2_.start - s1_.end)

        # In negative as want to minimize the final objective
        return - var_objective

    def create_objective(self, shifts, persons):
        """
        Create and set the constraints for the objective function.
        :param shifts: shifts to be assigned
        :param persons : persons to be assigned
        :return: -
        """
        self.model.Minimize(
            self.alpha * self.compute_nb_shifts_per_person_objective(shifts, persons) +
            self.beta * self.compute_has_manager_objective(shifts) +
            self.gamma * self.compute_break_objective(shifts, persons)
        )

    def build_model(self, persons, shifts, availability):
        """
        Create all the binary variables and add them to the
        model.

        :param persons: list of persons
        :param shifts: list of shifts
        :param availability: numpy array of shape (nb persons, nb shifts)
        containing True if the person is available at a given shift, False if not
        :return: -
        """
        # Add all variables and constraints for pairs of person-shift
        for i, p in enumerate(persons):
            for j, s in enumerate(shifts):
                # Create all the variables
                self.create_variables(s, p)

                # Create all the hard constraints for each pair
                # of shift and person
                self.create_constraints(s, p, availability[i, j])

        # Add general hard constraints
        self.create_general_constraints(shifts, persons)

        # Add the soft constraints as the objective
        self.create_objective(shifts, persons)

    def solve(self):
        """
        Solve the problem specified by this model
        :return: dict of the form: (shift: list of person assigned),
            string status, wall time
        """
        solver = cp_model.CpSolver()
        status = solver.Solve(self.model)
        status_str = self.status_str_invalid

        if status == cp_model.OPTIMAL:
            status_str = self.status_str_optimal
        elif status == cp_model.FEASIBLE:
            status_str = self.status_str_feasible
        elif status == cp_model.INFEASIBLE:
            status_str = self.status_str_infeasible

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            assignments = {}
            for (p, s), var in self.shifts_dict.items():
                if solver.Value(var):
                    if s in assignments.keys():
                        assignments[s] = assignments[s] + [p]
                    else:
                        assignments[s] = [p]

            return assignments, status_str, solver.WallTime()

        else:
            return None, status_str, solver.WallTime()

    def solve_test(self):
        """
        Solve the problem specified by this model
        :return: -
        """
        solver = cp_model.CpSolver()
        status = solver.Solve(self.model)
        status_str = 'invalid'

        if status == cp_model.OPTIMAL:
            status_str = 'Optimal'
        elif status == cp_model.FEASIBLE:
            status_str = 'Feasible'
        elif status == cp_model.INFEASIBLE:
            status_str = 'Infeasible'
        print(f'Status of the solution : {status_str}')

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print('Solution found')
            assignments = {}
            for (p, s), var in self.shifts_dict.items():
                if solver.Value(var):
                    if s in assignments.keys():
                        assignments[s] = assignments[s] + [p]
                    else:
                        assignments[s] = [p]

            for s, l in assignments.items():
                print(f'{s} : ')
                for p in l:
                    print(f'\t{p}')
        else:
            print('No solution found')
