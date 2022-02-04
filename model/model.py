import math
from ortools.sat.python import cp_model
from itertools import combinations


class Model:
    def __init__(self, min_nb_shifts, max_nb_shifts, min_break=1, alpha=1, beta=1, gamma=1):
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

        # Minimum time for the break
        self.min_break = min_break

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
            # Add the minimum number of shifts constraint for this person
            self.model.Add(sum(self.shifts_dict[(p, s)] for s in shifts) >= self.min_nb_shifts)

        # Add the no overlap constraints on the shifts by person
        self.create_no_overlap_constraints()

    def create_objective(self, shifts, persons):
        """
        Create and set the constraints for the objective function.
        :param shifts: shifts to be assigned
        :param persons : persons to be assigned
        :return: -
        """
        # Create the objective variable
        self.objective_var = self.model.NewIntVar(-math.inf, math.inf, 'objective')

        # Compute the objective
        def compute_nb_assigned_shifts(shifts, shifts_dict, p):
            nb_assigned = sum(shifts_dict[(p, s)] for s in shifts)
            return nb_assigned if nb_assigned > self.max_nb_shifts else 0

        # Compute the number of shifts that have a manager assigned to it
        def compute_has_manager(shifts_dict):
            has_manager = {}
            for (p, s), var in shifts_dict.items():
                if var and p.can_charge:
                    has_manager[s.id] = True

            return sum(has_manager.values())

        # Compute the least amount of break this person p has
        # in its assignment
        def compute_break_objective(shifts, shifts_dict, p):
            # Collect shifts assigned to this person p
            shifts_p = [(shifts_dict[(p, s)].start, shifts_dict[(p, s)].end) for s in shifts if shifts_dict[(p, s)]]

            # If this person has less than 2 shift, return 0
            if len(shifts_p) < 2:
                return 0

            min_break = math.inf
            for s1, s2 in combinations(shifts_p, 2):
                cur_break_1 = s2[0] - s1[1]
                cur_break_2 = s1[0] - s2[1]
                if 0 <= cur_break_1 < min_break:
                    min_break = cur_break_1
                if 0 <= cur_break_2 < min_break:
                    min_break = cur_break_2

            # Return negative min break as we minimize the objective function
            return -min_break if min_break < self.min_break else 0

        self.model.Add(
            self.alpha * sum(compute_nb_assigned_shifts(shifts, self.shifts_dict, p) for p in persons) +
            self.beta * compute_has_manager(self.shifts_dict) +
            self.gamma * sum(compute_break_objective(shifts, self.shifts_dict, p) for p in persons)
            == self.objective_var
        )

        self.model.Minimize(self.objective_var)

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

                # Create all the constraints
                self.create_constraints(s, p, availability[i, j])

        # Add general constraints
        self.create_general_constraints(shifts, persons)

        # Add the soft constraints as the objectives
        # self.create_objective(shifts, persons)

    def solve(self):
        """
        Solve the problem specified by this model
        :return: -
        """
        solver = cp_model.CpSolver()
        status = solver.Solve(self.model)

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
