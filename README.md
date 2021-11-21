# Jobs scheduling

## Model
### Variables

The scheduling involves assigning people to a set of given tasks.

A person is represented by :
- its name
- its age
- whether the person is
suited for being in charge of the other persons for a given task.
- its availability for different hours

A job/task is represented by :
- timetable. For ease of modelling, the start time and end time of the shift
  is taken in hours from a given reference point. This avoids confusion when
  shifts are in-between days (e.g. 23h00-02h00).
- number of people needed
- need for major people only

To model the person-task assignment, we create the binary variables representing
all the possible pair (person, task). A schedule is an assignment (0 if the 
person was not assigned to the job, 1 if yes) to all these binary variables.


### Constraints

We have the following constraints on the schedule :
- Hard constraints :
  - A person cannot be assigned to 2 tasks that overlap
  - A person cannot be assigned to a task where she/he is not available
  - A person must have at least min_nb_tasks assigned
  - A task requiring only major persons cannot have minor persons assigned to itself
  - A task must have the required number of persons
- Soft constraints :
  - A person should not do 2 tasks in a row (have at least 1h of break between them)
  - A person should have no more than max_nb_tasks assigned
  - A task should always have at least 1 person that can be in charge

The soft constraints are expressed in the objective function to minimize. We define this objective function as follows :
$$
R = \alpha \sum_{p \in persons} \{\text{nb assigned shifts to p}\} * I(\{\text{nb assigned shifts to p}\} > max_nb_tasks)
+ \beta \sum_{s \in shifts} I(\text{shift s has a manager assigned to itself})
+ \gamma \sum_{p \in persons} I(p \text{ has at least 1h of break between each of its shifts})
$$
where $\alpha$ and $\beta$ are parameters that can be tuned to give more or less importance to each term.