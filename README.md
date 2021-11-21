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
- timetable
- number of people needed
- need for major people only

To model the person-task assignment, we create the binary variables representing
all the possible pair (person, task). A schedule is an assignment (0 if the 
person was not assigned to the job, 1 if yes) to all these binary variables.


### Constraints

We have the following constraints on the schedule :
- A person cannot be assigned to 2 tasks that overlap
- A person cannot be assigned to a task where she/he is not available
- A person should not do 2 tasks in a row
- A person should have at least min_nb_tasks assigned
- A person should have no more than max_nb_tasks assigned
- A task requiring only major persons cannot have minor persons assigned to itself
- A task should always have at least 1 person that can be in charge

