# SimpleEnv
Simple OpenAI gym compatible environments to help develop DRL agents.
Standard gym environments don't provide enough feedback for algorithm development and debugging.
Goal here is to have environments that help develop bug free agents and algorithms.
At current state the project is very basic. Open to suggestions. Feel free to contribute. 

### What does it have?
Simple gym environment, with continuous state and action space.
State has 4 dimensions.
Action has 2 dimension.
Each episode will run for 1024 time steps.

#### all_ones
Can the agent learn to output(action) all ones irrespective of the input(state).  

**State :**
State is a numpy array with random values between 0 and 1. 
At every step new values are set randomly.

**Reward :**
Reward is the negative l1 loss between the action and an array of 1s.
#### copy_cat
Can the agent output(action) all ones or all zeros depending on the input(state) 

**State :**
State is either an array of all ones or all zeros, chosen randomly at each time step.

**Reward :**
Reward is the negative MSE between the action and an array of 1s.


### How to setup ?
* Clone the repo
* Go to `SimpleEnv` directory
* run `pip install -e .`
\
This will install `simple-env`

### How to use?
```
import gym
import simple_env

env_all_ones = gym.make('all_ones-v0')
env_copy_cat = gym.make('copy_cat-v0')
``` 
The environments are good for use now. 


### What's next?
* Making it available via pypi
* Adding more simple environments for different types of agents
* Maybe, add testcases for some standard functions and operations
\
(Project is not in active development right now)
