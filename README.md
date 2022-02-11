# Solution to the Flatland Challenge
2020 Edition - https://www.aicrowd.com/challenges/flatland-challenge

[[https://s3.eu-central-1.amazonaws.com/aicrowd-static/SBB/images/Flatland_Logo.svg|alt=octocat]]

# Usage
### Train
```bash
python src/dddqn/main.py

python src/a2c/main.py

python src/dqn/main.py
```
Python3.6/3.7 is suggested for compatibility with Flatland environment library

### Parameters
Specific parameters about model network can be found in the relativesection of the yml file:
`src/env/training_parameters.yml`
Additional parameters regarding the environment can be setted in:
`src/env/env_parameters.yml`

# Docs
## Observations
 The major improvements to the starter kit approach was the observer implemented in this case as reduced DAG graph created on the fly at each inizialization of the environment in order to ease the process of map traversing and searching. The implementation can be found in the  file `dag_observer.py`

## Networks
As model networks the project include three differents approaches: A2C, DQN and D3QN. All these have a dedicated folder with the implementation. 

## Multi agents interactions
The custom observer allowed also a more specific management of random malfunction events and collision avoidance practice

For more detailed information on the approaches see:

[Documentation with approach explained](reports/main.pdf)

[Presentation of the project](docs/Flatland_project_discussion.pdf)
