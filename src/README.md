Directory project structure
--------------------

    .
    ├--- AUTHORS.md
    ├--- README.md
    ├── models  <- compiled model .pkl or HDFS or .pb format
    ├── common  <- base superclass files
    ├── utils  <- utility functions
    ├── env <- Flatland environment files
    │   ├--- parameters.yml <- predefined env parameters configuartions
    ├── [approach] <- specific class implementations for  resolutive approach
    │   ├── hyperparameters <- WandB params tuning
    │   ├--- main.py <- starting point for specific [approach]
    ├── notebooks <- jupyter notebooks for exploratory analysis and explanation 
    |--- environment.yml <- file with libraries and library versions for recreating the analysis environment
   
Expanded documentation structure
--------------------

    ..
    ├── modules  <- inspirations and starter kit project
    ├── docs  <- usage documentation or reference papers
    ├── reports <- generated project artefacts eg. visualisations or tables
    │   └── figures
    └── src .

