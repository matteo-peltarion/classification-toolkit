# Classification toolkit

 * Decouple training from project configuration.
 * Uses `sacred` (https://github.com/IDSIA/sacred) for logging and bookkeeping.

## Installation

Due to some unresolved issues regarding paths to resources, this is the way to
install the package (will be fixed properly in the future™).

`pip install -e .`

Also install additional pip packages with `pip install -r requirements.txt`.

## Usage

 1. Make sure that the mongodb for sacred is running (refer to
    https://github.com/vivekratnavel/omniboard/blob/master/docs/quick-start.md#docker-compose).
 2. Create your own konfiguration.py file from template using command
    `pd-init.py`. Using option `--task TASK` you can start from different
    templates (currently `classification` for multiclass classification and
    `mlc` for multilabel classification).
 3. Edit generated file `konfiguration.py`, redefining the required functions
    and variables.
 4. Start an omniboard instance on your machine, using command
    `docker-compose up -d` while in the `extra/omniboard` folder.
 5. Launch training `pd-train.py`, possibly overriding default parameters using
    the Sacred syntax: e.g. `pd-train.py with lr=1e-5 batch_size=32`
