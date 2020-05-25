# Classification toolkit

 * Decouple training from project configuration.
 * Uses `sacred` (https://github.com/IDSIA/sacred) for logging and bookkeeping.

## Installation

`python setup.py install`

## Usage

 1. Make sure that the mongodb for sacred is running (refer to
    https://github.com/vivekratnavel/omniboard/blob/master/docs/quick-start.md#docker-compose)
 2. Create your own konfiguration.py file from template `pd-init.py`
 3. Edit file `konfiguration.py`
 4. Launch training `pd-train.py`
