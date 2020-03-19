FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

# Install python packages
ADD requirements.txt .
RUN pip install -r requirements.txt

# Install sidekick
RUN pip install git+https://github.com/Peltarion/sidekick#egg=sidekick
