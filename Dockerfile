FROM porepy-petsc:latest

RUN git -C ${HOME}/porepy pull

RUN git clone https://github.com/Yuriyzabegaev/FTHM-Solver.git && \
    pip install --no-cache-dir -r FTHM-Solver/requirements.txt

ENV PYTHONPATH=${PYTHONPATH}:${HOME}/FTHM-Solver
ENV PYTHONPATH=${PYTHONPATH}:${HOME}/FTHM-Solver/experiments

WORKDIR ${HOME}/FTHM-Solver

ENTRYPOINT [ "/bin/bash" ]