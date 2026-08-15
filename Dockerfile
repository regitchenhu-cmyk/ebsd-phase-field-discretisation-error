FROM dolfinx/dolfinx:v0.11.0@sha256:2ae4bfbc0d9077268880faf04c72750528bee986c94ab223a2c159969bd56fa8

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY src ./src
COPY examples ./examples
RUN python3 -m pip install --no-deps -e .

CMD ["python3", "-m", "graphfracture", "examples/sent_graph_hydrogen_onset.toml"]
