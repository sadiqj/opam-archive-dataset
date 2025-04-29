FROM ocaml/opam:archive

RUN apk --no-cache add curl python3
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app

# Create and initialize Python virtual environment
RUN uv venv venv
# Make sure we always use the venv going forward
ENV VIRTUAL_ENV="/app/venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN uv pip install semver pandas pyarrow huggingface_hub datasets tqdm

# Copy and set up our package finder script
COPY process_packages.py /usr/local/bin/process_packages.py
RUN chmod +x /usr/local/bin/process_packages.py

# Run the script when the container starts
RUN python3 /usr/local/bin/process_packages.py
