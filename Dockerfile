# Nvidia image for GPU support
FROM ubuntu:22.04

# Set environment variables
ENV WORKING_PORT=8080
ENV DEBUG=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Set environment variables
ENV PATH=/usr/local/python3.12/bin:$PATH
ENV NODE_ID=exo-node-1

# Set pipefail
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install dependencies and setup python3.12
RUN apt-get update && \
    apt-get install --no-install-recommends -y git build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install --no-install-recommends -y python3.12 python3.12-dev curl && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    rm -rf /var/lib/apt/lists/*

# Configure python3.12
RUN pip3 install --no-cache-dir --upgrade requests && ln -fs /usr/bin/python3.12 /usr/bin/python

# Copy installation files
COPY setup.py .

# Install exo
RUN pip3 install --no-cache-dir . && \
    pip3 install --no-cache-dir tensorflow && \
    pip3 cache purge

# Copy source code
# TODO: Change this to copy only the necessary files
COPY . .

# Run command
ENTRYPOINT ["/usr/bin/python"]
CMD ["main.py", "--disable-tui", "--node-id", "$NODE_ID"]

# Expose port
EXPOSE $WORKING_PORT
