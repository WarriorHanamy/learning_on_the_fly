# Use the already-built lotf image as base
FROM lotf:latest

# Set working directory
WORKDIR /app

# Set PATH to include uv
ENV PATH="/root/.local/bin:$PATH"

# Default command
CMD ["uv", "run", "python", "-m", "lotf", "--help"]
