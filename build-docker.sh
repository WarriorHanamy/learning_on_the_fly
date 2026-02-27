#!/usr/bin/env bash
#
# ==============================================================================
# Docker Build Script with BuildKit and Cache Mount
# ==============================================================================
# Why: Use BuildKit for better build performance and bind mount host cache to
#      avoid re-downloading packages on every build.
# ==============================================================================

# ==============================================================================
# Script metadata
# ==============================================================================
readonly SCRIPT_VERSION="1.0.0"
readonly SCRIPT_AUTHOR="LOTF Project"
readonly SCRIPT_DESCRIPTION="Build Docker image with BuildKit and host cache bind mount"

readonly SCRIPT_NAME="${BASH_SOURCE[0]##*/}"
SCRIPT_DIR="$(cd "${BASH_SOURCE[0]%/*}" && pwd)"
readonly SCRIPT_DIR

# ==============================================================================
# Configuration
# ==============================================================================
readonly IMAGE_NAME="lotf"
readonly IMAGE_TAG="${IMAGE_TAG:-latest}"
readonly HOST_CACHE_DIR="/home/rec/.cache/uv"
readonly CONTAINER_CACHE_DIR="/root/.cache/uv"

# ==============================================================================
# Strict mode
# ==============================================================================
set -Eeuo pipefail
set -o errtrace

# ==============================================================================
# Logging
# ==============================================================================
log_info() {
    printf '\033[32m%s\033[0m\n' "[INFO] $*" >&2
}

log_error() {
    printf '\033[31m%s\033[0m\n' "[ERROR] $*" >&2
}

log_warn() {
    printf '\033[33m%s\033[0m\n' "[WARN] $*" >&2
}

die() {
    log_error "$1"
    exit "${2:-1}"
}

# ==============================================================================
# Cleanup & traps
# ==============================================================================
cleanup() {
    local exit_status=$?
    set +e
    return "${exit_status}"
}

fct_on_error() {
    local exit_status=$?
    local line_no="${1:-?}"
    local command="${2:-?}"
    trap - ERR
    log_error "Command failed (exit ${exit_status}) at line ${line_no}: ${command}"
    exit "${exit_status}"
}

fct_setup_traps() {
    trap 'cleanup' EXIT
    trap 'fct_on_error "${LINENO}" "${BASH_COMMAND}"' ERR
}

# ==============================================================================
# Dependency checking
# ==============================================================================
fct_check_dependencies() {
    if ! command -v docker >/dev/null 2>&1; then
        die "Docker is not installed or not in PATH"
    fi
}

# ==============================================================================
# Main logic
# ==============================================================================
fct_execute_this() {
    log_info "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
    log_info "Using DOCKER_BUILDKIT=1 for optimal build performance"
    log_info "Mounting host cache from: ${HOST_CACHE_DIR}"

    # Check if host cache directory exists
    if [[ ! -d "${HOST_CACHE_DIR}" ]]; then
        log_warn "Host cache directory does not exist: ${HOST_CACHE_DIR}"
        log_warn "Build will continue, but caching may not be optimal"
    fi

    # Build with BuildKit and cache mount
    # Why: DOCKER_BUILDKIT=1 enables BuildKit for better caching and performance
    # Why: --mount type=bind mounts host cache to avoid re-downloading packages
    DOCKER_BUILDKIT=1 docker build \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --tag "${IMAGE_NAME}:${IMAGE_TAG}" \
        --file "${SCRIPT_DIR}/Dockerfile" \
        --mount "type=bind,source=${HOST_CACHE_DIR},target=${CONTAINER_CACHE_DIR}" \
        "${SCRIPT_DIR}"

    log_info "Build completed successfully: ${IMAGE_NAME}:${IMAGE_TAG}"
}

main() {
    fct_setup_traps
    fct_check_dependencies

    fct_execute_this
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
