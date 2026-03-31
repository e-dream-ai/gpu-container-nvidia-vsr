variable "DOCKERHUB_REPO" {
  default = "edreamai"
}

variable "DOCKERHUB_IMG" {
  default = "gpu-container-nvidia-vsr"
}

variable "RELEASE_VERSION" {
  default = "latest"
}

group "default" {
  targets = ["nvidia-vsr"]
}

target "base" {
  context = "."
  dockerfile = "Dockerfile"
  platforms = ["linux/amd64"]
  tags = ["${DOCKERHUB_REPO}/${DOCKERHUB_IMG}:${RELEASE_VERSION}"]
}

target "nvidia-vsr" {
  context = "."
  dockerfile = "Dockerfile"
  tags = ["${DOCKERHUB_REPO}/${DOCKERHUB_IMG}:${RELEASE_VERSION}"]
  inherits = ["base"]
}
