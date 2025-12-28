# ML-Library

### Hardware-Agnostic Machine Learning Powered by Pure Python

This library is being developed as an open-source alternative to the proprietary ecosystems of major manufacturers like NVIDIA and AMD (CUDA/ROCm). Our mission is to leverage the power of **PyOpenCL** to provide a flexible, modular machine learning infrastructure that delivers high performance regardless of your hardware brand.

---

## Why This Library?

Today’s machine learning landscape is often restricted to specific hardware ecosystems. This project aims to break those boundaries:

* **Hardware Freedom:** Achieve full performance on any GPU (NVIDIA, AMD, Intel) via PyOpenCL integration.
* **The Power of Pure Python:** The core architecture is built entirely in Python, making it transparent, readable, and highly extensible.
* **Future Vision:** While currently in an experimental stage (undergoing optimization), our goal is to achieve maximum speed and a simplified API that even end-users can deploy with ease.

---

## Key Features

The library includes all the essential building blocks for designing and training artificial neural networks from the ground up:

### Core Architecture

* **Flexible Model Template:** A high-level `Model` class that allows for any custom network design.
* **Dynamic Layer Management:** Automated weight and bias calculations based on variable layer counts.
* **Advanced Derivative Engine:** A mathematical foundation containing derivatives for all functions, optimized for backpropagation.

### Functional Richness

* **Parameter Initialization:** Various strategic functions for initializing network parameters.
* **Comprehensive Activations:** A full suite of industry-standard activation functions.
* **Versatile Loss Functions:** Optimized error calculation methods tailored for diverse network designs.

### Developer Tools

* **Helper Classes:** Manage activations, network designs, and optimizers via string inputs—eliminating the need to memorize complex parameters.
* **Tools Class:** A dedicated utility class for common transformations and helper functions used in machine learning workflows.

---

## Roadmap
**Phase 1 (Current):** Establishing the core mathematical foundation and classification structure in pure Python.
**Phase 2:** Implementing GPU acceleration via PyOpenCL integration.
**Phase 3:** Reducing complexity, simplifying logic, and executing performance optimizations.
**Phase 4:** Releasing a UI based version for end-users.
