# T5 Translation API

A FastAPI-based service that provides translation using Google's T5 model. This project leverages the power of the T5 (Text-to-Text Transfer Transformer) model for high-quality machine translation while maintaining efficient performance through batched predictions.

## Features

- 🤖 T5-small model for English to German translation
- 🚀 FastAPI for high-performance async API endpoints
- 🔄 Batch translation support for multiple sentences
- 🎯 Token length validation and input preprocessing
- 🐳 Multi-stage Docker builds for optimized container size
- ✨ Automatic model caching
- 🧪 Comprehensive test coverage
- 📊 Translation quality metrics
- ⚡ UV package manager for faster dependency management

## Technical Stack

- **Machine Learning**: 
  - HuggingFace Transformers
  - T5-small model (optimized for translation tasks)
  - PyTorch for model inference
  
- **Backend**:
  - FastAPI framework
  - Pydantic for data validation
  - Uvicorn ASGI server
  
- **Development Tools**:
  - UV for dependency management
  - Pytest for testing
  - Docker for containerization
  ## Getting Started

This project provides a FastAPI-based translation API.  Here are the steps to get it running:

### Prerequisites

*   **Docker:**  If you are using the provided development container (recommended), Docker and VS Code are pre-configured.  Otherwise, ensure Docker is installed on your system.
*   **VS Code:**  For the best development experience, use VS Code with the Remote - Containers extension.
*   **uv:** We are using uv package manager for installation. Please make sure that you have it installed

### Running the Application

**1. Clone the Repository:**

```bash
git clone https://github.com/Tyiooo/sm-internship.git
cd sm-internship
```

**2. Development Container (Recommended):**

*   If you're using VS Code with the Remote - Containers extension, VS Code should automatically prompt you to open the project in the container.  Click "Reopen in Container" to build and connect to the development environment. All dependencies will be pre-installed

**3. Local Development (Without Container):**


    ```bash
    # Create a virtual environment (recommended)
    python3 -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    # or
    .venv\Scripts\activate  # On Windows

    # Install project dependencies using uv
    uv pip install -r requirements.txt
    ```

**4. Run the Application:**

```bash
uv run main.py
```

This will start the FastAPI application.  You can then access the API endpoints (e.g., the health check at `/ping`) in your browser or using a tool like `curl` or Postman.
