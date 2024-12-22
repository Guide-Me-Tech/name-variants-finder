# Name variants generator for Uzbek Names and Russian names

Using Milvus DB as vector database and sentence-transformers for embeddings

# Usage

```
fastapi run
```

requires Milvus DB to be installed before hand.

link to the swagger: http://127.0.0.1:8000/docs



.
├── .git/                                      # Git metadata (branches, commits, hooks, etc.)
│   ├── FETCH_HEAD
│   ├── HEAD
│   ├── config
│   ├── description
│   ├── hooks/                                 # Sample Git hook scripts (pre-commit, pre-push, etc.)
│   ├── index
│   ├── info/
│   ├── logs/
│   ├── objects/
│   ├── packed-refs
│   └── refs/
├── .github/
│   └── workflows/
│       └── pylint.yml                         # GitHub Actions workflow for running PyLint on pushes/PRs
├── .gitignore                                 # Lists files & patterns for Git to ignore (e.g., /venv, *.pyc)
├── README.md                                  # Main project documentation & instructions
├── data_names/                                # Directory holding raw text files of names (Uzbek, Russian, etc.)
│   ├── russian_female_names.txt               # Russian female names, one per line
│   ├── russian_male_names.txt                 # Russian male names
│   ├── russian_names_set_merged.txt           # Combined Russian names dataset
│   ├── uzbek_names.txt                        # Uzbek names dataset
│   ├── uzbek_names_set_merged.txt             # Merged Uzbek names
│   ├── uzname-men-name-lat-v1.1b 2.txt        # Another Uzbek male name list (Latin script)
│   └── uzname-women-name-lat-v1.1b.txt        # Another Uzbek female name list (Latin script)
├── deploy.sh                                  # Deployment script (could build/run containers or other tasks)
├── docker-compose.yml                         # Multi-container Docker configuration (e.g., Milvus + FastAPI)
├── dockerfile.server                          # Dockerfile for building the Python/FastAPI server image
├── embedEtcd.yaml                             # Configuration for embedded Etcd (often used with Milvus standalone)
├── embed_milvus.py                            # Script that reads names, generates embeddings, & inserts into Milvus
├── entrypoint.sh                              # Shell script that might be the Docker container entrypoint (runs server)
├── etc/                                       # Miscellaneous or supplemental scripts
│   ├── embed_names.py                         # Additional script for embedding names
│   ├── embed_names_redis.py                  # Script for embedding names & possibly storing/retrieving in Redis
│   ├── get_names.py                           # Fetches or processes name data (could be from files or elsewhere)
│   ├── main.py                                # Another entry point or test script
│   ├── search.py                              # Simple search logic script (possibly local or partial testing)
│   └── search_milvus.py                       # Searches Milvus for name matches
├── requirements.txt                           # Python dependencies for the project (install with pip)
├── server_search_names.py                     # FastAPI server code (routes for searching names via Milvus)
├── standalone_embed.sh                        # Shell script to run Milvus in standalone mode (with embedded Etcd)
├── test/
│   └── test.ipynb                             # Jupyter notebook for testing or prototyping code behavior
├── user.yaml                                  # Overrides or custom settings for Milvus or other services
└── utils/                                     # Utility scripts & helper modules
    ├── convert_between_latin_and_cyril.py     # Name transliteration logic for Uzbek/Cyrillic ↔ Latin
    ├── docker_checker.py                      # Checks Docker availability/running containers
    ├── download_embedding_model.py            # Downloads a required embedding model if not present
    ├── download_nltk.py                       # Installs or configures NLTK data
    ├── embedddigs.py                          # Core logic/functions for embedding & searching with Milvus
    ├── initializers.py                        # Script that may set global defaults or environment variables
    ├── install.py                             # Possibly installs dependencies or sets up the environment
    ├── load_model.py                          # Loads ML or NLP models into memory
    ├── loop.py                                # Contains a custom loop or concurrency logic
    ├── printing.py                            # Color-coded print functions (e.g., printgreen, printred)
    ├── shell_scripts/
    │   ├── deploy.classifier_server.sh        # Shell script for deploying a "classifier server" (specific use case)
    │   ├── deploy.sh                          # Another deployment script
    │   ├── development.deploy.classifier_server.sh  # Dev version of classifier server deploy
    │   ├── development.deploy.sh             # Dev version of general deploy script
    │   └── restart.sh                        # Script to restart services
    └── timer.py                               # Decorator or function to measure code execution time



# **Comprehensive README: An In-Depth Guide to the Project**

**Table of Contents**  
1. [Introduction and Purpose](#introduction-and-purpose)  
2. [Project Structure and File Overview](#project-structure-and-file-overview)  
   1. [.git/ (Git Repository Metadata)](#git-directory)  
   2. [.github/workflows/pylint.yml](#pylint-workflow)  
   3. [.gitignore](#gitignore-file)  
   4. [README.md](#readme-file)  
   5. [data_names/ (Name Data Files)](#data-names-directory)  
   6. [deploy.sh](#deploy-script)  
   7. [docker-compose.yml](#docker-compose-file)  
   8. [dockerfile.server](#dockerfile-server)  
   9. [embedEtcd.yaml](#embeddetcd-file)  
   10. [embed_milvus.py](#embed-milvus-py)  
   11. [entrypoint.sh](#entrypoint-sh)  
   12. [etc/ (Miscellaneous Scripts)](#etc-directory)  
   13. [requirements.txt](#requirements-txt)  
   14. [server_search_names.py](#server_search_names-py)  
   15. [standalone_embed.sh](#standalone-embed-sh)  
   16. [test/ (Testing Directory)](#test-directory)  
   17. [user.yaml](#user-yaml)  
   18. [utils/ (Utilities Directory)](#utils-directory)  
3. [How the Project Works: Detailed Explanations](#how-the-project-works-detailed-explanations)  
   1. [Data Ingestion and Preparation](#data-ingestion-and-preparation)  
   2. [Embedding Mechanisms](#embedding-mechanisms)  
   3. [Search and Retrieval Logic](#search-and-retrieval-logic)  
   4. [Dockerization and Deployment](#dockerization-and-deployment)  
4. [Visual Flow Diagrams](#visual-flow-diagrams)  
   1. [Visual Flow 1: Data Preparation Process](#visual-flow-1-data-preparation-process)  
   2. [Visual Flow 2: Running Milvus Locally with Embedded Etcd](#visual-flow-2-running-milvus-locally-with-embedded-etcd)  
   3. [Visual Flow 3: FastAPI Search Workflow](#visual-flow-3-fastapi-search-workflow)  
   4. [Visual Flow 4: Interaction with External Repositories or Services](#visual-flow-4-interaction-with-external-repositories-or-services)  
5. [Interactions with Other Repositories](#interactions-with-other-repositories)  
6. [Step-by-Step Usage Guide](#step-by-step-usage-guide)  

---

## **Introduction and Purpose**
This repository aims to provide a comprehensive system for embedding various kinds of names (particularly Uzbek and Russian) into vector representations for fast and efficient search. It leverages [Milvus](https://milvus.io/), an open-source vector database, as well as standard Docker-based deployments to simplify the environment setup. Additionally, it includes FastAPI endpoints that allow you to query these embedded names with various search parameters, returning the best matches from the Milvus database.

In the realm of natural language processing (NLP) and name-entity handling, it is increasingly crucial to have reliable tools for **quickly searching and comparing** textual items. This repository addresses that challenge with a multi-step solution:

1. **Collecting raw data** (various text files in `data_names/`).  
2. **Generating vector embeddings** using specialized embedding models (e.g., `all-MiniLM-L6-v2`).  
3. **Storing and indexing** these embeddings in Milvus for quick vector similarity searches.  
4. **Providing an API** (via FastAPI) to query these embeddings, enabling you to look up matches in real time.  
5. **Seamless Docker-based deployment** for consistent and reproducible environments.

By the end of reading this README, you should:

- Understand **how the project is structured** and what each directory/file is responsible for.  
- Gain insight into **how the embedding and retrieval logic** is implemented.  
- Learn how to **deploy and run** the project locally using Docker or a more manual approach.  
- Be able to trace data from its raw form in `data_names/` all the way to the API query results you obtain in `server_search_names.py`.  
- Appreciate how each script ties into the overall pipeline, including optional advanced settings in `embedEtcd.yaml`, `user.yaml`, or `docker-compose.yml`.

This documentation is extremely detailed—over multiple thousands of words—to ensure that a newcomer to the project can understand everything thoroughly.

---

## **Project Structure and File Overview**

Below is the top-level tree, followed by an explanation of each directory/file. The structure is laid out to keep the code for embeddings, Docker configurations, data files, testing, and utility scripts in clearly separated areas.

### **.git Directory** <a id="git-directory"></a>
```
.git
├── FETCH_HEAD
├── HEAD
├── config
├── description
├── hooks/
├── index
├── info/
├── logs/
├── objects/
├── packed-refs
└── refs/
```
- **Purpose**: Standard Git metadata folder containing all version control information (branches, commits, hooks, references, etc.).  
- **Relevance**: Not directly involved in the runtime logic. No modifications here are typically necessary unless you want to set up Git hooks (like `pre-commit` or `pre-push`).  

### **.github/workflows/pylint.yml** <a id="pylint-workflow"></a>
- **Purpose**: This is a GitHub Actions workflow that runs [PyLint](https://pylint.org/) on push events.  
- **Function**: Maintains code quality by linting Python files, ensuring the codebase adheres to style guidelines and best practices.  
- **Typical Usage**: Automatically triggered by GitHub whenever you push or open a pull request. If issues are found, the lint job will fail.

### **.gitignore** <a id="gitignore-file"></a>
- **Purpose**: Specifies intentionally untracked files to ignore (e.g., `__pycache__` directories, `.env` files with credentials, or large ephemeral data).  
- **Typical Content**: Usually includes patterns for ignoring compiled code, logs, local environment variables, and other files that should not be committed to the repository.

### **README.md** <a id="readme-file"></a>
- **Purpose**: The project’s main entry point for human-readable documentation (i.e., the file you are reading).  
- **Recommended**: To keep it updated with essential steps for installation, usage, and development instructions, plus architectural overviews.

### **data_names Directory** <a id="data-names-directory"></a>
```
data_names/
├── russian_female_names.txt
├── russian_male_names.txt
├── russian_names_set_merged.txt
├── uzbek_names.txt
├── uzbek_names_set_merged.txt
├── uzname-men-name-lat-v1.1b 2.txt
└── uzname-women-name-lat-v1.1b.txt
```
- **Purpose**: Holds various text files containing names. Each file typically includes one name per line.  
- **Usage**:  
  - `russian_female_names.txt` and `russian_male_names.txt` store male/female Russian names.  
  - `russian_names_set_merged.txt` is a combined list of Russian names for easier processing.  
  - `uzbek_names.txt`, `uzbek_names_set_merged.txt`, `uzname-men-name-lat-v1.1b 2.txt`, `uzname-women-name-lat-v1.1b.txt` store Uzbek names.  
- **Role in Pipeline**: These files are the starting raw data for any embedding or searching process. Scripts like `embed_milvus.py` or others in `etc/` read these to generate embeddings.

### **deploy.sh** <a id="deploy-script"></a>
- **Purpose**: A shell script meant to handle some form of deployment.  
- **Typical Usage**: Might be invoked in a CI/CD pipeline or locally to stand up or tear down certain containers or environments.  
- **What it Might Contain**: Docker commands (`docker build`, `docker run`) or even direct calls to other scripts (like `standalone_embed.sh`) to orchestrate a deployment sequence.

### **docker-compose.yml** <a id="docker-compose-file"></a>
- **Purpose**: Defines multi-container Docker applications. Typically includes services like `fastapi` and `milvus`.  
- **Contents**: Might contain references to images for Milvus, or a Python-based server container.  
- **Role in Pipeline**: When you run `docker-compose up -d`, it stands up the environment—network, containers, volumes, etc.—all in one go.

### **dockerfile.server** <a id="dockerfile-server"></a>
- **Purpose**: A Dockerfile used to build an image for the FastAPI or Python-based server.  
- **Key Steps**: Usually includes copying `requirements.txt`, installing dependencies with `pip install`, and eventually running a command to start the server.  
- **Integration**: Combined with `docker-compose.yml` or used stand-alone with `docker build -f dockerfile.server .`.

### **embedEtcd.yaml** <a id="embeddetcd-file"></a>
- **Purpose**: A YAML configuration file for embedded Etcd usage. This is often used with Milvus in standalone mode so that Etcd runs inside the same container.  
- **Contents**: Contains settings for client URLs, compaction settings, etc.  
- **When**: Typically mounted into the Milvus container so it can read from this config to start embedded Etcd.

### **embed_milvus.py** <a id="embed-milvus-py"></a>
- **Purpose**: A Python script that likely handles the logic of reading names from files and inserting them into Milvus after generating embeddings.  
- **Core**: Usually includes references to SentenceTransformers or other embedding libraries, then sets up connections to Milvus.

### **entrypoint.sh** <a id="entrypoint-sh"></a>
- **Purpose**: Script used as the container’s entrypoint (particularly in `dockerfile.server`), orchestrating final startup steps.  
- **Typical Commands**: Might run a Python server with `uvicorn server_search_names:app --host 0.0.0.0 --port 8000` or another main process.

### **etc Directory** <a id="etc-directory"></a>
```
etc/
├── embed_names.py
├── embed_names_redis.py
├── get_names.py
├── main.py
├── search.py
└── search_milvus.py
```
- **Purpose**: A collection of miscellaneous Python scripts, possibly prototypes or side utilities:  
  - `embed_names.py`: Another script for embedding logic.  
  - `embed_names_redis.py`: Possibly deals with storing or retrieving embeddings from Redis.  
  - `get_names.py`: Script for retrieving name data from external sources or local files.  
  - `main.py`: Could be an alternate entry point for a different approach or older testing code.  
  - `search.py`: Possibly a basic script to test searching functionality in a local or remote DB.  
  - `search_milvus.py`: Likely a specialized script focusing specifically on searching names within Milvus.

### **requirements.txt** <a id="requirements-txt"></a>
- **Purpose**: Lists all Python dependencies needed for the project. Typically read by `pip install -r requirements.txt`.  
- **Common Inclusions**:  
  - `fastapi`, `pydantic`, `pymilvus`, `sentence-transformers`, `prometheus_client`, etc.

### **server_search_names.py** <a id="server_search_names-py"></a>
- **Purpose**: A main FastAPI server script that defines routes (e.g., `/search/{name_input}`) to perform name lookups.  
- **Workflow**: Often includes code to connect to Milvus or Redis, embed queries, and return search results.  
- **Interaction**: Integrates with `utils/`, `data_names/`, and the `MilvusSearch` class to respond to real-time requests.

### **standalone_embed.sh** <a id="standalone-embed-sh"></a>
- **Purpose**: Another shell script that typically starts or manages a single Milvus instance with embedded Etcd.  
- **Usage**: Called manually or by other scripts. Possibly includes Docker run commands.  
- **Relevance**: Key for spinning up a local environment for embedding and searching.

### **test Directory** <a id="test-directory"></a>
```
test/
└── test.ipynb
```
- **Purpose**: Contains Jupyter notebooks or other testing frameworks.  
- **Example**: `test.ipynb` might have initial sanity checks for embeddings or name lookups.

### **user.yaml** <a id="user-yaml"></a>
- **Purpose**: Another YAML config meant to override or extend default settings (for example, the default `milvus.yaml`).  
- **Usage**: Possibly mounted into the Milvus container to override environment or DB settings.

### **utils Directory** <a id="utils-directory"></a>
```
utils/
├── convert_between_latin_and_cyril.py
├── docker_checker.py
├── download_embedding_model.py
├── download_nltk.py
├── embedddigs.py
├── initializers.py
├── install.py
├── load_model.py
├── loop.py
├── printing.py
├── shell_scripts/
│   ├── deploy.classifier_server.sh
│   ├── deploy.sh
│   ├── development.deploy.classifier_server.sh
│   ├── development.deploy.sh
│   └── restart.sh
└── timer.py
```
- **Purpose**: Houses various utility scripts and subdirectories. Each file typically addresses a small cross-cutting concern or shared function.  
  - `convert_between_latin_and_cyril.py`: Name transliteration logic between Latin script and Cyrillic.  
  - `docker_checker.py`: Possibly checks if Docker is installed or if containers are running.  
  - `download_embedding_model.py`: Script for downloading large pre-trained models if not found locally.  
  - `download_nltk.py`: Installs or downloads NLTK data for text processing.  
  - `embedddigs.py`: Contains classes or functions that handle embedding creation or searching logic.  
  - `initializers.py`: Might set up environment variables or default parameters.  
  - `install.py`: Possibly a script that orchestrates pip or system installs for dependencies.  
  - `load_model.py`: Another specialized script for loading ML or NLP models into memory.  
  - `loop.py`: A utility loop script or a specialized concurrency function.  
  - `printing.py`: Contains custom print functions with color-coded output (`printgreen`, `printblue`, `printred`).  
  - `shell_scripts/`: Contains scripts for deployment, development, restarts.  
  - `timer.py`: A small Python decorator or function for timing code execution.

---

## **How the Project Works:**

### 1. **Data Ingestion and Preparation**
This project starts with a large volume of name files, located under `data_names/`. Each file has lines of text, each line representing a distinct name (e.g., “Ivan”, “Olga”, “Ravshan”, etc.). The **scripts** that parse these files typically do the following:

1. **Read** the raw text line by line.  
2. **Normalize** or cleanse the data (trimming whitespace, removing special characters).  
3. **Combine** or merge data if necessary (e.g., merging male/female names into one file).

Once read, these names are ready to be **embedded** into vector form. This bridging of text to numerical vector is crucial for similarity-based searches in vector databases like Milvus.

### 2. **Embedding Mechanisms**
The embedding often uses the [SentenceTransformers library](https://www.sbert.net/), specifically the `all-MiniLM-L6-v2` model or any user-specified model. Key steps:

1. **Load Model**: You might see calls like `model.dense.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")`.  
2. **Encode**: The script transforms each name from plain text into a floating-point vector, typically 384 or 768 dimensions (depending on model).  
3. **Store**: These vectors are upserted into Milvus or another database along with an ID or name field, enabling subsequent retrieval.

The embedding scripts could be found in `embed_milvus.py`, `embed_names.py`, or `embedddigs.py` in `utils/`. Each script ensures consistent embedding transformations.

### 3. **Search and Retrieval Logic**
When the FastAPI server (in `server_search_names.py`) receives a query, it:

1. **Parses** the input string into tokens (individual name segments).  
2. **Optionally** handles transliteration or language detection (e.g., `convert_between_latin_and_cyril.py`).  
3. **Embeds** each token on the fly, generating vector representations.  
4. **Searches** Milvus collections (“rus_names”, “uzbek_names”) using approximate nearest neighbor (ANN) queries.  
5. **Ranks** the results by distance or similarity, returning them as JSON.

### 4. **Dockerization and Deployment**
Several Docker-related files (`docker-compose.yml`, `dockerfile.server`, `standalone_embed.sh`, `deploy.sh`) ensure you can spin up everything quickly. For example:

- **`standalone_embed.sh`**: Creates or starts a Milvus container with embedded Etcd.  
- **`docker-compose.yml`**: Could run both the FastAPI server and Milvus container on a shared network.  
- **`dockerfile.server`**: Builds a container image that includes your Python code plus dependencies.

The advantage: You don’t need to install the entire environment manually. Once Docker is running, you can simply do `./standalone_embed.sh start` or `docker-compose up -d` to have the entire system working.

---

## **Visual Flow Diagrams**

Below are **four visual flows** , illustrating how the project components interact. Right below each flow is an explanation of **how** it works.

### **Visual Flow 1: Data Preparation Process** <a id="visual-flow-1-data-preparation-process"></a>

```
+---------------------------+
|     data_names/*.txt      |
| (Raw name data per line)  |
+-----------+---------------+
            |
            v
+---------------------------+
|  Script (embed_milvus.py) |
|  or etc/embed_names.py    |
+-----------+---------------+
            |
            |  Read lines
            v
+---------------------------+
|  Embedding Model          |
|  (SentenceTransformers)   |
+-----------+---------------+
            |
            |  Convert name -> vector
            v
+---------------------------+
|   Output: List of Vectors |
+-----------+---------------+
            |
            v
+---------------------------+
|   Milvus (Insert Vectors) |
|   - rus_names, uzbek_names|
+---------------------------+
```

**Explanation**  
1. The script reads raw text names from `data_names/`.  
2. An embedding model converts each name into a vector.  
3. Those vectors are inserted into Milvus for future queries.

---

### **Visual Flow 2: Running Milvus Locally with Embedded Etcd** <a id="visual-flow-2-running-milvus-locally-with-embedded-etcd"></a>

```
[standalone_embed.sh] 
     |
     v
+-----------------------------------+
| Check if 'milvus-standalone'      |
| container exists/running          |
+----------------+------------------+
                 |
                 | If not running
                 v
+-----------------------------------+
| Generate embedEtcd.yaml           |
| Generate user.yaml                |
+----------------+------------------+
                 |
                 | docker run -d ...
                 v
+-----------------------------------+
| Docker Container: milvus-standalone
| - Milvus + embedded Etcd inside
|   reading embedEtcd.yaml config
+----------------+------------------+
                 |
                 v
+-----------------------------------+
| Wait for Container "healthy"      |
+-----------------------------------+
                 |
                 v
 [ Milvus ready at port 19530 ]
```

**Explanation**  
1. `standalone_embed.sh start` checks if the container is up. If not, it creates configuration files (`embedEtcd.yaml`, `user.yaml`).  
2. It launches a Docker container with the Milvus image. Inside this container, Etcd is embedded.  
3. The script polls Docker until the container is marked “healthy,” indicating that Milvus is ready to accept requests on port **19530**.

---

### **Visual Flow 3: FastAPI Search Workflow** <a id="visual-flow-3-fastapi-search-workflow"></a>

```
         +--------------------------+
 User -->|  /search/{name_input}    | (HTTP GET)
 Request  +-----------+------------+
                     |
                     | (FastAPI: server_search_names.py)
                     v
          +-------------------------+
          | Parse 'name_input'     |
          | Possibly transliterate |
          +-----------+------------+
                      |
                      | embed name
                      v
          +-------------------------+
          | Embedding Function     |
          | (all-MiniLM-L6-v2)     |
          +-----------+------------+
                      |
                      | Vector
                      v
          +-------------------------+
          |  Milvus (rus/uz names) |
          |  Query for matches     |
          +-----------+------------+
                      |
                      | Similarity results
                      v
          +-------------------------+
          |  Return JSON to user   |
          |  with matched names    |
          +------------------------+
```

**Explanation**  
1. A user calls `GET /search/NameX`.  
2. The FastAPI route processes the request and calls the embedding function.  
3. The embedding is sent to Milvus for vector search.  
4. The closest matches are returned to the user as JSON.

---

### **Visual Flow 4: Interaction with External Repositories or Services** <a id="visual-flow-4-interaction-with-external-repositories-or-services"></a>

```
[Local Repo/Codebase]
|  (Your Scripts,
|   Dockerfiles,
|   data_names)
+---------------------+
        |
        | (pull/push images)
        v
[ DockerHub / Docker Registry ]
| (milvusdb/milvus:v2.4.5)
+---------------------+
        |
        | Milvus Container
        v
[Local Docker Engine]
|  Runs milvus-standalone
|  Possibly runs FastAPI
+---------------------+
        |
        | (User requests)
        v
[ Internet/External Clients ]
   (Sends HTTP calls to your
   local or cloud server)
```

**Explanation**  
1. Your local repository interacts with **DockerHub** (or any registry) to pull/push images.  
2. Docker containers (Milvus + FastAPI) run locally via Docker Engine or Docker Compose.  
3. End-users connect to your local or cloud-hosted environment, making HTTP requests to the FastAPI endpoints.

---

## **Interactions with Other Repositories** <a id="interactions-with-other-repositories"></a>

While the provided tree does not explicitly list external submodules, this repository *could* reference or interact with other repos in the following ways:

- **PyPI Libraries**: Many imports in `requirements.txt` are actually packages from the Python Package Index, not local repos.  
- **Docker Images**: `milvusdb/milvus:v2.4.5` is pulled from DockerHub.  
- **Shared libraries**: If you see references in the code to something like `myorg/nameparser` (just an example), it might be a private GitHub repo.

In a typical workflow, you clone this repo, ensure Docker is installed, run the appropriate shell scripts, and everything needed is either in this directory or automatically fetched from DockerHub or PyPI.

---

## **Step-by-Step Usage Guide** <a id="step-by-step-usage-guide"></a>

Here is a suggested workflow for a new user who wants to replicate or test the system:

1. **Clone the Repo**:  
   ```bash
   git clone https://github.com/your-repo/names-embedding-search.git
   cd names-embedding-search
   ```

2. **Check Prerequisites**:  
   - Docker installed?  
   - Python 3.9+ installed (if you want to run Python scripts locally instead of Docker)?  
   - Enough system memory and disk space? (This is important for large embedding models.)

3. **Install Dependencies** (if running locally):  
   ```bash
   pip install -r requirements.txt
   ```
   Or, if you want to rely on Docker for everything, skip this step.

4. **Start Milvus with Embedded Etcd**:  
   ```bash
   ./standalone_embed.sh start
   ```
   - Wait for the script’s output: “Milvus started successfully.”  
   - Verify container is running: `docker ps`

5. **Embed the Names**:  
   - One approach is to run `python embed_milvus.py`, which will read the data from `data_names/`, embed them with `all-MiniLM-L6-v2`, and store them in the Milvus instance.

6. **Start the FastAPI Server**:  
   - If you have a local environment:  
     ```bash
     uvicorn server_search_names:app --host 0.0.0.0 --port 8000
     ```
   - Or if you rely on `docker-compose.yml`, do:  
     ```bash
     docker-compose up -d
     ```
   - Alternatively, you might run `./entrypoint.sh` if it’s set to do the same.

7. **Check the API**:  
   - Go to `http://localhost:8000/docs` if you used FastAPI’s default docs.  
   - Try a GET request to `http://localhost:8000/search/SomeName`.  
   - You should see JSON results with “uzbek_names” and “russian_names” arrays containing matched items.

8. **Stop the Services**:  
   - `./standalone_embed.sh stop` to stop Milvus or `docker stop milvus-standalone`.  
   - If using `docker-compose`, do `docker-compose down`.

9. **Optional**: Modify `user.yaml` or `embedEtcd.yaml` for advanced config, then restart services.



