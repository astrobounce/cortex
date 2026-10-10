## Cortex is an agentic version of querying database in natural language 

## How to run ?

### Step 1: Create Virtual environment
```
python3 -m venv .venv
```
### Step 2: Install Dependencies
```
pip install -r requirements.txt
````
### Step3 : Set required environment variables
```
export ANTHROPIC_API_KEY=sk-ant-*
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/pagila
```
### Step 4: Run the Agent
```
python agent.py
```
