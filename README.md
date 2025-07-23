# Electric and Alternative Fuel Vehicle ETL Pipeline

## 📌 Overview

This project provides an **ETL (Extract, Transform, Load)** script that processes the  *Electric Vehicle Population* dataset and prepares it for loading into a data warehouse following a **star schema**.

The code is written in **Python** and includes data exploration, cleaning, transformation, and loading logic.

---

## 📁 Dataset

- **Source**: [Electric Vehicle Population Data - Washington State](https://data.wa.gov/api/views/f6w7-q2d2/rows.csv)
- **Description**: Contains details of electric and alternative fuel vehicles registered in Washington State.

---

## ⚙️ Steps Performed

### 1. Extract
- Loaded CSV dataset from the state data portal.

### 2. Explore
- Displayed dataset size and types.
- Computed summary statistics for key numerical features (`Electric Range`, `Base MSRP`).
- Visualized distributions of categorical features such as:
  - `Model Year`
  - `Make`
  - `Electric Vehicle Type`
  - `CAFV Eligibility`
  - `Electric Utility`, etc.

### 3. Clean and Transform
- Handled missing data:
  - Filled numeric fields based on grouped values by `Model`, `Year`, and `Make`.
  - Filled string-type missing values with `'Unknown'`.
- Converted data types for efficient storage.
- Encoded categorical fields (`EV Type`, `CAFV Eligibility`, `Electric Utility`) using `LabelEncoder`.

### 4. Load
- Constructed dimension tables:
  - `dim_vehicle`
  - `dim_location`
  - `dim_EVtype`
  - `dim_CAFV`
  - `dim_electric_utility`
- Constructed `fact_ev` fact table with `event_id` as primary key.
- Extracted longitude and latitude from location strings for spatial data storage.

- Included sample logic to load each table into a Microsoft SQL Server database.

---

## 🧱 Data Warehouse Schema

The schema follows a **star model**:
      dim_EVtype     dim_CAFV
           |           |
            dim  vehicle 
                 |
dim_location — fact_ev — dim_utility

---

## ▶️ How to Run

### Prerequisites
- Python 3.7+
- Required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, `pymssql`, `keyring`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ev-etl-pipeline.git
   cd ev-etl-pipeline

2. Install required packages:
pip install -r requirements.txt

3. Run the script:
python etl_script.py

4. Ensure your SQL Server credentials are securely managed with keyring or environment prompts.



