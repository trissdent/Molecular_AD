import asyncio, os
import pandas as pd
from dotenv import load_dotenv
from db_interaction import fetch_many_postgresql
from datetime import datetime

load_dotenv()
POSTGRESQL_URL = os.environ["POSTGRESQL_URL"]

def calculate_age(record_date_str, birth_year, birth_month):
    if not record_date_str or birth_year is None:
        return None
    try:
        record_date = datetime.strptime(record_date_str.split(' ')[0], '%Y-%m-%d')
        age = record_date.year - birth_year
        if birth_month:
            age += (record_date.month - birth_month) / 12.0
        return round(age, 2)
    except Exception:
        return None

def fetch_demographics(image_ids):
    if not image_ids:
        return []
    
    clean_ids = [str(id).strip() for id in image_ids if id]
    
    query = """
    WITH image_visits AS (
        SELECT image_id, visit_id, 'MRI' AS modality FROM mri WHERE image_id = ANY($1)
        UNION
        SELECT image_id, visit_id, 'DTI' AS modality FROM dti WHERE image_id = ANY($1)
    )
    SELECT 
        iv.image_id,
        s.adni_id,
        m.total_score AS mmse,
        s.gender,
        s.hand,
        s.birth_year,
        s.birth_month,
        v.record_date,
        v.diagnosis,
        v.education,
        iv.visit_id,
        iv.modality
    FROM image_visits iv
    JOIN visit v ON iv.visit_id = v.id
    JOIN subject s ON v.subject_id = s.id
    LEFT JOIN mmse m ON m.visit_id = iv.visit_id;
    """
    
    records = asyncio.run(fetch_many_postgresql(query, (clean_ids,), POSTGRESQL_URL))
    
    for r in records:
        r['age'] = calculate_age(r['record_date'], r['birth_year'], r['birth_month'])
        r['gender_str'] = 'Male' if r['gender'] == 1 else 'Female' if r['gender'] == 2 else 'Unknown'
        
    return records

if __name__ == "__main__":
    DATA_DIR = "/home/minhtri/Molecular_AD/data"
    
    all_ids = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(DATA_DIR, filename), "r") as f:
                ids = [id.strip() for id in f.read().replace(',', ' ').split() if id.strip()]
                all_ids.extend(ids)
    
    all_ids = list(set(all_ids))
    print(f"Total unique IDs: {len(all_ids)}")
    
    records = fetch_demographics(all_ids)
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(DATA_DIR, "all_demographics_2.csv"), index=False)