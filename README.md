# Parkinsons – ML App (Pro, v3)

פרויקט סופי עם EDA ומנגנון **auto-create** לדאטה:
- אם `data/parkinsons.csv` לא קיים (למשל בענן), האפליקציה יוצרת אותו אוטומטית וממשיכה לרוץ.
- אימון עם CV ו-GridSearch, קידום Candidate→Production, גרפים (ROC/PR/CM), ו-Batch/Single Predict.

## ריצה
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## מבנה
```
parkinsons_streamlit_pro_final_v3/
├─ config.py
├─ model_pipeline.py
├─ streamlit_app.py
├─ data/parkinsons.csv
├─ assets/single_row_template.csv
├─ models/  (נוצר בזמן ריצה)
└─ requirements.txt
```
