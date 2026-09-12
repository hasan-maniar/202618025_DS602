# Restaurant Tipping Behavior Analysis — Statistical Methods Lab 4

An exploratory and statistical analysis of the classic **Tips dataset**, examining
what factors influence how much customers tip at a restaurant, with an interactive
Streamlit app to explore the data live.

## 📁 Project Structure

```
Lab4_DS602/
├── data/
│   └── tips.csv                # dataset
├── analysis.py                 # data exploration, statistics, plots
├── app.py                      # Streamlit interactive app
├── requirements.txt
└── README.md
```

## ⚙️ Setup

```bash
git clone https://github.com/hasan-maniar/202618025_DS602.git
cd 202618025_DS602
pip install -r requirements.txt
```

## ▶️ Run the Analysis

```bash
python analysis.py
```

## ▶️ Run the App

```bash
python -m streamlit run app.py
```

**Live app:** `<<paste your Streamlit Cloud link here>>`

---

## 📊 Dataset

The Tips dataset contains **244 records** of restaurant bills, with the following columns:

| Column | Description |
|---|---|
| `total_bill` | Total bill amount ($) |
| `tip` | Tip amount given ($) |
| `sex` | Gender of the person paying the bill |
| `smoker` | Whether the party included a smoker |
| `day` | Day of the week |
| `time` | Lunch or Dinner |
| `size` | Number of people in the party |

## 🔍 Analysis Performed

- **Exploratory Data Analysis:** distribution of bill amounts and tip amounts, checked for outliers and missing values.
- **Correlation analysis:** examined the relationship between `total_bill` and `tip` — `<<state your finding, e.g. "a strong positive correlation (r ≈ 0.68) was found, meaning larger bills tend to come with larger tips">>`.
- **Group comparisons:** compared average tip amounts/percentages across `<<day / time / sex / smoker>>` to see if any group tips more generously than others.
- `<<If you ran a regression: "Built a linear regression model predicting tip amount from total_bill, size, and other features, achieving an R² of <<X>>.">>`
- `<<If you ran hypothesis tests: "Conducted a t-test / ANOVA to check whether tipping behavior differs significantly between <<groups>>, finding <<result, e.g. 'no statistically significant difference (p = 0.XX)'>>.">>`

## 📈 Key Findings

- `<<Fill in your top 2-3 takeaways here, e.g.:>>`
- Bill size is the strongest predictor of tip amount.
- Tip percentage tends to be slightly higher at dinner than lunch.
- Smoker status showed `<<little/no/some>>` measurable effect on tipping behavior.

## 💻 Streamlit App

The app lets users:
- Filter the dataset by day, time, or party size
- Visualize the relationship between bill amount and tip (scatter plot / regression line)
- View summary statistics (mean, median tip %) for selected filters

## 📝 Limitations

- Small dataset (244 records) — results may not generalize beyond this specific restaurant/time period.
- No information on service quality, seating location, or other factors that could plausibly affect tipping.
- Correlation findings do not imply causation.

## 🔗 Links
- **GitHub Repo:** https://github.com/hasan-maniar/202618025_DS602
- **Deployed App:** `<<paste your Streamlit Cloud link here>>`
