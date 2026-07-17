# Python-class-Session6-MiniProject
# Mini-Project: From Public API to a Deployed Classification Dashboard
-----

## Overview
In this project, you will build a small but complete data science pipeline, end to end:
Pull data from a public API that requires no authentication/token
Explore and clean it
Train and evaluate a classification model
Wrap everything in an interactive Streamlit dashboard
Deploy it for free so it's reachable at a public URL
This mirrors what a real applied data science task looks like: the modeling is often the smallest part - sourcing data, cleaning it, and shipping something usable matter just as much.
You can use AI for assistance.

## Learning Objectives
By the end of this project, you should be able to:
Retrieve structured data from a REST API using Python (requests), including handling pagination
Perform exploratory data analysis (EDA) and handle missing/inconsistent data with justified decisions
Engineer features and train/evaluate a classification model, and persist it for reuse
Build a multi-section interactive dashboard with Streamlit
Deploy a Python web app to a free public hosting platform and debug common deployment failures


## Prerequisites & Setup
Before starting Task 1, get your environment ready:
A free GitHub account (https://github.com/join) - required for deployment later (you should have one already)
Git installed locally (git --version to check).
An editor (VS Code, PyCharm, Jupyter — your choice).

## Suggested project structure to keep things organized from the start:
project/
├── data/
│   ├── raw_data.csv
│   └── clean_data.csv
├── notebooks/
│   └── eda.ipynb              (or a plain .py script — either is fine)
├── requirements.txt
└── README.md
Create the repo on GitHub early (even empty).


## Task 1 - Data Acquisition (No API Key Required)

### Step 1: Choose your API
Pick one of the following public, key-free APIs (or propose an alternative that also needs no token/registration — get it confirmed by your instructor first):

PokéAPI :
https://pokeapi.co/api/v2/pokemon?limit=200
Predict a Pokémon's primary type from its base stats (HP, attack, defense, speed, etc.)
Fun, well-documented; expect modest accuracy since stats genuinely overlap across types

   REST Countries : 
https://restcountries.com/v3.1/all
Predict a country's region/subregion from features like population, area, and number of languages
Single API call returns everything — no pagination needed

   Open Brewery DB :
https://api.openbrewerydb.org/v1/breweries
Predict brewery type (micro, brewpub, large, etc.) from location and name-derived features
Paginated; good practice for handling page/per_page params

### Step 2: Explore before you code
Before writing any script, open the endpoint directly in your browser (or a tool like Postman/Insomnia) and look at the raw JSON. Answer for yourself:
What does one record look like? Which fields do you actually need?
Is the data paginated, or does one call return everything?
Are there nested fields (e.g., a list of "types" or "stats") you'll need to flatten?

### Step 3: Write the fetch script
Install the library you need:
pip install requests pandas
Your script should:
Call the collection/list endpoint.
If the API paginates results, loop through pages (using limit/offset, page/per_page, or a next URL depending on the API) until you've collected enough records.
For APIs where the list endpoint only gives summary info (like PokéAPI), make a second call per item to get full details — add a short time.sleep(0.1–0.5) between calls so you don't hammer the free API.
Flatten nested JSON into flat columns (e.g., pull stats[i].base_stat into individual columns like hp, attack, defense).
Build a pandas.DataFrame from the collected records.
Save it to data/raw_data.csv with df.to_csv(..., index=False) — never hand-edit or manually download this file; it should be fully reproducible by re-running your script.

### Checklist for Task 1
[ ] Data pulled programmatically (no manual CSV download)
[ ] Pagination handled correctly (no silently-truncated dataset)
[ ] At least ~200 rows collected
[ ] Target column present and mostly non-null
[ ] raw_data.csv saved and committed to your repo


## Task 2 - EDA and Data Cleaning

### Step 1: First look
Note down, in plain language: how many rows/columns, which columns have missing values, and whether you see obvious duplicates or garbage values.

### Step 2: Visualize before you touch anything
Produce, at minimum:
A class-balance plot for your target column (bar chart of value counts) — this tells you if some classes barely have any examples, which will matter in Task 3.
A correlation heatmap of your numeric features.
2–3 relationship plots (boxplots of a feature grouped by class, scatter plots between two features colored by class, etc.).

### Step 3: Clean, with justification for each decision
Common cleaning steps to consider (not every dataset needs all of these):
Missing target values — you generally can't use these rows for training a classifier, so drop them.
Missing feature values — impute (median for skewed numeric data, mean for roughly symmetric data) or drop the column if it's missing too much to be useful (e.g., >40–50%).
Duplicates — drop exact duplicate rows.
Outliers — investigate rather than blindly remove; a Pokémon with unusually high stats might be a legendary, not an error. Use the IQR method or domain judgment.
Rare classes — if your target has classes with only 1–2 examples, consider merging them into an "other" bucket, since a model can't learn a pattern from a single example and it will break stratified train/test splitting otherwise.
Categorical encoding — one-hot encode or label-encode any categorical features (not the target, which gets its own LabelEncoder in Task 3).
Write a short paragraph (in a notebook markdown cell, or a NOTES.md) explaining why you made each decision — this is part of what gets graded, not just the resulting clean file.

### Step 4: Save your output
Checklist for Task 2
[ ] Missing values, duplicates, and outliers explicitly checked and addressed
[ ] Class balance plot produced
[ ] Correlation heatmap + 2–3 relationship plots produced
[ ] Cleaning decisions documented with reasoning
[ ] clean_data.csv saved and committed


## Task 3 - Classification Model

### Step 1: Prepare features and target

### Step 2: Train/test split
Use stratify=y_encoded so rare classes are proportionally represented in both splits.

### Step 3: Scale features (needed for linear/distance-based models)

### Step 4: Train at least two models
Pick a simple baseline plus something more flexible

### Step 5: Evaluate and compare
Use macro-averaged precision/recall/F1 (not accuracy alone) as your primary comparison metric when classes are imbalanced — accuracy can look deceptively high if one class dominates.
When computing the confusion matrix, explicitly pass labels=range(len(le.classes_)) so it always has one row/column per class, even if a rare class happens to be entirely in the train split for a given run

### Step 6: Save everything the dashboard will need
Saving these artifacts means your Streamlit app can load the trained model instantly instead of retraining it every time someone visits the dashboard.

### Checklist for Task 3
[ ] At least two classifiers trained and compared
[ ] Accuracy, macro-precision, macro-recall, macro-F1, and confusion matrix all reported
[ ] Brief written justification for which model you'd deploy
[ ] model.pkl, scaler.pkl, label_encoder.pkl, model_metrics.json saved and committed


## Task 4 - Streamlit Dashboard
Build a single app (app.py) with the following sections. Use st.cache_data for loading dataframes and st.cache_resource for loading the model/scaler/encoder, so they only load once per session instead of on every interaction.
#### Header / Intro section
Project title, one-paragraph description, a link to your data source.
A few headline metrics (row count, number of classes, which model is "best").
#### Data overview section
Raw vs. cleaned data preview, ideally in tabs or an expander.
Key stats: row/column counts, missing-value counts before vs. after cleaning.
#### EDA section
At least 2 interactive charts (Plotly or Altair, not static matplotlib images) — e.g., class-balance histogram, correlation heatmap.
A sidebar filter (e.g., a multiselect on your target categories) that updates the charts live.
#### Model performance section
A metrics table comparing all trained models.
A confusion matrix plot for the selected/best model.
A feature importance chart (from feature_importances_ for tree models, or coefficient magnitude for linear models).
#### Live prediction section
Input widgets (sliders/number inputs/select boxes) covering every feature the model needs.
A "Predict" button that scales the input, runs the saved model, and displays the predicted class — plus a probability bar chart if the model supports predict_proba.
#### Sidebar navigation 
connecting all five sections (e.g., st.sidebar.radio), and a small footer noting the data source and last-updated date.
Test locally before touching deployment
Open the local URL it prints and click through every section. Do not attempt deployment until the app runs cleanly locally — debugging is far easier on your own machine than through a deployment platform's logs.

### Checklist for Task 4
[ ] All 5 sections present and working
[ ] Sidebar navigation implemented
[ ] Model/data loading is cached (st.cache_data / st.cache_resource)
[ ] App runs with no errors via streamlit run app.py


## Task 5 - Free Deployment (Step-by-Step)
Deploy your dashboard so it's reachable at a public URL. Below are full walkthroughs for two free options — pick one (or try both).
Before you deploy, regardless of platform
[ ] Your entire project (code + requirements.txt + saved model artifacts: model.pkl, scaler.pkl, label_encoder.pkl, model_metrics.json, and your data CSVs) is committed to a git repository.
[ ] Your requirements.txt lists every package your app imports, e.g.:
streamlit pandas numpy scikit-learn plotly joblib requests
[ ] The app runs locally with no errors via streamlit run app.py.
[ ] Your code uses relative paths ("clean_data.csv"), never absolute local paths ("C:/Users/you/Desktop/..."), since the deployment environment has a different filesystem.

### Option A: Streamlit Community Cloud
2. Create a free account at https://share.streamlit.io by signing in with your GitHub account (this also authorizes Streamlit to see your repos).
3. Click "New app" in the Streamlit Cloud dashboard.
4. Fill in the deployment form:
Repository: <your-username>/<your-repo>
Branch: main
Main file path: app.py
5. (Optional) Open "Advanced settings" to pin a Python version matching your local environment, and to add any secrets (not needed for this project, since no API key is required for the data source).
6. Click "Deploy." Streamlit installs everything from requirements.txt and starts the app. First deploys typically take a couple of minutes — watch the build log that streams in.
7. Get your link. Once live, your app is reachable at https://<your-app-name>.streamlit.app. This is the URL to submit.
8. Updating your app. Any new commit pushed to main triggers an automatic redeploy — no need to redo the deployment steps.
9. Managing your app. From your Streamlit Cloud workspace, you can view logs, reboot the app, or delete it entirely.

### Option B: Hugging Face Spaces
1. Create a free account at https://huggingface.co/join (requires email verification).
2. Create a new Space: click your profile icon → New Space.
3. Configure the Space:
Name it (e.g., pokemon-type-classifier)
SDK: Streamlit
Hardware: CPU basic (the free tier)
Visibility: public or private, your choice
4. Create the Space. Hugging Face generates a git repository for it at a URL like: https://huggingface.co/spaces/<your-username>/<space-name>
5. Clone the (empty) Space repo locally:
git clone https://huggingface.co/spaces/<your-username>/<space-name>
cd <space-name>
6. Copy your project files in — app.py, requirements.txt, your saved model artifacts, and your CSVs.
7. Commit and push:
git add .
git commit -m "Initial dashboard"
git push
You'll be prompted for credentials on push. Use a Hugging Face access token as your password (generate one under Settings → Access Tokens, "write" permission). This token is only for pushing your code to your Space — it has nothing to do with the data API, so it doesn't conflict with the "no token needed for data" requirement of this project.
8. Watch the build. Hugging Face automatically detects the Streamlit SDK, installs requirements.txt, and launches the app. Follow progress under the Space's "Logs" tab.
9. Get your link. Once running, your dashboard is live at: https://huggingface.co/spaces/<your-username>/<space-name>
10. Sleep behavior. Free CPU-basic Spaces go to sleep after roughly 48 hours of no visits, and wake automatically (with a short delay) the next time someone opens the link — this is normal and expected, not a bug.

### Final deployment checklist
[ ] requirements.txt committed and accurate
[ ] Model/data artifacts committed to the repo
[ ] App verified locally with zero errors before deploying
[ ] Public URL opened in an incognito/private window to confirm it truly works for someone who isn't logged into your account
[ ] URL included in your final write-up/submission


## Deliverables
GitHub repository (or Hugging Face Space repo) containing: data-fetch script, cleaning/EDA notebook or script, model training script, app.py, requirements.txt, saved model artifacts, and a README.md with setup instructions.
Live deployed dashboard URL, verified working in an incognito window.
A one-page write-up covering: data source and why you chose it, key cleaning decisions and why, model comparison results, and one honest limitation or next step.
Troubleshooting FAQ
"My API pull only returns 20 results." - Most APIs cap results per call; check for a limit/per_page parameter or pagination links, and loop through pages until you have enough data.
"My model's accuracy is low." - For genuinely hard multi-class problems (e.g., predicting Pokémon type from stats alone), modest accuracy is expected and fine; the point of the project is the pipeline, not a leaderboard score. Explain this honestly in your write-up rather than chasing an artificially high number.
"It works locally but not when deployed." - Almost always one of: a missing package in requirements.txt, an absolute file path that doesn't exist on the server, or a file that wasn't actually committed to git. Check the platform's logs first.
"Do I need a credit card for either platform?" - No. Both Streamlit Community Cloud and Hugging Face Spaces (CPU-basic tier) are free with no payment method required for a project of this size.


## Resources
Streamlit docs: https://docs.streamlit.io
Streamlit Community Cloud docs: https://docs.streamlit.io/deploy/streamlit-community-cloud
Hugging Face Spaces docs: https://huggingface.co/docs/hub/spaces
scikit-learn model evaluation guide: https://scikit-learn.org/stable/modules/model_evaluation.html
