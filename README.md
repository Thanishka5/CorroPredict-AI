# CorroPredict AI — Cloud deployment

This project is a Streamlit application for estimating mild-steel weight loss from
NaCl concentration and exposure duration using a Random Forest Regression model.

## Deploy without installing Python on your computer

1. Sign in to GitHub and create a new repository named `CorroPredict-AI`.
2. Upload `app.py`, `requirements.txt`, and the `data` folder (including
   `aqueous_corrosion_data.csv`) to the repository.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app**, select your repository and branch, and set the main file
   path to `app.py`.
5. Deploy. Streamlit Community Cloud will install the requirements and publish
   the app at a shareable URL.

## Dataset scope

The model uses only `nacl_pct` and `exposure_days` as inputs and predicts
`weight_loss_mg`. Temperature, pH and humidity are not model inputs; they are
discussed separately in the theoretical part of the project.

The app calculates leave-one-out cross-validation metrics from the included
dataset. These metrics describe performance on this dataset and are not a
guarantee of general accuracy.
