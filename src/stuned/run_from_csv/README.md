# Run experiments from csv (Google Sheet)

## Generate credentials

To use Google Sheets for jobs submission
please generate "credentials.json" by using instructions from here: https://docs.gspread.org/en/latest/oauth2.html#for-end-users-using-oauth-client-id
(NOTE: put it inside ~/.config/gauth/credentials.json instead of ~/.config/gspread/credentials.json).

## Examples

- For main.py and config examples please see: "./demo/" folder.

- Google Sheets example can be found here: https://docs.google.com/spreadsheets/d/1yvdSDBvLIOJ1sBdrzylX5CcxuekISF9b7xWTUX-B6T0/edit?usp=sharing

- Example usage:

    ```Bash
    python -m stuned.run_from_csv --csv_path https://docs.google.com/spreadsheets/d/1yvdSDBvLIOJ1sBdrzylX5CcxuekISF9b7xWTUX-B6T0::"expanded_DemoExp" --conda_env ${CONDA_ENV} # Note: $CONDA_ENV should be the anaconda environment which has "stuned" package installed
    ```

TODO(Alex | 30.04.2023): write readme
