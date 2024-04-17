<h1 align="center" id="title">SmartSquad - Your Fantasy Premier League Advisor</h1>
<p align="center">
  <img src="https://github.com/yonatanko/SmartSquad/blob/main/app_image.png" width=300 />
</p>
<p id= "description">Welcome to SmartSquad! An AI-based agent that will raise your performance in The Fantasy competition to the next level.</p>
<p>To get started, Head to our app: https://smartsquad-kyapteetv7kedimlycvwjd.streamlit.app/ </p>

<h2>:gear: Configurations </h2>
<p>Our entire app is Web-based and accessible through the link added above.</p>
<p>If you wish to enter you own Gemini key, follow the following steps:</p>
<p> 1. Edit the config.json file</p>

```
{
  "gemini_ky": {enter your key here}
}
```
<p> 2. Edit utils.py get_gemini_key function. comment the first line and uncomment the rest. </p>

```
def get_gemini_key():
    gemini_key = st.secrets["gemini_key"] # get the api key from the streamlit secrets

    # here you can set the api key for the model by using the configure method:
    #
    # with open('config.json', 'r') as file:
    #         config = json.load(file)
    # gemini_key = config['gemini_key']
    #
    return gemini_key
```

<h2>:recycle: Reproduction of Model results</h2>
<p>If you wish to reproduce the scores_df.csv and the difficulties_df.csv : </p>
<p> 1. Clone the GitHub repository:</p>

```
git clone https://github.com/yonatanko/SmartSquad.git
```
<p> 2. Install the required packages using pip: </p>

```
pip install -r requirements.txt
```

<p> 3. Run the Predectors.py </p>

```
python Predectors.py
```
Both CSV files will appear in your folder afterward.

<h2>:file_folder: Project Structure</h2>

```
.
├── .streamlit                  # Streamlit configuration files
├── Fantasy-Premier-Leaguue     # Static Data Folder                   
├── data_collection             # Dynamic Data collection folder
    ├── fpl_api_collection.py   # Dynamic Data collection code         
├── pages                       # Web pages folder
    ├── main_page.py            # Main page code
    ├── stats_page.py           # Statistics and charts page code
├── Builders.py                 # Builder functions for the AI prediction algorithms in Predectors.py
├── Predectors.py               # Scores and difficulties prediction code
├── SmartSquad.py               # Welcome page code
├── app_image.png               # app logo
├── difficulties_df.csv         # fixtures difficulties data predicted by our AI model
├── scores_df.csv               # Players' scores prediction data predicted by our AI model
├── requirements.txt            # Requirements of the Project
└── README.md
```
