# 🎬 Movie Recommendation System

A **Content-Based Movie Recommendation System** built using **Python, Pandas, Scikit-learn, TF-IDF, Cosine Similarity, and Streamlit**.

The system recommends the **top 5 movies similar to a movie selected by the user**, based on genre similarity.

---

## 📌 Project Overview

This project analyzes movie genres and uses **Natural Language Processing (NLP)** techniques to convert genres into numerical vectors.

The similarity between movies is then calculated using **Cosine Similarity**.

The user can select a movie from the Streamlit interface, click **Recommend**, and get 5 similar movies.

### 🔄 Project Workflow

```text
Movie Dataset
      ↓
Load Dataset
      ↓
Clean Genres
      ↓
Split Genres
      ↓
Create Final Text
      ↓
TF-IDF Vectorization
      ↓
Cosine Similarity
      ↓
Find Similar Movies
      ↓
Top 5 Recommendations
      ↓
Display using Streamlit
```

---

## 🧠 Recommendation Technique

This project uses **Content-Based Filtering**.

Instead of using user ratings or other users' preferences, the system compares the **genres of movies**.

For example:

```text
Toy Story (1995)

Adventure | Animation | Children | Comedy | Fantasy
```

The genres are converted into:

```text
Adventure Animation Children Comedy Fantasy
```

This final text is then processed using TF-IDF.

---

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* TF-IDF Vectorizer
* Cosine Similarity
* Streamlit

---

## 📂 Dataset

The project uses a `movies.csv` dataset containing **9,659 movies**.

The dataset contains three columns:

| Column    | Description                      |
| --------- | -------------------------------- |
| `movieId` | Unique ID of the movie           |
| `title`   | Movie title                      |
| `genres`  | Genres associated with the movie |

Example:

```text
movieId: 1
title: Toy Story (1995)
genres: Adventure|Animation|Children|Comedy|Fantasy
```

---

## 🧹 Data Preprocessing

The movie genres are separated using the `|` character.

For example:

```text
Adventure|Animation|Children|Comedy|Fantasy
```

is converted into:

```text
['Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy']
```

The genres are then joined together to create a single text representation:

```text
Adventure Animation Children Comedy Fantasy
```

This is stored in the `final_sentence` column.

## The same preprocessing is implemented in both the recommendation script and Streamlit application.

## 🔢 TF-IDF Vectorization

TF-IDF (**Term Frequency-Inverse Document Frequency**) is used to convert the movie genre text into numerical vectors.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

model = TfidfVectorizer()
X = model.fit_transform(df["final_sentence"])
```

Each movie is represented as a numerical vector based on its genres.

---

## 📐 Cosine Similarity

After converting the genres into numerical vectors, **Cosine Similarity** is used to calculate how similar the movies are.

```python
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(X)
```

The resulting similarity matrix contains similarity scores between the movies.

A higher similarity score means that the movies have more similar genre information.

---

## 🎯 Recommendation Logic

When a movie is selected, its index is found in the dataset.

The system then:

1. Gets the similarity scores for that movie.
2. Sorts the scores in descending order.
3. Removes the selected movie itself.
4. Selects the top 5 similar movies.
5. Displays their titles.

The recommendation logic is:

```python
movie_index = df[df["title"] == movie_name].index[0]

similar_movies = similarity[movie_index].argsort()[::-1][1:6]
```

## The same approach is used in the Streamlit application.

## 🖥️ Streamlit Application

The project includes a Streamlit interface called:

**🎬 Movie Recommendation System**

The application provides:

* Movie selection dropdown
* Dataset overview
* Total movie count
* Unique genre count
* Recommend button
* Top 5 movie recommendations
* Genres of recommended movies

The Streamlit page is configured with the title **"Movie Recommendation System"** and a centered layout.

### Dataset Overview

The sidebar displays:

```text
Total Movies
Unique Genres
```

These values are calculated directly from the dataset.

### Movie Selection

The application uses a dropdown instead of requiring the user to manually type the movie name. This helps prevent movie-name typing errors.

### Recommendations

After clicking **Recommend**, the application displays the top 5 similar movies along with their genres.

---

## 📁 Project Structure

```text
Movie-Recommendation/
│
├── .gitattributes
├── app.py
├── movie recommendation.py
├── movies.csv
└── README.md
```

### Files

#### `app.py`

The main **Streamlit application**.

It loads the dataset, processes genres, calculates similarity, allows movie selection, and displays recommendations.

#### `movie recommendation.py`

The main development/testing script used to:

* Explore the dataset
* Check missing values
* Check duplicates
* Process genres
* Apply TF-IDF
* Calculate cosine similarity
* Generate movie recommendations

The script also prints dataset information and similarity results during development.

#### `movies.csv`

The movie dataset containing:

```text
movieId
title
genres
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the Project

```bash
cd Movie-Recommendation
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

### 5. Install Required Libraries

```bash
pip install pandas scikit-learn streamlit
```

---

## ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your web browser.

---

## 🎮 How to Use

### Step 1

Run the Streamlit application:

```bash
streamlit run app.py
```

### Step 2

Select a movie from:

```text
Type or select a movie you like
```

### Step 3

Click:

```text
Recommend
```

### Step 4

The system displays:

```text
Recommended Movies:

1. Movie Name
2. Movie Name
3. Movie Name
4. Movie Name
5. Movie Name
```

along with the genres of each recommendation.

---

## 📊 Example

If the user selects:

```text
Toy Story (1995)
```

the system compares its genre vector against the other movies and returns the five movies with the highest similarity scores.

The recommendations are generated automatically from the similarity matrix rather than being manually specified.

---

## 🧪 Concepts Practiced

This project helped practice:

* Python
* Pandas
* Data preprocessing
* Missing-value handling
* Duplicate checking
* String manipulation
* NLP fundamentals
* TF-IDF
* Vectorization
* Cosine Similarity
* Similarity matrices
* Content-Based Recommendation
* Streamlit
* Python functions
* Caching with Streamlit

---

## 🔮 Future Improvements

The current system can be improved further by adding:

* Fuzzy movie-name matching
* Movie posters
* Movie descriptions
* Movie ratings
* Release year filtering
* Genre filtering
* Search suggestions
* Movie details
* Popularity-based recommendations
* Hybrid recommendation system
* Collaborative filtering
* External movie API integration

---

## ⚠️ Current Limitation

The recommendation system is based **only on movie genres**.

Therefore, two movies with similar genres may be recommended even if their story, actors, or audience preferences are very different.

Adding information such as descriptions, keywords, cast, ratings, and other metadata could make the recommendations more sophisticated.

---

## 🎯 Project Type

**Machine Learning / NLP / Recommendation System**

### Recommendation Method

**Content-Based Filtering**

### Main Techniques

**TF-IDF + Cosine Similarity**

---

## 👨‍💻 Author

**Deepak Garg**

BCA Student | AI/ML & Generative AI Enthusiast

---

## ⭐ Support

If you found this project useful, consider giving the repository a ⭐ on GitHub.
