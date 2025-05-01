from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path

app = Flask(__name__)

# Initialize database
def init_db():
    db_path = Path('recommender.db')
    if db_path.exists():
        return
    
    conn = sqlite3.connect('recommender.db')
    c = conn.cursor()
    
    # Create tables
    c.execute('''CREATE TABLE news
                 (id INTEGER PRIMARY KEY, title TEXT, content TEXT, keywords TEXT)''')
    
    c.execute('''CREATE TABLE blogs
                 (id INTEGER PRIMARY KEY, title TEXT, content TEXT, author TEXT, tags TEXT)''')
    
    c.execute('''CREATE TABLE music
                 (id INTEGER PRIMARY KEY, title TEXT, artist TEXT, album TEXT, genre TEXT, duration INTEGER)''')
    
    c.execute('''CREATE TABLE users
                 (id INTEGER PRIMARY KEY, username TEXT, preferences TEXT)''')
    
    c.execute('''CREATE TABLE ratings
                 (user_id INTEGER, item_id INTEGER, item_type TEXT, rating REAL)''')
    
    # Insert sample data
    news = [
        (1, 'Climate Summit', 'World leaders gather to discuss climate change...', 'environment, politics'),
        (2, 'Tech Breakthrough', 'New AI model achieves human-level performance...', 'technology, AI')
    ]
    c.executemany('INSERT INTO news VALUES (?,?,?,?)', news)
    
    blogs = [
        (1, 'Travel Guide', 'My amazing experience visiting Japan...', 'Traveler123', 'travel, japan'),
        (2, 'Python Tips', 'How to optimize your Python code...', 'CodeMaster', 'programming, python')
    ]
    c.executemany('INSERT INTO blogs VALUES (?,?,?,?,?)', blogs)
    
    music = [
        (1, 'Bohemian Rhapsody', 'Queen', 'A Night at the Opera', 'rock', 354),
        (2, 'Blinding Lights', 'The Weeknd', 'After Hours', 'pop', 203)
    ]
    c.executemany('INSERT INTO music VALUES (?,?,?,?,?,?)', music)
    
    conn.commit()
    conn.close()

# Initialize the database when app starts
init_db()

# Helper function to get recommendations
def get_recommendations(content_type, item_id, n=3):
    conn = sqlite3.connect('recommender.db')
    
    if content_type == 'news':
        df = pd.read_sql('SELECT * FROM news', conn)
        text_field = 'content + " " + keywords'
    elif content_type == 'blogs':
        df = pd.read_sql('SELECT * FROM blogs', conn)
        text_field = 'content + " " + tags'
    elif content_type == 'music':
        df = pd.read_sql('SELECT * FROM music', conn)
        text_field = 'artist + " " + genre + " " + album'
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df[text_field])
    
    idx = df[df['id'] == item_id].index[0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    
    recommendations = []
    for i, score in sim_scores:
        rec = {'id': df.iloc[i]['id'], 'score': round(score, 2)}
        if content_type == 'music':
            rec.update({
                'title': df.iloc[i]['title'],
                'artist': df.iloc[i]['artist']
            })
        else:
            rec['title'] = df.iloc[i]['title']
        recommendations.append(rec)
    
    conn.close()
    return recommendations

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/news')
def news():
    conn = sqlite3.connect('recommender.db')
    news_items = pd.read_sql('SELECT * FROM news', conn)
    conn.close()
    return render_template('news.html', news_items=news_items.to_dict('records'))

@app.route('/blogs')
def blogs():
    conn = sqlite3.connect('recommender.db')
    blog_items = pd.read_sql('SELECT * FROM blogs', conn)
    conn.close()
    return render_template('blogs.html', blog_items=blog_items.to_dict('records'))

@app.route('/music')
def music():
    conn = sqlite3.connect('recommender.db')
    music_items = pd.read_sql('SELECT * FROM music', conn)
    conn.close()
    return render_template('music.html', music_items=music_items.to_dict('records'))

@app.route('/recommend/<content_type>', methods=['POST'])
def recommend(content_type):
    item_id = int(request.form['item_id'])
    recommendations = get_recommendations(content_type, item_id)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
