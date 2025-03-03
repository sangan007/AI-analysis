from flask import Flask, request, render_template, jsonify, session
from werkzeug.utils import secure_filename
from transformers import pipeline, DistilBertTokenizer
import textstat
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
import os

app = Flask(__name__)
app.secret_key = "call-quality-analyst"  
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    try:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        print(f"Created uploads folder at: {UPLOAD_FOLDER}")
    except Exception as e:
        print(f"Error creating uploads folder: {e}")

# Load Hugging Face models for transcription and sentiment analysis
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

@app.before_request
def set_default_settings():
    if 'theme' not in session:
        session['theme'] = 'classic-beige' 
    if 'font_size' not in session:
        session['font_size'] = 'medium'  

@app.route("/", methods=["GET", "POST"])
def home():
    report = None
    if request.method == "POST":
        if "audio" not in request.files:
            return jsonify({"error": "No audio file uploaded!"}), 400
        audio_file = request.files["audio"]
        
        # Save the audio 
        filename = secure_filename(audio_file.filename)
        audio_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        audio_file.save(audio_path)
        
        # Analyze the audio 
        report = analyze_audio(audio_path)
    
    return render_template("index.html", report=report, theme=session['theme'], font_size=session['font_size'])

@app.route("/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
       
        session['theme'] = request.form.get('theme', 'classic-beige')
        session['font_size'] = request.form.get('font_size', 'medium')
        return jsonify({"success": "Settings updated!"}), 200
    
    return render_template("settings.html", theme=session['theme'], font_size=session['font_size'])

def analyze_audio(audio_path):
    # Hugging Face's Whisper model 
    try:
        result = transcriber(audio_path)
        text = result["text"] if "text" in result else "No speech detected"
    except Exception as e:
        return {"Error": {"score": 0, "comment": f"Could not transcribe audio: {str(e)}"}}
    
    # Analyze the text with AI
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    
    # Check sentiment and question marks
    tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=512)
    truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)
    print(f"Token length after truncation: {len(tokens)}")  
    sentiment = sentiment_analyzer(truncated_text)[0]
    engagement_score = 15
    engagement_comments = []
    if sentiment["label"] == "POSITIVE":
        engagement_score += 3
        engagement_comments.append("The agent spoke positively, which helps engage the listener.")
    else:
        engagement_score -= 3
        engagement_comments.append("The agent’s tone was not very positive, which may reduce engagement.")
    question_count = text.count("?")
    if question_count > 2:
        engagement_score += 2
        engagement_comments.append("The agent asked questions, which is great for engaging the listener!")
    else:
        engagement_comments.append("The agent could ask more questions to engage the listener better.")
    
    # Flesch-Kincaid score
    readability = textstat.flesch_reading_ease(text)
    clarity_score = 15
    clarity_comments = []
    if readability > 60:
        clarity_score += 3
        clarity_comments.append("The agent’s language is easy to understand.")
    else:
        clarity_score -= 3
        clarity_comments.append("The agent used complex language that might be hard for beginners to follow.")
    long_sentences = sum(1 for sentence in sentences if len(word_tokenize(sentence)) > 20)
    if long_sentences > len(sentences) / 2:
        clarity_score -= 2
        clarity_comments.append("The agent used too many long sentences, which can reduce clarity.")
    
    # Product Knowledge
    product_keywords = ["course", "program", "training", "benefit", "feature"]
    keyword_count = sum(text.lower().count(keyword) for keyword in product_keywords)
    noun_count = len([word for word, pos in pos_tags if pos.startswith("NN")])
    product_score = 15
    product_comments = []
    if keyword_count > 3:
        product_score += 3
        product_comments.append("The agent mentioned product details well, showing good knowledge.")
    else:
        product_score -= 3
        product_comments.append("The agent didn’t mention enough product details.")
    if noun_count > len(words) * 0.3:
        product_comments.append("The agent used specific terms, which helps explain the product clearly.")
    
    # Listening Skills
    acknowledgment_words = ["yes", "I understand", "I see", "right", "okay"]
    acknowledgment_count = sum(text.lower().count(word) for word in acknowledgment_words)
    listening_score = 15
    listening_comments = []
    if acknowledgment_count > 2:
        listening_score += 3
        listening_comments.append("The agent used acknowledgment words, showing they listened well.")
    else:
        listening_score -= 3
        listening_comments.append("The agent could use more acknowledgment words to show they’re listening.")
    
    # Handling Objections
    objection_phrases = ["I understand your concern", "let me explain", "that’s a good point"]
    objection_count = sum(text.lower().count(phrase) for phrase in objection_phrases)
    objection_score = 15
    objection_comments = []
    if objection_count > 0:
        objection_score += 3
        objection_comments.append("The agent handled objections well by addressing concerns.")
    else:
        objection_score -= 3
        objection_comments.append("The agent didn’t address objections clearly.")
    
    # Closing
    closing_phrases = ["thank you", "next steps", "sign up", "register"]
    closing_count = sum(text.lower().count(phrase) for phrase in closing_phrases)
    closing_score = 15
    closing_comments = []
    if closing_count > 1:
        closing_score += 3
        closing_comments.append("The agent used good closing phrases to end the call.")
    else:
        closing_score -= 3
        closing_comments.append("The agent could use more closing phrases to end the call strongly.")
    
    # scores between 0 and 20
    engagement_score = min(20, max(0, engagement_score))
    clarity_score = min(20, max(0, clarity_score))
    product_score = min(20, max(0, product_score))
    listening_score = min(20, max(0, listening_score))
    objection_score = min(20, max(0, objection_score))
    closing_score = min(20, max(0, closing_score))
    
    # report
    report = {
        "Engagement": {
            "score": engagement_score,
            "comment": " ".join(engagement_comments)
        },
        "Clarity": {
            "score": clarity_score,
            "comment": " ".join(clarity_comments)
        },
        "Product Knowledge": {
            "score": product_score,
            "comment": " ".join(product_comments)
        },
        "Listening Skills": {
            "score": listening_score,
            "comment": " ".join(listening_comments)
        },
        "Handling Objections": {
            "score": objection_score,
            "comment": " ".join(objection_comments)
        },
        "Closing": {
            "score": closing_score,
            "comment": " ".join(closing_comments)
        }
    }
    return report

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  
    app.run(host="0.0.0.0", port=port)